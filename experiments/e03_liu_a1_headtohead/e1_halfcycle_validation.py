#!/usr/bin/env python3
"""E1 — half-cycle sequential layer validation on the Go2 simulator (Sprint 7 Block E1).

Streams: go2_urdf_sym trot in place; 20 warm-up + K_cal = 400 calibration cycles + onset + K_post monitored cycles.
Faults: LF-KFE gain kappa in {0.7, 0.9} (R runs each); nominal streams (R) for FAR / ARL0 (half of them calibrate the
CUSUM thresholds, the other half evaluate). Detectors on the half-cycle mirror score (detect.sequential): e-process,
e-CUSUM, conformal-CUSUM, plus the Sprint-2/4 reference (5-cycle-window flip test + e-process / calibrated e-CUSUM).
Targets: median R- delay <= 2 cycles at kappa 0.7; nominal false-alarm probability over the horizon <= alpha
(equivalently ARL0 >= horizon/alpha-ish; reported as ARL0 censored mean and P(alarm within K_post)).

    python experiments/e03_liu_a1_headtohead/e1_halfcycle_validation.py [--run-id ID] [--quick] [--workers N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from geofdi.detect.monitors import MirrorMonitor, calibrate_ecusum_threshold, ecusum, eprocess_alarm
from geofdi.detect.sequential import ConformalCusum, ECusum, EProcess, calibrate_threshold, calibration_scale, half_cycles, mirror_scores
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, pmap
from geofdi.sim.telemetry import z_channel_names

REPO = Path(__file__).resolve().parents[2]
SIM = dict(model="go2_urdf_sym", gait="trot", speed=0.0, terrain="flat", ctrl_dt=0.005, sim_dt=0.0025,
           controller={"period_s": 0.5, "kp": [80, 80, 80], "kd": [2.0, 2.0, 2.0], "lift_kfe": 0.45, "lift_hfe": 0.20, "stab_k_wz": 0.2},
           noise={"actuator_std": 0.02, "encoder_pos_std": 0.002, "encoder_vel_std": 0.03, "torque_meas_std": 0.2, "imu_acc_std": 0.1, "imu_gyro_std": 0.01})
DROP = 20; N = 64; ALPHA = 0.05; M = 512; WINDOW = 5


def _run(seed, K_cal, K_post, fault):
    cfg = SimConfig(**SIM); cfg.seed = int(seed); cfg.duration_s = (DROP + K_cal + K_post + 2) * 0.5
    if fault is not None:
        cfg.faults = [dict(fault, t_onset=(DROP + K_cal) * 0.5)]
    df, man = rollout(cfg); chans = z_channel_names(man); rep = C2Rep(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=DROP); Z = Z[:K_cal + K_post]
    H = half_cycles(Z); scale = calibration_scale(H[:2 * K_cal], rep)
    s = mirror_scores(H, rep, scale)                                     # index j -> half-cycle j (j >= 1)
    s_cal = s[:2 * K_cal - 1]; s_mon = s[2 * K_cal - 1:]                  # monitored half-cycles start at 2*K_cal
    # reference: 5-cycle-window flip test p-values (S2 protocol) on the whole stream
    mm = MirrorMonitor(rep, window=WINDOW, M=M, statistic="paired_energy", alpha=ALPHA)
    pw = mm.window_pvalues(Z, seed=int(seed))
    return {"s_cal": s_cal.astype(np.float32), "s_mon": s_mon.astype(np.float32), "pw": pw, "K": int(Z.shape[0])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args(); out = REPO / "results" / "e03_liu_a1_headtohead" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    R = 6 if a.quick else 50; K_cal = 100 if a.quick else 400; K_post = 40 if a.quick else 60; K_post_nom = 100 if a.quick else 400
    conds = [("nominal", None, K_post_nom), ("gain_LF-KFE_0.7", dict(type="actuator_gain", leg="LF", joint="KFE", magnitude=-0.3), K_post),
             ("gain_LF-KFE_0.9", dict(type="actuator_gain", leg="LF", joint="KFE", magnitude=-0.1), K_post)]
    res = {}
    for ci, (name, fault, kp) in enumerate(conds):
        res[name] = pmap(_run, [(41000 + 100 * ci + r, K_cal, kp, fault) for r in range(R)], a.workers); print(f"  [E1] {name} done", flush=True)
    # thresholds from the first half of the nominal runs (held-out monitored segments), evaluation on the second half
    nom = res["nominal"]; n_thr = len(nom) // 2
    horizon_half_nom = 2 * K_post_nom                                     # the CUSUM must control FAR over the NOMINAL horizon
    from geofdi.detect.sequential import ConformalCusum as _CC
    dets = {"eprocess": EProcess(ALPHA), "eprocess_decim": EProcess(ALPHA), "ecusum": ECusum(ALPHA), "conformal_cusum": ConformalCusum(ALPHA)}
    for k, d in dets.items():
        if k in ("ecusum", "conformal_cusum"):
            d.calibrate(np.concatenate([r["s_cal"] for r in nom[:n_thr]]))
            d.h = calibrate_threshold(d, [r["s_mon"] for r in nom[:n_thr]], horizon=horizon_half_nom, far=ALPHA, n_boot=1000, rng=np.random.default_rng(1))
    w0 = K_cal // WINDOW
    h_ref = calibrate_ecusum_threshold([r["pw"][:w0] for r in nom[:n_thr]], K_post // WINDOW, far=ALPHA, n_boot=1000, rng=np.random.default_rng(2))
    rows = []
    for name in res:
        runs = res[name] if name != "nominal" else nom[n_thr:]
        for k, d in dets.items():
            delays = []
            for r in runs:
                d.calibrate(r["s_cal"][::2] if k == "eprocess_decim" else r["s_cal"])
                s_mon = r["s_mon"][::2] if k == "eprocess_decim" else r["s_mon"]
                S, al = d.run(s_mon)
                step = 1.0 if k == "eprocess_decim" else 0.5                     # decimated: one score per cycle
                delays.append(np.nan if al is None else (al + 1) * step)
            dl = np.array(delays); horizon = (K_post_nom if name == "nominal" else K_post)
            rows.append({"condition": name, "detector": f"halfcycle_{k}", "R": len(runs), "alarm_frac": float(np.mean(~np.isnan(dl))), "det_within_horizon": float(np.mean(dl <= horizon)),
                         "delay_median_cycles": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan, "delay_q90_cycles": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan,
                         "det2": float(np.mean(dl <= 2)), "arl_censored_cycles": float(np.nanmean(np.where(np.isnan(dl), horizon, dl))), "h": getattr(d, "h", np.nan)})
        for k in ("eproc", "ecusum"):
            delays = []
            for r in runs:
                pw = r["pw"]
                if k == "eproc":
                    E, al = eprocess_alarm(pw, ALPHA, start=w0)
                else:
                    S, al = ecusum(pw, h_ref, start=w0)
                delays.append(np.nan if al is None else (al - w0 + 1) * WINDOW)
            dl = np.array(delays); horizon = (K_post_nom if name == "nominal" else K_post)
            rows.append({"condition": name, "detector": f"window5_{k}", "R": len(runs), "alarm_frac": float(np.mean(~np.isnan(dl))), "det_within_horizon": float(np.mean(dl <= horizon)),
                         "delay_median_cycles": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan, "delay_q90_cycles": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan,
                         "det2": float(np.mean(dl <= 2)), "arl_censored_cycles": float(np.nanmean(np.where(np.isnan(dl), horizon, dl))), "h": h_ref if k == "ecusum" else np.nan})
    tab = pd.DataFrame(rows); tab.to_csv(out / "e1_halfcycle_validation.csv", index=False)
    pd.set_option("display.width", 250)
    def g(c, d, col): return float(tab[(tab.condition == c) & (tab.detector == d)][col].iloc[0])
    line = ("[E1] half-cycle mirror score + conformal (K_cal=%d): kappa 0.7 median delay [cycles] eprocess(adjacent) %.1f / eprocess(decimated) %.1f / e-CUSUM %.1f / conformal-CUSUM %.1f (det<=2 cycles %.2f/%s/%.2f/%.2f); kappa 0.9: %.1f / %s / %.1f / %.1f; "
            "nominal false-alarm within %d cycles: %.3f / %.3f / %.3f (ARL0 censored %.0f / %.0f / %.0f cycles) | reference 5-cycle-window flip test: kappa 0.7 delay eproc %.0f / e-CUSUM %.0f, nominal FA %.3f / %.3f"
            % (K_cal, g("gain_LF-KFE_0.7", "halfcycle_eprocess", "delay_median_cycles"), g("gain_LF-KFE_0.7", "halfcycle_eprocess_decim", "delay_median_cycles"), g("gain_LF-KFE_0.7", "halfcycle_ecusum", "delay_median_cycles"), g("gain_LF-KFE_0.7", "halfcycle_conformal_cusum", "delay_median_cycles"),
               g("gain_LF-KFE_0.7", "halfcycle_eprocess", "det2"), f"{g('gain_LF-KFE_0.7','halfcycle_eprocess_decim','det2'):.2f}", g("gain_LF-KFE_0.7", "halfcycle_ecusum", "det2"), g("gain_LF-KFE_0.7", "halfcycle_conformal_cusum", "det2"),
               g("gain_LF-KFE_0.9", "halfcycle_eprocess", "delay_median_cycles"), f"{g('gain_LF-KFE_0.9','halfcycle_eprocess_decim','delay_median_cycles'):.1f}", g("gain_LF-KFE_0.9", "halfcycle_ecusum", "delay_median_cycles"), g("gain_LF-KFE_0.9", "halfcycle_conformal_cusum", "delay_median_cycles"),
               K_post_nom, g("nominal", "halfcycle_eprocess", "alarm_frac"), g("nominal", "halfcycle_ecusum", "alarm_frac"), g("nominal", "halfcycle_conformal_cusum", "alarm_frac"),
               g("nominal", "halfcycle_eprocess", "arl_censored_cycles"), g("nominal", "halfcycle_ecusum", "arl_censored_cycles"), g("nominal", "halfcycle_conformal_cusum", "arl_censored_cycles"),
               g("gain_LF-KFE_0.7", "window5_eproc", "delay_median_cycles"), g("gain_LF-KFE_0.7", "window5_ecusum", "delay_median_cycles"), g("nominal", "window5_eproc", "alarm_frac"), g("nominal", "window5_ecusum", "alarm_frac")))
    print(line); (out / "conclusions.txt").open("a").write(line + "\n"); print(tab.to_string())


if __name__ == "__main__":
    main()
