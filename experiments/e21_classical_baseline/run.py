#!/usr/bin/env python3
"""e21 — the classical model-based baseline row (Sprint 9 B1).

De Luca-Mattone generalised-momentum observer + chi-square threshold (the method Haddadin et al.'s survey treats as the
standard), evaluated on the SAME rollouts as the GeoFDI detectors so the rows merge into the e07 baseline table:

  classical_fixed      residual quadratic form vs the FIXED chi^2_{k,1-alpha} threshold (the textbook recipe)
  classical_far_matched same statistic, threshold re-calibrated on the run's own nominal cycles to FAR = alpha
  Rminus_ecusum        the GeoFDI mirror channel (FAR-calibrated e-CUSUM), for reference on identical data

Fault grid follows e07/e08 (LF joint, actuator_gain / actuator_bias / friction_scale), plus the e07 nuisance rows
(symmetric drift, symmetric payload) where the classical detector's lack of nuisance invariance should show.

    python experiments/e21_classical_baseline/run.py [--quick]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.baselines.momentum_chi2 import MomentumChi2
from geofdi.detect.evalue import eprocess
from geofdi.detect.monitors import MirrorMonitor
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, pmap
from geofdi.sim.telemetry import z_channel_names

DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
CFG = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())


def _sim(seed, **over):
    s = dict(CFG["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    s.update(over); return s


def _worker(seed, sim_cfg, K_cal, K_post, N, df0, ocfg, alpha, debounce):
    cfg = SimConfig(**sim_cfg); per = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (K_cal + K_post + df0 + 3) * per
    df, man = rollout(cfg)
    t = df["t"].to_numpy(); dt = float(np.median(np.diff(t)))
    t_on = (K_cal + df0) * per
    # --- classical: momentum-observer residual, joint rows
    dyn = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    r = run_observer(df, dyn, dt=dt, cutoff_hz=ocfg["cutoff_hz"], torque=ocfg["torque"])[:, 6:]
    cal = (t >= df0 * per) & (t < t_on); mon = t >= t_on
    horizon = int(K_post * per / dt)                      # the monitoring horizon in samples
    half = int(cal.sum() // 2)
    idx_cal = np.flatnonzero(cal)
    det = MomentumChi2(alpha=alpha, debounce=debounce, horizon=horizon).fit(r[idx_cal[:half]])   # fit on the FIRST half
    sc = det.score(r[mon])
    out = {"cal_exceed_at_fixed_thr": sc["cal_exceed_rate_at_fixed_thr"]}
    for tag in ("fixed", "far_matched"):
        fa = sc[f"first_alarm_{tag}"]
        out[f"classical_{tag}_alarm"] = fa is not None
        out[f"classical_{tag}_delay_cycles"] = (fa * dt / per) if fa is not None else np.nan
        out[f"classical_{tag}_rate"] = sc[f"alarm_rate_{tag}"]
        # nominal false alarm: the same rule on the HELD-OUT second half of the nominal stretch (never used to fit)
        sc_cal = det.score(r[idx_cal[half:]])
        out[f"classical_{tag}_nominal_alarm"] = sc_cal[f"first_alarm_{tag}"] is not None
    # --- GeoFDI R-: window p-values -> e-process
    chans = z_channel_names(man)
    Z, _ = register_cycles(df, chans, N=N, drop_first=df0)
    K = Z.shape[0]
    if K >= K_cal + 10:
        rep = C2Rep(man)
        mm = MirrorMonitor(rep, window=5, M=256, statistic="paired_energy", alpha=alpha)
        pw = mm.window_pvalues(Z[K_cal:], seed=int(seed))
        E, al = eprocess(np.asarray(pw), alpha)
        out["rminus_alarm"] = al is not None
        out["rminus_delay_cycles"] = (al + 1) * 5.0 if al is not None else np.nan
        pw0 = mm.window_pvalues(Z[:K_cal], seed=int(seed) + 7)
        out["rminus_nominal_alarm"] = eprocess(np.asarray(pw0), alpha)[1] is not None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--run-id", default=None); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); workers = a.workers or CFG["workers"]
    pr = CFG["protocol"]; R = 4 if a.quick else pr["R"]
    K_cal, K_post = (20, 30) if a.quick else (pr["K_cal"], pr["K_post"])
    N = CFG["registration"]["N"]; df0 = CFG["registration"]["drop_first"]; oc = CFG["observer"]
    alpha = CFG["detect"]["alpha"]; deb = CFG["detect"]["debounce"]
    res = DATA / "results" / "e21_classical_baseline" / (a.run_id or f"e21-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}")
    res.mkdir(parents=True, exist_ok=True); (res / "config_snapshot.yaml").write_text(yaml.safe_dump(CFG, sort_keys=False))
    cells = []
    for ft, mags in CFG["faults"].items():
        for m in (mags[-1:] if a.quick else mags):
            cells.append((ft, m))
    rows = []
    t_on_cycles = K_cal + df0
    for ft, m in cells:
        magnitude = {"actuator_gain": m - 1.0, "friction_scale": m - 1.0}.get(ft, m)
        fault = dict(type=ft, t_onset=t_on_cycles * float(CFG["sim"]["controller"]["period_s"]), leg=pr["leg"], joint=pr["joint"], magnitude=float(magnitude))
        outs = [o for o in pmap(_worker, [(pr["seed_base"] + 100 * len(rows) + i, _sim(pr["seed_base"] + i, faults=[fault]), K_cal, K_post, N, df0, oc, alpha, deb) for i in range(R)], workers) if o]
        for dname in ("classical_fixed", "classical_far_matched", "rminus"):
            key = f"{dname}_alarm" if dname != "rminus" else "rminus_alarm"
            if key not in outs[0]:
                continue
            det100 = float(np.mean([o[key] for o in outs]))
            dly = np.array([o[f"{dname}_delay_cycles"] for o in outs], float)
            far = float(np.mean([o[f"{dname}_nominal_alarm"] for o in outs]))
            rows.append({"fault": ft, "magnitude": m, "joint": pr["joint"], "detector": dname, "R": len(outs),
                         "det100": det100, "delay_median_cycles": float(np.nanmedian(dly)),
                         "nominal_alarm_rate": far, "ci_lo": binom_ci(int(det100 * len(outs)), len(outs))[0],
                         "ci_hi": binom_ci(int(det100 * len(outs)), len(outs))[1]})
        print(f"[e21] {ft} {m}: " + "; ".join(f"{r['detector']} det {r['det100']:.2f} (FAR {r['nominal_alarm_rate']:.2f}, delay {r['delay_median_cycles']:.0f})" for r in rows[-3:]), flush=True)
    # nuisance rows
    for nz in CFG["nuisance"]:
        spec = dict(nz); nm = spec.pop("name")
        outs = [o for o in pmap(_worker, [(pr["seed_base"] + 5000 + i, _sim(pr["seed_base"] + i, nuisance=[spec]), K_cal, K_post, N, df0, oc, alpha, deb) for i in range(R)], workers) if o]
        for dname in ("classical_fixed", "classical_far_matched", "rminus"):
            key = f"{dname}_alarm"
            if key not in outs[0]:
                continue
            rows.append({"fault": f"nuisance:{nm}", "magnitude": np.nan, "joint": "-", "detector": dname, "R": len(outs),
                         "det100": float(np.mean([o[key] for o in outs])), "delay_median_cycles": np.nan,
                         "nominal_alarm_rate": float(np.mean([o[f"{dname}_nominal_alarm"] for o in outs])), "ci_lo": np.nan, "ci_hi": np.nan})
        print(f"[e21] nuisance {nm}: " + "; ".join(f"{r['detector']} alarm {r['det100']:.2f}" for r in rows[-3:]), flush=True)
    T = pd.DataFrame(rows); T.to_csv(res / "e21_classical_baseline.csv", index=False)
    cf = T[(T.detector == "classical_fixed") & (~T.fault.str.startswith("nuisance"))]
    cm = T[(T.detector == "classical_far_matched") & (~T.fault.str.startswith("nuisance"))]
    rm = T[(T.detector == "rminus") & (~T.fault.str.startswith("nuisance"))]
    nz = T[T.fault.str.startswith("nuisance")]
    line = (f"[e21] classical momentum-observer chi2 vs GeoFDI R- on identical rollouts. Nominal alarm rate (should be <= alpha={alpha}): "
            f"classical_fixed {cf.nominal_alarm_rate.mean():.2f}, classical_far_matched {cm.nominal_alarm_rate.mean():.2f}, R- {rm.nominal_alarm_rate.mean():.2f}. "
            f"Mean detection over the fault grid: classical_fixed {cf.det100.mean():.2f}, classical_far_matched {cm.det100.mean():.2f}, R- {rm.det100.mean():.2f}. "
            f"Nuisance alarm (symmetric, must stay silent): " + "; ".join(f"{r.detector} {r.det100:.2f}" for _, r in nz.iterrows()))
    (res / "conclusions.txt").write_text(line + "\n"); print(line); print(f"[e21] results -> {res}")


if __name__ == "__main__":
    main()
