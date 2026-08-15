#!/usr/bin/env python3
"""Gate-1 (A2) estimator rehearsal (Sprint 7 Block W4): precision and honesty of the mirrored-command gap estimator on
sessions with a KNOWN controller asymmetry, for the rolling M1 and the trotting Go2.

For each condition (nominal; single-side HIP kp x1.02; single-side wheel-rate x1.02 [M1] / LF-HFE kp x1.05 [Go2]):
  population value  eps_pop_j = |mean over R_pop seeds of the per-session signed mirrored-command mean| (long-run truth of the
                    closed loop's command asymmetry, the quantity Gate 1 is defined on)
  single-session    eps_hat_j from one session (60 blocks / cycles) with its bootstrap CI
  counterfactual    eps_cf_j = |mean(tau_cmd - tau_sym(recorded state))| — the injected offset at the SAME state; the closed
                    loop absorbs gain-type asymmetries (the torque is set by the load), so eps_cf > eps_pop by design.
Gate reading: relative error |eps_hat - eps_pop| / eps_pop on the injected channel (median over sessions) < 0.30, and the
bootstrap CI covers eps_pop in >= 80 % of the sessions.

    python experiments/e01w_rolling_h0/gate1_rehearsal.py [--run-id ID] [--quick]
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

from geofdi.detect.gate1 import gate1_estimate
from geofdi.groups.c2 import C2Rep
from geofdi.phase.estimator import estimate_phase
from geofdi.phase.registration import register_blocks, register_cycles, straight_mask
from geofdi.sim.controller import TrotController, TrotParams
from geofdi.sim.controller_wheeled import RollingController, RollingParams
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.env_m1 import SimConfigM1, rollout_m1
from geofdi.sim.pipeline import pmap
from geofdi.sim.telemetry import z_channel_names as zn_go2
from geofdi.sim.telemetry_m1 import JOINTS as J1, LEGS as L1, z_channel_names as zn_m1

REPO = Path(__file__).resolve().parents[2]


def m1_session(seed, asym):
    cfg = SimConfigM1(model="m1_wheeled_sym", speed=1.0, duration_s=66.0, seed=seed, controller={"ramp_s": 3.0, "asymmetry": asym})
    df, man = rollout_m1(cfg); chans = zn_m1(man); rep = C2Rep(man)
    Z, _ = register_blocks(df, chans, L_s=1.0, N=64, mask=straight_mask(df, warmup_s=6.0), max_blocks=60)
    est = gate1_estimate(Z, rep, chans, rng=np.random.default_rng(seed))
    ctrl = RollingController(RollingParams(speed=1.0, ramp_s=3.0))
    q = df[[f"q_{l}_{j}" for l in L1 for j in J1]].to_numpy(); dq = df[[f"dq_{l}_{j}" for l in L1 for j in J1]].to_numpy(); tau = df[[f"tau_cmd_{l}_{j}" for l in L1 for j in J1]].to_numpy()
    sel = df.t.to_numpy() > 6.0
    tsym = np.array([ctrl.torque(q[k], dq[k], df.t.iloc[k], body={"w_z": df.imu_w_z.iloc[k]})[0] for k in np.where(sel)[0]])
    cf = np.abs((tau[sel] - tsym).mean(axis=0)); names = [f"tau_cmd_{l}_{j}" for l in L1 for j in J1]
    return {"est": est, "cf": dict(zip(names, cf.tolist()))}


def go2_session(seed, asym):
    cfg = SimConfig(model="go2_urdf_sym", speed=0.5, duration_s=45.0, seed=seed, controller={"asymmetry": asym})
    df, man = rollout(cfg); chans = zn_go2(man); rep = C2Rep(man)
    th, info = estimate_phase(df, joint="KFE"); d2 = df.copy(); d2["th"] = th
    Z, _ = register_cycles(d2, chans, N=64, theta_col="th", drop_first=20, drop_last=5)
    Z = Z[:60]
    est = gate1_estimate(Z, rep, chans, rng=np.random.default_rng(seed))
    ctrl = TrotController(TrotParams(**{k: v for k, v in dict(period_s=0.5, kp=(80, 80, 80), kd=(2.0, 2.0, 2.0), lift_kfe=0.45, lift_hfe=0.20, stab_k_wz=0.2, speed=0.5).items()}))
    from geofdi.sim.telemetry import JOINTS as J2, LEGS as L2
    q = df[[f"q_{l}_{j}" for l in L2 for j in J2]].to_numpy(); dq = df[[f"dq_{l}_{j}" for l in L2 for j in J2]].to_numpy(); tau = df[[f"tau_cmd_{l}_{j}" for l in L2 for j in J2]].to_numpy()
    sel = np.where(df.t.to_numpy() > 10.0)[0]
    tsym = np.array([ctrl.torque(q[k], dq[k], df.theta.iloc[k], df.t.iloc[k], body={"w_z": df.imu_w_z.iloc[k]})[0] for k in sel])
    cf = np.abs((tau[sel] - tsym).mean(axis=0)); names = [f"tau_cmd_{l}_{j}" for l in L2 for j in J2]
    return {"est": est, "cf": dict(zip(names, cf.tolist()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    out = REPO / "results" / "e01w_rolling_h0" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    R_pop = 6 if a.quick else 20; R_one = 4 if a.quick else 10
    conds = {"m1": [("nominal", [], None), ("hip_kp_1.02", [{"leg": "LF", "joint": "HIP", "kp_gain": 1.02}], "tau_cmd_LF_HIP"),
                    ("wheel_rate_1.02", [{"leg": "RF", "joint": "WHEEL", "rate_gain": 1.02}], "tau_cmd_RF_WHEEL")],
             "go2": [("nominal", [], None), ("hfe_kp_1.05", [{"leg": "LF", "joint": "HFE", "kp_gain": 1.05}], "tau_cmd_LF_HFE")]}
    rows = []
    for robot, cl in conds.items():
        fn = m1_session if robot == "m1" else go2_session
        for ci, (name, asym, target) in enumerate(cl):
            seeds = [91000 + 100 * ci + r for r in range(R_pop)]
            res = pmap(fn, [(s, asym) for s in seeds], min(22, R_pop))
            chans = list(res[0]["est"]["per_channel"].keys())
            signed = np.array([[r["est"]["per_channel"][c]["signed_mean"] for c in chans] for r in res])     # (R_pop, n_ch)
            pop = np.abs(signed.mean(axis=0)); pop_map = dict(zip(chans, pop))
            for r_i, r in enumerate(res[:R_one]):
                for c in ([target] if target else chans[:1]):
                    e = r["est"]["per_channel"][c]; ep = pop_map[c]
                    lo, hi = abs(e["signed_mean"]) - 1.96 * e["boot_std"], abs(e["signed_mean"]) + 1.96 * e["boot_std"]
                    rows.append({"robot": robot, "condition": name, "channel": c, "session": r_i, "eps_hat": e["eps_hat"], "boot_std": e["boot_std"], "z": e["z"], "eps_pop": ep,
                                 "rel_err": abs(e["eps_hat"] - ep) / max(ep, 1e-9), "ci_covers_pop": bool(lo <= ep <= hi), "eps_counterfactual": r["cf"][c],
                                 "argmax_channel": r["est"]["argmax_channel"], "eps_hat_ctrl": r["est"]["eps_hat_ctrl"]})
            print(f"  [gate1] {robot} {name} done: pop {target}: {pop_map.get(target, float('nan')):.4f}", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(out / "gate1_rehearsal.csv", index=False)
    summ = tab[tab.condition != "nominal"].groupby(["robot", "condition"]).agg(rel_err_median=("rel_err", "median"), rel_err_q90=("rel_err", lambda x: x.quantile(0.9)), coverage=("ci_covers_pop", "mean"),
                                                                                eps_pop=("eps_pop", "first"), eps_cf_mean=("eps_counterfactual", "mean"), z_median=("z", "median")).reset_index()
    nom = tab[tab.condition == "nominal"].groupby("robot").agg(z_median=("z", "median"), eps_hat_mean=("eps_hat", "mean")).reset_index()
    summ.to_csv(out / "gate1_summary.csv", index=False)
    line = "[gate1] " + "; ".join(f"{r.robot} {r.condition}: eps_pop {r.eps_pop:.4f} N m (counterfactual {r.eps_cf_mean:.4f}), single-session rel err median {r.rel_err_median:.2f} q90 {r.rel_err_q90:.2f}, CI coverage {r.coverage:.2f}, z {r.z_median:.1f}" for r in summ.itertuples()) \
           + " | nominal: " + "; ".join(f"{r.robot} z {r.z_median:.1f} eps_hat {r.eps_hat_mean:.4f}" for r in nom.itertuples())
    print(line); (out / "conclusions.txt").open("a").write(line + "\n")


if __name__ == "__main__":
    main()
