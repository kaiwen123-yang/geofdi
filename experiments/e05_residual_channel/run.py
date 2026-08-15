#!/usr/bin/env python3
"""e05 — residual R+ channel (generalized-momentum observer on the go2_urdf_sym world):
 (a) FAR under the S2 symmetric drift for rplus_resid (tau_cmd / tau_meas) vs rplus_track;
 (b) the two S2 R+-blind cells (HFE gain, KFE bias): rplus_resid detection + R- e-CUSUM reference;
 (c) lateral payload vs single-leg fault: joint reading of R- and the per-leg residual pattern.

    python experiments/e05_residual_channel/run.py --stage a|b|c|all [--run-id ID] [--quick]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.detect.monitors import (MirrorMonitor, calibrate_ecusum_threshold, conformal_pvalues, ecusum, eprocess_alarm,
                                    tracking_scores)
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry import JOINTS, LEGS, z_channel_names

EXP_NAME = "e05_residual_channel"
REPO = Path(__file__).resolve().parents[2]
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]
BASE_COLS = ["res_base_fx", "res_base_fy", "res_base_fz", "res_base_mx", "res_base_my", "res_base_mz"]
_DYN = {}


def _dyn(ocfg):
    key = (ocfg["armature"], ocfg["damping"], ocfg["frictionloss"])
    if key not in _DYN:
        _DYN[key] = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    return _DYN[key]


def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        s[k] = {**s.get("controller", {}), **v} if k == "controller" else v
    return s


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


def _onset_time(K_cal, drop_first, period=0.5):
    return (K_cal + drop_first) * period


def _run(sim_cfg, n_cycles, N, drop_first, ocfg, torques=("tau_cmd",)):
    """One rollout: registered Z (K,d,N), q_ref cycles, residual cycles per torque source, manifest, channel names."""
    cfg = SimConfig(**sim_cfg); period = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (n_cycles + drop_first + 2) * period
    df, man = rollout(cfg)
    chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first)
    Zq, _ = register_cycles(df, QREF, N=N, drop_first=drop_first)
    dyn = _dyn(ocfg); Zr = {}; Zb = {}
    for tq in torques:
        r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=ocfg["cutoff_hz"], torque=tq)
        Zr[tq], _ = registered_residuals(df, r[:, 6:], N=N, drop_first=drop_first)
        dfb = df.copy()
        for i, c in enumerate(BASE_COLS):
            dfb[c] = r[:, i]
        Zb[tq], _ = register_cycles(dfb, BASE_COLS, N=N, drop_first=drop_first)      # floating-base rows (body-frame force, moment)
    K = min(n_cycles, Z.shape[0], Zq.shape[0], *[z.shape[0] for z in Zr.values()])
    _run.last_base = {k: v[:K] for k, v in Zb.items()}
    return Z[:K], Zq[:K], {k: v[:K] for k, v in Zr.items()}, man, chans


# ------------------------------------------------------------------------------------ workers
def _rep_a(sim_cfg, K_cal, K_mon, N, drop_first, ocfg, alpha):
    Z, Zq, Zr, man, chans = _run(sim_cfg, K_cal + K_mon, N, drop_first, ocfg, torques=("tau_cmd", "tau_meas"))
    out = {}
    s_track = tracking_scores(Z, chans, Zq); out["rplus_track"] = conformal_pvalues(s_track[:K_cal], s_track[K_cal:])
    for tq in ("tau_cmd", "tau_meas"):
        s = residual_scores(Zr[tq]); out[f"rplus_resid_{tq}"] = conformal_pvalues(s[:K_cal], s[K_cal:])
        out[f"_score_{tq}"] = s
    out["_track"] = s_track
    return out


def _rep_b(sim_cfg, K_cal, K_post, N, drop_first, ocfg, alpha, M, window, seed):
    Z, Zq, Zr, man, chans = _run(sim_cfg, K_cal + K_post, N, drop_first, ocfg, torques=(ocfg["torque"],))
    rep = C2Rep(man); out = {}
    for stat in ("paired_energy", "energy_distance"):
        mm = MirrorMonitor(rep, window=window, M=M, statistic=stat, alpha=alpha)
        ps = mm.window_pvalues(Z, seed=seed); out[f"_p_{stat}"] = ps
    s_track = tracking_scores(Z, chans, Zq); pt = conformal_pvalues(s_track[:K_cal], s_track[K_cal:])
    E, al = eprocess_alarm(pt, alpha, start=0); out["rplus_track_delay"] = None if al is None else al + 1
    s = residual_scores(Zr[ocfg["torque"]]); pr = conformal_pvalues(s[:K_cal], s[K_cal:])
    E, al = eprocess_alarm(pr, alpha, start=0); out["rplus_resid_delay"] = None if al is None else al + 1
    out["_p_resid"] = pr; out["_p_track"] = pt
    # per-leg residual mean deviation (post - cal), for the isolation reading
    sl = residual_scores(Zr[ocfg["torque"]], per_leg=True)
    out["leg_dev"] = ((sl[K_cal:].mean(0) - sl[:K_cal].mean(0)) / (sl[:K_cal].std(0) + 1e-12)).tolist()
    out["_Zr"] = Zr[ocfg["torque"]]; out["_Zb"] = _run.last_base[ocfg["torque"]]
    return out


def _rep_c(sim_cfg, K_cal, K_post, N, drop_first, ocfg, alpha, M, window, seed):
    out = _rep_b(sim_cfg, K_cal, K_post, N, drop_first, ocfg, alpha, M, window, seed)
    R = out.pop("_Zr")                                            # (K, 12, N) joint residual cycles (from the same rollout)
    Bz = out.pop("_Zb")                                           # (K, 6, N) base rows: body-frame force (N) and moment (N m)
    bshift = Bz[K_cal:].mean(axis=(0, 2)) - Bz[:K_cal].mean(axis=(0, 2)); bscale = Bz[:K_cal].std(axis=(0, 2)) + 1e-9
    out["base_shift"] = bshift.tolist(); out["base_shift_z"] = (bshift / bscale).tolist()
    sb = np.sqrt((Bz ** 2).mean(axis=(1, 2)))                    # per-cycle base-residual magnitude score -> conformal
    pb = conformal_pvalues(sb[:K_cal], sb[K_cal:]); E, al = eprocess_alarm(pb, alpha, start=0)
    out["rplus_base_delay"] = None if al is None else al + 1
    # per-(leg, joint) mean residual shift and its symmetric / antisymmetric decomposition
    shift = R[K_cal:].mean(axis=(0, 2)) - R[:K_cal].mean(axis=(0, 2))       # (12,) mean residual change per joint
    scale = R[:K_cal].std(axis=(0, 2)) + 1e-9
    out["joint_shift"] = (shift / scale).tolist()
    # energy share of the most affected leg (1/4 = evenly spread, 1 = single leg)
    per_leg = np.array([np.sum((shift[3 * i:3 * i + 3] / scale[3 * i:3 * i + 3]) ** 2) for i in range(4)])
    out["max_leg_share"] = float(per_leg.max() / (per_leg.sum() + 1e-12))
    # mirror decomposition of the shift: left/right pairs
    perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]; sgn = np.tile([-1, 1, 1], 4)
    ms = shift[perm] * sgn
    out["anti_share"] = float(np.sum((shift - ms) ** 2) / (np.sum((shift - ms) ** 2) + np.sum((shift + ms) ** 2) + 1e-12))
    return out


# ------------------------------------------------------------------------------------ stages
def stage_a(cfg, res_dir, quick=False):
    sa = cfg["stage_a"]; R = 8 if quick else sa["R"]; K_mon = 40 if quick else sa["K_mon"]; K_cal = sa["K_cal"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; alpha = cfg["detect"]["alpha"]; oc = cfg["observer"]
    rows = []
    for cond, nuis in (("nominal", []), (f"drift_sym_{sa['drift_symmetric']['magnitude']}",
                                        [dict(type="drift_symmetric", magnitude=sa["drift_symmetric"]["magnitude"], params={"tau_s": sa["drift_symmetric"]["tau_s"]})])):
        args = [(_sim(cfg, sa["seed_base"] + r, nuisance=nuis), K_cal, K_mon, N, df0, oc, alpha) for r in range(R)]
        res = pmap(_rep_a, args, cfg["workers"])
        for det in ("rplus_track", "rplus_resid_tau_cmd", "rplus_resid_tau_meas"):
            P = np.concatenate([r[det] for r in res]); n = len(P); k = int(np.sum(P <= alpha)); band = nominal_band(alpha, n)
            alarm = float(np.mean([eprocess_alarm(r[det], alpha)[1] is not None for r in res]))
            rows.append({"condition": cond, "detector": det, "R": R, "n_cycles": n, "far_per_cycle": k / n, "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1],
                         "band_lo": band[0], "band_hi": band[1], "in_band": bool(band[0] <= k / n <= band[1]), "alarm_fraction": alarm})
        print(f"  [a] {cond} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e05a_far.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); dets = ["rplus_track", "rplus_resid_tau_cmd", "rplus_resid_tau_meas"]
    for i, cond in enumerate(tab.condition.unique()):
        sub = tab[tab.condition == cond]
        ax.bar(np.arange(3) + 0.35 * i - 0.17, sub.far_per_cycle, width=0.33, label=cond,
               yerr=[sub.far_per_cycle - sub.ci_lo, sub.ci_hi - sub.far_per_cycle], capsize=3)
    ax.axhline(alpha, color="k", ls="--", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(dets, fontsize=8); ax.set_ylabel("per-cycle FAR (conformal p <= 0.05)")
    ax.legend(fontsize=8); ax.set_title("e05a — R+ channels under the S2 symmetric torque/friction drift", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e05a_far.png", dpi=150); plt.close(fig)
    g = lambda c, d, col: float(tab[(tab.condition == c) & (tab.detector == d)][col].iloc[0])
    dcond = [c for c in tab.condition.unique() if c != "nominal"][0]
    ok_meas = bool(tab[(tab.condition == dcond) & (tab.detector == "rplus_resid_tau_meas")].in_band.iloc[0])
    ok_cmd = bool(tab[(tab.condition == dcond) & (tab.detector == "rplus_resid_tau_cmd")].in_band.iloc[0])
    _conclude(res_dir, f"[e05a] {'PASS' if ok_meas else 'FAIL'}(tau_meas) / {'PASS' if ok_cmd else 'FAIL'}(tau_cmd): per-cycle FAR under {dcond}: "
              f"rplus_track {g(dcond,'rplus_track','far_per_cycle'):.3f}, rplus_resid(tau_cmd) {g(dcond,'rplus_resid_tau_cmd','far_per_cycle'):.3f}, "
              f"rplus_resid(tau_meas) {g(dcond,'rplus_resid_tau_meas','far_per_cycle'):.3f} (band [{g(dcond,'rplus_track','band_lo'):.3f},{g(dcond,'rplus_track','band_hi'):.3f}]); "
              f"nominal: track {g('nominal','rplus_track','far_per_cycle'):.3f}, resid cmd {g('nominal','rplus_resid_tau_cmd','far_per_cycle'):.3f}, resid meas {g('nominal','rplus_resid_tau_meas','far_per_cycle'):.3f}. "
              f"NOTE drift_symmetric scales the APPLIED torque by 1+g_t (k_t/temperature drift): a residual driven by the commanded torque sees it as an unmodelled symmetric torque by construction.")
    return {"pass": ok_meas}


def stage_b(cfg, res_dir, quick=False):
    sb = cfg["stage_b"]; R = 6 if quick else sb["R"]; K_cal, K_post = sb["K_cal"], sb["K_post"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha, M, window = det["alpha"], det["M"], det["window_rminus"]; oc = cfg["observer"]
    t_on = _onset_time(K_cal, df0)
    cells = [("actuator_gain", k, "HFE", k - 1.0) for k in (sb["hfe_gain"][:1] if quick else sb["hfe_gain"])] + \
            [("actuator_bias", b, "KFE", b) for b in (sb["kfe_bias"][:1] if quick else sb["kfe_bias"])]
    rows = []; nom_p = {"paired_energy": [], "energy_distance": []}; results = {}
    for ftype, mag, joint, magnitude in cells:
        fault = dict(type=ftype, t_onset=t_on, leg=sb["leg"], joint=joint, magnitude=float(magnitude))
        args = [(_sim(cfg, sb["seed_base"] + r, faults=[fault]), K_cal, K_post, N, df0, oc, alpha, M, window, sb["seed_base"] + 90000 + r) for r in range(R)]
        res = pmap(_rep_b, args, cfg["workers"])
        for r in res:
            r.pop("_Zr", None); r.pop("_Zb", None)
        results[(ftype, mag, joint)] = res
        for stat in nom_p:
            nom_p[stat] += [r[f"_p_{stat}"][:K_cal // window] for r in res]
        print(f"  [b] {ftype} {mag} {joint} done", flush=True)
    h = {stat: calibrate_ecusum_threshold(nom_p[stat], det["ecusum_horizon_windows"], alpha, det["ecusum_boot"], np.random.default_rng(5)) for stat in nom_p}
    for (ftype, mag, joint), res in results.items():
        for dname in ("rplus_resid", "rplus_track"):
            dl = np.array([np.nan if r[f"{dname}_delay"] is None else r[f"{dname}_delay"] for r in res], dtype=float)
            rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": dname, "R": R, "det_rate_100": float(np.mean(dl <= K_post)),
                         "det_rate_20": float(np.mean(dl <= 20)), "delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                         "delay_q90": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan})
        for stat in nom_p:
            dl = []
            for r in res:
                S, al = ecusum(r[f"_p_{stat}"], h[stat], start=K_cal // window)
                dl.append(np.nan if al is None else (al - K_cal // window + 1) * window)
            dl = np.array(dl, dtype=float)
            rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": f"Rminus_{stat}_ecusum", "R": R, "det_rate_100": float(np.mean(dl <= K_post)),
                         "det_rate_20": float(np.mean(dl <= 20)), "delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                         "delay_q90": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan})
        legdev = np.mean([r["leg_dev"] for r in res], axis=0)
        rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": "resid_leg_dev(LF,RF,LH,RH)", "R": R, "det_rate_100": np.nan, "det_rate_20": np.nan,
                     "delay_median": np.nan, "delay_q90": np.nan, "leg_dev": np.round(legdev, 2).tolist(), "argmax_leg": LEGS[int(np.argmax(legdev))]})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e05b_power.csv", index=False)
    pd.DataFrame([{"stat": k, "h": v} for k, v in h.items()]).to_csv(res_dir / "e05b_ecusum_thresholds.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, ftype, xl in ((axes[0], "actuator_gain", "gain kappa (LF-HFE)"), (axes[1], "actuator_bias", "bias [N m] (LF-KFE)")):
        for dname, mk in (("rplus_resid", "o-"), ("rplus_track", "s--"), ("Rminus_paired_energy_ecusum", "^-.")):
            sub = tab[(tab.fault == ftype) & (tab.detector == dname)].sort_values("magnitude")
            ax.plot(sub.magnitude, sub.det_rate_100, mk, label=dname)
        ax.set_xlabel(xl); ax.set_ylabel("detection rate within 100 cycles"); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
        if ftype == "actuator_gain":
            ax.invert_xaxis()
    axes[0].legend(fontsize=7); fig.suptitle("e05b — the two S2 R+-blind cells with the residual channel", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e05b_power.png", dpi=150); plt.close(fig)
    ok = bool((tab[tab.detector == "rplus_resid"].det_rate_100 >= 0.9).all())
    _conclude(res_dir, f"[e05b] {'PASS' if ok else 'FAIL'}: rplus_resid det100 >= 0.9 in all cells: {ok} | "
              + "; ".join(f"{r['fault']} {r['magnitude']} {r['joint']}: {r['detector']} det100={r['det_rate_100']:.2f} det20={r['det_rate_20']:.2f} med={r['delay_median']}" for r in tab.to_dict('records') if isinstance(r["detector"], str) and not r["detector"].startswith("resid_leg"))
              + " | leg_dev: " + "; ".join(f"{r['fault']} {r['magnitude']} {r['joint']}: {r['leg_dev']} -> {r['argmax_leg']}" for r in tab.to_dict("records") if str(r["detector"]).startswith("resid_leg")))
    return {"pass": ok, "h": h}


def stage_c(cfg, res_dir, quick=False):
    sc = cfg["stage_c"]; R = 6 if quick else sc["R"]; K_cal, K_post = sc["K_cal"], sc["K_post"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha, M, window = det["alpha"], det["M"], det["window_rminus"]; oc = cfg["observer"]
    t_on = _onset_time(K_cal, df0)
    conds = [(f"payload_asym_{m}kg", dict(nuisance=[dict(type="payload_asymmetric", magnitude=m, params={"offset_y": 0.05}, t_onset=t_on)])) for m in sc["payload_asymmetric"]]
    conds += [(f"{f['type']}_{f['leg']}-{f['joint']}_{f['magnitude']}", dict(faults=[dict(f, t_onset=t_on)])) for f in sc["compare_faults"]]
    if quick:
        conds = conds[:1] + conds[-1:]
    rows = []
    for name, over in conds:
        args = [(_sim(cfg, sc["seed_base"] + r, **over), K_cal, K_post, N, df0, oc, alpha, M, window, sc["seed_base"] + 90000 + r) for r in range(R)]
        res = pmap(_rep_c, args, cfg["workers"])
        pw = np.concatenate([r["_p_paired_energy"][K_cal // window:] for r in res])
        rminus_e = float(np.mean([eprocess_alarm(r["_p_paired_energy"], alpha, start=K_cal // window)[1] is not None for r in res]))
        dl = np.array([np.nan if r["rplus_resid_delay"] is None else r["rplus_resid_delay"] for r in res], dtype=float)
        dlb = np.array([np.nan if r["rplus_base_delay"] is None else r["rplus_base_delay"] for r in res], dtype=float)
        js = np.mean([r["joint_shift"] for r in res], axis=0); bs = np.mean([r["base_shift"] for r in res], axis=0); bz = np.mean([r["base_shift_z"] for r in res], axis=0)
        rows.append({"condition": name, "R": R, "Rminus_alarm_frac": rminus_e, "Rminus_window_rejection": float(np.mean(pw <= alpha)),
                     "rplus_resid_det100": float(np.mean(dl <= K_post)), "rplus_resid_delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                     "rplus_base_det100": float(np.mean(dlb <= K_post)), "rplus_base_delay_median": float(np.nanmedian(dlb)) if np.isfinite(dlb).any() else np.nan,
                     "max_leg_share": float(np.mean([r["max_leg_share"] for r in res])), "anti_share": float(np.mean([r["anti_share"] for r in res])),
                     "joint_shift_z": np.round(js, 2).tolist(), "base_shift_(fx,fy,fz,mx,my,mz)": np.round(bs, 3).tolist(), "base_shift_z": np.round(bz, 2).tolist()})
        print(f"  [c] {name} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e05c_joint_reading.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for _, r in tab.iterrows():
        ax.scatter(r.max_leg_share, r.anti_share, s=60, label=r.condition)
    ax.set_xlabel("energy share of the most affected leg (residual mean shift; 0.25 = even, 1 = single leg)"); ax.set_ylabel("antisymmetric share of the residual shift")
    ax.set_xlim(0.2, 1.02); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title("e05c — lateral payload vs single-leg fault in the residual R+ pattern", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e05c_joint_reading.png", dpi=150); plt.close(fig)
    pay = tab[tab.condition.str.startswith("payload")]; flt = tab[~tab.condition.str.startswith("payload")]
    sep = bool((pay.max_leg_share.max() < flt.max_leg_share.min()) or ((pay.rplus_resid_det100 <= 0.1).all() and (pay.rplus_base_det100 >= 0.9).all() and (flt.rplus_resid_det100 >= 0.9).all())) if len(pay) and len(flt) else False
    _conclude(res_dir, f"[e05c] joint reading — payload: R- fires, joint-residual quiet, BASE-residual rows carry it; single-leg fault: R- fires, joint residual on one leg. Separable by pattern (payload max-leg share < every fault's OR joint det100 payload==0 & base det100>0): {sep} | "
              + "; ".join(f"{r['condition']}: R- alarm {r['Rminus_alarm_frac']:.2f}, joint-resid det100 {r['rplus_resid_det100']:.2f} (med {r['rplus_resid_delay_median']}), base-resid det100 {r['rplus_base_det100']:.2f} (med {r['rplus_base_delay_median']}), max-leg share {r['max_leg_share']:.2f}, anti share {r['anti_share']:.2f}, joint shift z {r['joint_shift_z']}, base shift {r['base_shift_(fx,fy,fz,mx,my,mz)']}" for r in tab.to_dict("records")))
    return {"pass": sep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["a", "b", "c"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"a": stage_a, "b": stage_b, "c": stage_c}[s](cfg, res_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E05 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
