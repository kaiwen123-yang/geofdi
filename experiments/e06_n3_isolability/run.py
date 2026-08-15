#!/usr/bin/env python3
"""e06 — N3 isolability certificate (Block L1).

  stage iso : fault runs (5 joints x {gain, bias, friction} x 2 magnitudes x R) on go2_urdf_sym; residuals from the
              analytic momentum observer and from the DeLaN models (full + degraded ladder); per run: signature
              dictionary from the run's own nominal cycles, beta_op^2 per model from the nominal residual profiles,
              nearest-subspace classification of the post-onset top direction, DK certificate -> confusion matrices,
              certificate-vs-outcome agreement, isolation accuracy vs beta^2/beta^2_threshold curve.
  stage weld: welded trunk (leg = 3-dof arm, LF): nominal DeLaN-weld training + the same pipeline; side-by-side table.

    python experiments/e06_n3_isolability/run.py --stage iso|weld|all [--run-id ID] [--quick]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.detect.rplus import registered_residuals
from geofdi.dynamics.delan import DeLaNQuadruped, contact_torques_all, delan_residuals, leg_arrays, train_leg, beta_hat
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.isolation.n3 import TYPES, beta_op2, build_dictionary, dk_certificate, nearest_class, principal_angle_matrix, top_direction
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import pmap
from geofdi.sim.telemetry import JOINTS, LEGS

EXP_NAME = "e06_n3_isolability"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
JOINT_NAMES = [f"{l}-{j}" for l in LEGS for j in JOINTS]
TAU_COLS = [f"tau_cmd_{l}_{j}" for l in LEGS for j in JOINTS]; DQ_COLS = [f"dq_{l}_{j}" for l in LEGS for j in JOINTS]
_MODELS = {}


def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        s[k] = {**s.get("controller", {}), **v} if k == "controller" else v
    return s


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


def _fault_spec(ftype, joint, m, t_on):
    leg, j = joint.split("-")
    if ftype == "gain":
        return dict(type="actuator_gain", leg=leg, joint=j, magnitude=-float(m), t_onset=t_on)          # kappa = 1 - m
    if ftype == "bias":
        return dict(type="actuator_bias", leg=leg, joint=j, magnitude=float(m), t_onset=t_on)
    return dict(type="friction_scale", leg=leg, joint=j, magnitude=float(m) / 0.2, t_onset=t_on)     # dmu = 0.2 * scale


def _classify(Zr, K_cal, D, keys, A, fault):
    """Registered residual cycles (K, 12, N) -> beta^2, nearest class, DK certificate (profiles centred by the calibration mean)."""
    prof = Zr.reshape(Zr.shape[0], -1)
    prof = prof - prof[:K_cal].mean(0)                           # change relative to the calibrated nominal residual profile
    b2 = beta_op2(prof[:K_cal]); v = top_direction(prof[K_cal:])
    c1, s1, c2, s2 = nearest_class(v, D)
    rec = {"beta2": b2, "pred": c1, "cos1": s1, "second": c2, "cos2": s2,
           "floor_rms_per_joint": np.sqrt((Zr[:K_cal] ** 2).mean(axis=(0, 2))).tolist()}
    if fault is not None:
        rec["cert"] = dk_certificate(D, _true_class(fault), _true_mag(fault), b2, keys, A)
    return rec


def _rep(sim_cfg, fault, K_cal, K_post, N, drop_first, ocfg, need_arrays, weld=False):
    """One run (worker): rollout, dictionary from the run's nominal cycles, analytic residual classification, and — if
    need_arrays — the raw per-leg arrays for the DeLaN residuals (computed in the parent: autograd cannot run in
    forked workers once the parent has used it)."""
    cfg = SimConfig(**sim_cfg); period = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (K_cal + K_post + drop_first + 2) * period
    if fault is not None:
        cfg.faults = [fault]
    df, man = rollout(cfg)
    dyn = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    Zt, _ = register_cycles(df, TAU_COLS, N=N, drop_first=drop_first); Zd, _ = register_cycles(df, DQ_COLS, N=N, drop_first=drop_first)
    K = min(K_cal + K_post, Zt.shape[0])
    D = build_dictionary(Zt[:K_cal], Zd[:K_cal], JOINT_NAMES); keys, A = principal_angle_matrix(D)
    r_an = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=ocfg["cutoff_hz"], torque=ocfg["torque"])[:, 6:]
    Zr, _ = registered_residuals(df, r_an, N=N, drop_first=drop_first)
    out = {"K": K, "models": {"analytic": _classify(Zr[:K], K_cal, D, keys, A, fault)}, "D": D, "keys": keys, "A": A, "fault": fault}
    if need_arrays:
        jt = contact_torques_all(df, dyn) if not weld else np.zeros((len(df), 12))
        legs = {}
        for li, leg in enumerate(LEGS):
            q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt)
            legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32),
                         "y": (tau + jt[:, 3 * li:3 * li + 3]).astype(np.float32)}
        out["arrays"] = {"legs": legs, "theta": df["theta"].to_numpy(), "t": df["t"].to_numpy(), "N": N, "drop_first": drop_first}
    return out


def _delan_classify(out, tags, K_cal):
    """Parent-side: DeLaN residuals for each model tag -> classification records added to out['models']."""
    arr = out.pop("arrays", None)
    if arr is None:
        return out
    K = out["K"]; T = len(arr["theta"]); r = {tag: np.zeros((T, 12)) for tag in tags}
    for li, leg in enumerate(LEGS):
        L = arr["legs"][leg]
        for tag in tags:
            r[tag][:, 3 * li:3 * li + 3] = L["y"] - _MODELS[tag].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
    dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]})
    for tag in tags:
        Zr, _ = registered_residuals(dfr, r[tag], N=arr["N"], drop_first=arr["drop_first"])
        out["models"][tag] = _classify(Zr[:K], K_cal, out["D"], out["keys"], out["A"], out["fault"])
    return out


def _true_class(fault):
    typ = {"actuator_gain": "gain", "actuator_bias": "bias", "friction_scale": "friction"}[fault["type"]]
    return (typ, f"{fault['leg']}-{fault['joint']}")


def _true_mag(fault):
    return {"actuator_gain": -fault["magnitude"], "actuator_bias": fault["magnitude"], "friction_scale": 0.2 * fault["magnitude"]}[fault["type"]]


def _load_models(tags, weld_dir=None):
    for tag in tags:
        if tag in _MODELS:
            continue
        path = (weld_dir or DATA_ROOT / "models" / "delan") / tag
        if (path / "meta.json").exists():
            _MODELS[tag] = DeLaNQuadruped.load(path, device="cpu")
        else:
            print(f"  [e06] WARNING: DeLaN model '{tag}' not found at {path}; skipped", flush=True)
    return [t for t in tags if t in _MODELS]


def _analyse(results, models, res_dir, prefix, quick):
    """Confusion matrices, certificate agreement, accuracy-vs-beta curve."""
    rows = []
    for (ftype, joint, m, r), out in results:
        true = (ftype, joint)
        for name in models:
            rec = out["models"][name]; cert = rec["cert"]
            rows.append({"model": name, "true_type": ftype, "true_joint": joint, "magnitude": m, "rep": r, "pred_type": rec["pred"][0], "pred_joint": rec["pred"][1],
                         "correct": rec["pred"] == true, "joint_correct": rec["pred"][1] == joint, "beta2": rec["beta2"], "beta2_threshold": cert["beta2_threshold"],
                         "ratio": rec["beta2"] / cert["beta2_threshold"], "certified": cert["certified"], "theta_min_deg": cert["theta_min_deg"], "nearest_class": f"{cert['nearest'][0]}:{cert['nearest'][1]}",
                         "dk_perturbation_deg": cert["dk_perturbation_deg"], "cos1": rec["cos1"], "cos2": rec["cos2"]})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / f"{prefix}_runs.csv", index=False)
    # confusion (type-level within the true joint and full class-level) per model
    summ = []
    for name in models:
        t = tab[tab.model == name]
        cm = pd.crosstab(t.true_type, t.pred_type).reindex(index=list(TYPES), columns=list(TYPES), fill_value=0)
        cm.to_csv(res_dir / f"{prefix}_confusion_type_{name}.csv")
        agree = float(np.mean(t.certified == t.correct)); acc = float(t.correct.mean()); jacc = float(t.joint_correct.mean())
        acc_cert = float(t[t.certified].correct.mean()) if t.certified.any() else np.nan; acc_uncert = float(t[~t.certified].correct.mean()) if (~t.certified).any() else np.nan
        summ.append({"model": name, "accuracy_class": acc, "accuracy_joint": jacc, "n_certified": int(t.certified.sum()), "n": len(t), "accuracy_when_certified": acc_cert,
                     "accuracy_when_not_certified": acc_uncert, "certificate_outcome_agreement": agree, "beta2_median": float(t.beta2.median()),
                     "beta_rms_floor_median": float(np.median([np.sqrt(np.mean(np.square(out["models"][name]["floor_rms_per_joint"]))) for _, out in results]))})
    S = pd.DataFrame(summ); S.to_csv(res_dir / f"{prefix}_summary.csv", index=False)
    # accuracy vs beta^2 / beta^2_threshold (all models pooled; log-x bins)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    for name, mk in zip(models, "osd^v<>ph*"):
        t = tab[tab.model == name]
        ax.scatter(t.ratio, t.correct.astype(float) + np.random.default_rng(1).uniform(-0.03, 0.03, len(t)), s=14, marker=mk, alpha=0.6, label=name)
    edges = np.logspace(-3, 3, 13); mids = np.sqrt(edges[:-1] * edges[1:]); acc_b = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = tab[(tab.ratio >= lo) & (tab.ratio < hi)]; acc_b.append(sel.correct.mean() if len(sel) else np.nan)
    ax.plot(mids, acc_b, "k-", lw=2, label="binned accuracy (all models)")
    ax.axvline(1.0, color="r", ls="--", label="DK threshold (ratio = 1)"); ax.set_xscale("log"); ax.set_xlabel("beta_op^2 / beta^2_threshold(class, magnitude)"); ax.set_ylabel("isolation correct")
    ax.legend(fontsize=6); ax.grid(alpha=0.3); ax.set_title("nearest-subspace isolation vs DK certificate", fontsize=9)
    ax = axes[1]
    ax.bar(np.arange(len(S)), S.accuracy_class, width=0.4, label="class accuracy"); ax.bar(np.arange(len(S)) + 0.4, S.accuracy_joint, width=0.4, label="joint accuracy")
    ax.set_xticks(np.arange(len(S)) + 0.2); ax.set_xticklabels(S.model, rotation=45, fontsize=7); ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.set_title("isolation accuracy per residual model", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / f"{prefix}_accuracy.png", dpi=150); plt.close(fig)
    return tab, S


def stage_iso(cfg, res_dir, quick=False):
    si = cfg["stage_iso"]; R = 1 if quick else si["R"]; K_cal, K_post = si["K_cal"], si["K_post"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]
    tags = _load_models(cfg["delan_models"]); models = ["analytic"] + tags
    t_on = (K_cal + df0) * 0.5
    cells = [(ft, j, m) for ft, mags in si["faults"].items() for j in si["joints"] for m in mags]
    if quick:
        cells = [(ft, "LF-KFE", mags[-1]) for ft, mags in si["faults"].items()]
    args = []; index = []
    for ci, (ft, j, m) in enumerate(cells):
        for r in range(R):
            args.append((_sim(cfg, si["seed_base"] + 100 * ci + r), _fault_spec(ft, j, m, t_on), K_cal, K_post, N, df0, oc, bool(tags))); index.append((ft, j, m, r))
    print(f"  [iso] {len(args)} runs, models {models}", flush=True)
    res = pmap(_rep, args, cfg["workers"])
    res = [_delan_classify(o, tags, K_cal) for o in res]
    results = list(zip(index, res))
    # principal-angle matrix from the first run's dictionary is run-specific; report the analytic theta_min per class
    tab, S = _analyse(results, models, res_dir, "e06iso", quick)
    # angle matrix figure from one nominal-cycles dictionary
    Z0 = res[0]
    ang = tab.groupby(["true_type", "true_joint"]).theta_min_deg.median().reset_index(); ang.to_csv(res_dir / "e06iso_theta_min.csv", index=False)
    an = S[S.model == "analytic"].iloc[0]; fu = S[S.model == "full"].iloc[0] if "full" in models else None
    lad = S[S.model.isin([m for m in models if m != "analytic"])].sort_values("beta2_median")
    # (iii) turn: accuracy should be high where beta2/threshold < 1 and drop above
    below = tab[tab.ratio < 1].correct.mean() if (tab.ratio < 1).any() else np.nan; above = tab[tab.ratio >= 1].correct.mean() if (tab.ratio >= 1).any() else np.nan
    ok_i = an.certificate_outcome_agreement >= 0.8; ok_ii = (fu is not None) and fu.certificate_outcome_agreement >= 0.8
    ok_iii = (not np.isnan(below)) and (not np.isnan(above)) and below > above + 0.2
    _conclude(res_dir, f"[e06 iso] (i) analytic: class acc {an.accuracy_class:.2f}, joint acc {an.accuracy_joint:.2f}, DK agreement {an.certificate_outcome_agreement:.2f} ({'PASS' if ok_i else 'FAIL'}); "
              + (f"(ii) DeLaN full: class acc {fu.accuracy_class:.2f}, joint acc {fu.accuracy_joint:.2f}, DK agreement {fu.certificate_outcome_agreement:.2f} ({'PASS' if ok_ii else 'FAIL'}); " if fu is not None else "(ii) DeLaN full: MISSING; ")
              + f"(iii) accuracy below DK threshold {below:.2f} vs above {above:.2f} ({'PASS' if ok_iii else 'FAIL'}); ladder (beta2 median, class acc): "
              + ", ".join(f"{r.model}: {r.beta2_median:.3g}/{r.accuracy_class:.2f}" for r in lad.itertuples()))
    return {"pass": bool(ok_i and ok_ii and ok_iii)}


def _weld_rollout_arrays(sim_cfg, seed):
    cfg = SimConfig(**sim_cfg); cfg.seed = int(seed); df, man = rollout(cfg)
    out = {}
    for li, leg in enumerate(LEGS):
        q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt); out[leg] = {"q": q, "dq": dq, "ddq": ddq, "a": a, "y": tau}
    out["n"] = len(df); return out


def stage_weld(cfg, res_dir, quick=False):
    import torch
    sw = cfg["stage_weld"]; R = 1 if quick else sw["R"]; K_cal, K_post = sw["K_cal"], sw["K_post"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]
    wdir = DATA_ROOT / "models" / "delan_weld"; tag = sw["delan_tag"]
    # 1) nominal DeLaN-weld training (legs in the air: contact term = 0)
    if not (wdir / tag / "meta.json").exists():
        tr = sw["train"]; nro = 4 if quick else tr["rollouts"]
        args = [(dict(_sim(cfg, tr["seed_base"] + i), weld_base=True, duration_s=float(tr["duration_s"]) if not quick else 8.0), tr["seed_base"] + i) for i in range(nro)]
        res = pmap(_weld_rollout_arrays, args, cfg["workers"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        quad = DeLaNQuadruped.build(hidden=128, depth=3, eps=1e-3, damping=oc["damping"], frictionloss=oc["frictionloss"], device=dev)
        nva = max(1, nro // 5); rep = {"legs": {}}
        for leg in LEGS:
            d = {"train": {k: np.concatenate([r[leg][k] for r in res[nva:]]) for k in ("q", "dq", "ddq", "a", "y")},
                 "val": {k: np.concatenate([r[leg][k] for r in res[:nva]]) for k in ("q", "dq", "ddq", "a", "y")}}
            hist, resid = train_leg(quad.nets[leg], d, epochs=(3 if quick else tr["epochs"]), batch=4096, lr=1e-3, device=dev, log=lambda *a: None)
            rep["legs"][leg] = {"final_val_rmse_per_joint": hist[-1]["val_rmse_per_joint"], "beta_hat": beta_hat(d["val"]["q"], resid)}
        quad.save(wdir / tag); (wdir / tag / "report.json").write_text(json.dumps(rep, indent=1))
        print(f"  [weld] DeLaN-weld trained: val rmse per leg " + str({l: np.round(rep['legs'][l]['final_val_rmse_per_joint'], 3).tolist() for l in LEGS}), flush=True)
    tags = _load_models([tag], weld_dir=wdir); models = ["analytic"] + tags
    t_on = (K_cal + df0) * 0.5
    si = cfg["stage_iso"]
    cells = [(ft, j, m) for ft, mags in si["faults"].items() for j in sw["joints"] for m in mags]
    if quick:
        cells = [(ft, "LF-KFE", mags[-1]) for ft, mags in si["faults"].items()]
    args = []; index = []
    for ci, (ft, j, m) in enumerate(cells):
        for r in range(R):
            args.append((dict(_sim(cfg, sw["seed_base"] + 100 * ci + r), weld_base=True), _fault_spec(ft, j, m, t_on), K_cal, K_post, N, df0, oc, bool(tags), True)); index.append((ft, j, m, r))
    print(f"  [weld] {len(args)} runs", flush=True)
    res = pmap(_rep, args, cfg["workers"])
    res = [_delan_classify(o, tags, K_cal) for o in res]
    tab, S = _analyse(list(zip(index, res)), models, res_dir, "e06weld", quick)
    # side-by-side table with the floating-base results if present
    side = []
    fl = res_dir / "e06iso_summary.csv"
    if fl.exists():
        F = pd.read_csv(fl)
        for _, r in F[F.model.isin(["analytic", "full"])].iterrows():
            side.append({"world": "floating-base trot", **r.to_dict()})
    for _, r in S.iterrows():
        side.append({"world": "welded base (leg = arm)", **r.to_dict()})
    pd.DataFrame(side).to_csv(res_dir / "e06_leg_arm_bridge.csv", index=False)
    an = S[S.model == "analytic"].iloc[0]
    _conclude(res_dir, f"[e06 weld] leg=arm: analytic class acc {an.accuracy_class:.2f} joint acc {an.accuracy_joint:.2f} DK agreement {an.certificate_outcome_agreement:.2f}, floor rms {an.beta_rms_floor_median:.3f}; "
              + "; ".join(f"{r.model}: acc {r.accuracy_class:.2f} agreement {r.certificate_outcome_agreement:.2f} floor {r.beta_rms_floor_median:.3f}" for r in S.itertuples() if r.model != "analytic"))
    return {"pass": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["iso", "weld", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["iso", "weld"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"iso": stage_iso, "weld": stage_weld}[s](cfg, res_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E06 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
