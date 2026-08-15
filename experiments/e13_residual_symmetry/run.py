#!/usr/bin/env python3
"""e13 — residual symmetry tests (Sprint 6, Block R). Theory Part 2 anchors.

  a  residual R^- (analytic momentum observer / equivariant DeLaN) vs raw-signal R^- on the low-SNR grid, same flip test
     and the same FAR-calibrated e-CUSUM protocol; nominal variance decomposition Var(Pi^- y) vs Var(Pi^- r) row by row;
     Mahalanobis (magnitude) reference; minimal detectable magnitude table.                            [Prop N3-2]
  b  size of the residual flip test under H0 against the model's equivariance defect delta_f^(0.95): plain DeLaN ladder
     (full/n50k/n10k/n2k) vs equivariant ladder vs analytic residual vs raw; K = 60 and K = 200 cycles; calibration-
     centred variants (Lemma centring).                                                                  [Cor contamination]
  c  e04c isotypic groups on the residual: R^- on Pi^- r, R^+ on Pi^+ r, projection-energy shares; three-channel
     isolation confusion (raw+track / analytic rows / equivariant rows), R = 20 per class.                [Cor two-channel]
  d  (wheeled M1) not runnable: the M1 world (Sprint 5 Block W) is not in the repo.

    python experiments/e13_residual_symmetry/run.py --stage a|b|c|all [--run-id ID] [--quick] [--workers N]

Calibration discipline as in e04/e05/e07: every detector is calibrated on the run's own nominal cycles only; the e-CUSUM
threshold per R^- variant is calibrated on the pooled pre-onset windows of the stage. Nothing is trained here: the
DeLaN models come from Block Q (nominal data only). DeLaN inference runs in the parent (GPU); workers never touch torch.
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

from geofdi.baselines.mahalanobis import MahalanobisGate, cycle_features
from geofdi.detect.monitors import (LEGS, MirrorMonitor, calibrate_ecusum_threshold, channel_projection_energy, conformal_pvalues,
                                    ecusum, eprocess_alarm, leg_magnitude_deviation, rank_groups, tracking_scores)
from geofdi.detect.permutation import hg_permutation_test, pooled_scale
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.dynamics.delan import contact_torques_all, leg_arrays
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import BASE_COLS, RES_COLS, isotypic_split, residual_manifest
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry import JOINTS, z_channel_names

EXP_NAME = "e13_residual_symmetry"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]
TAU_COLS = [f"tau_cmd_{l}_{j}" for l in LEGS for j in JOINTS]
_MODELS: dict = {}
_RES_REP = C2Rep(residual_manifest(include_base=False))
_RES_MAN = residual_manifest(include_base=False)


# ------------------------------------------------------------------------------------ helpers
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


def _load_models(tags):
    """Parent-side DeLaN models (GPU if available)."""
    import torch
    from geofdi.dynamics.delan_equiv import load_delan
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for t in tags:
        if t in _MODELS:
            continue
        p = DATA_ROOT / "models" / "delan" / t
        if (p / "meta.json").exists():
            _MODELS[t] = load_delan(p, device=dev)
        else:
            print(f"  [e13] WARNING: DeLaN model '{t}' missing at {p}", flush=True)
    return [t for t in tags if t in _MODELS]


def _model_defects(tags):
    out = {}
    for t in tags:
        p = DATA_ROOT / "models" / "delan" / t / "report.json"
        if p.exists():
            r = json.loads(p.read_text()); d = r.get("defect", {})
            out[t] = {"delta_q95": d.get("q95", np.nan), "delta_q50": d.get("q50", np.nan), "delta_max": d.get("max", np.nan),
                      "n_train_per_leg": r.get("n_train_per_leg"), "equivariant": bool(r.get("equivariant", False)),
                      "beta_hat_q95": r.get("beta_hat_global_q95_mean_over_legs", np.nan)}
    return out


# ------------------------------------------------------------------------------------ worker
def _worker(sim_cfg, K_total, N, drop_first, ocfg, need_arrays=True):
    """One rollout -> raw cycles Z, q_ref cycles, analytic residual cycles (joint rows + base rows), per-leg arrays for
    the DeLaN residuals (computed in the parent), manifest and channel names."""
    cfg = SimConfig(**sim_cfg); period = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (K_total + drop_first + 2) * period
    df, man = rollout(cfg)
    chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first)
    Zq, _ = register_cycles(df, QREF, N=N, drop_first=drop_first)
    dyn = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=ocfg["cutoff_hz"], torque=ocfg["torque"])
    Zr, _ = registered_residuals(df, r[:, 6:], N=N, drop_first=drop_first)
    dfb = df[["t", "theta"]].copy()
    for i, c in enumerate(BASE_COLS):
        dfb[c] = r[:, i]
    Zb, _ = register_cycles(dfb, BASE_COLS, N=N, drop_first=drop_first)
    K = min(K_total, Z.shape[0], Zq.shape[0], Zr.shape[0], Zb.shape[0])
    out = {"K": K, "Z": Z[:K].astype(np.float32), "Zq": Zq[:K].astype(np.float32), "Zr_an": Zr[:K].astype(np.float32),
           "Zb": Zb[:K].astype(np.float32), "chans": chans, "man": man}
    if need_arrays:
        jt = contact_torques_all(df, dyn)
        legs = {}
        for li, leg in enumerate(LEGS):
            q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt)
            legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32),
                         "y": (tau + jt[:, 3 * li:3 * li + 3]).astype(np.float32)}
        out["arrays"] = {"legs": legs, "theta": df["theta"].to_numpy(), "t": df["t"].to_numpy(), "N": N, "drop_first": drop_first}
    return out


def _delan_cycles(out, tags):
    """Parent-side: DeLaN residual cycles (K, 12, N) per model tag; frees the arrays."""
    arr = out.pop("arrays", None)
    if arr is None:
        return {}
    T = len(arr["theta"]); res = {}
    dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]})
    for tag in tags:
        r = np.zeros((T, 12), dtype=np.float32)
        for li, leg in enumerate(LEGS):
            L = arr["legs"][leg]
            r[:, 3 * li:3 * li + 3] = L["y"] - _MODELS[tag].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
        Zr, _ = registered_residuals(dfr, r, N=arr["N"], drop_first=arr["drop_first"])
        res[tag] = Zr[:out["K"]].astype(np.float32)
    return res


# ------------------------------------------------------------------------------------ detectors
def _rminus_pvals(Zx, rep, det, seed):
    mm = MirrorMonitor(rep, window=det["window_rminus"], M=det["M"], statistic="paired_energy", alpha=det["alpha"])
    return mm.window_pvalues(Zx, seed=seed)


def _score_alarm(scores, K_cal, alpha):
    p = conformal_pvalues(scores[:K_cal], scores[K_cal:])
    E, al = eprocess_alarm(p, alpha, start=0)
    return None if al is None else al + 1, p


def _sym_scores(Zr):
    Zp, Zm = isotypic_split(Zr, _RES_REP)
    return residual_scores(Zp), residual_scores(Zm)


def _variance_decomposition(Z, chans, Zr_dict, K_cal):
    """Nominal per-row variance of the sign component and the fault SNR (post vs cal mean shift of Pi^- rows) for the
    tau_cmd rows of the raw element and for each residual variant. Returns dict of {name: (var (12,), snr (12,))}."""
    idx = [chans.index(c) for c in TAU_COLS]
    Ztau = Z[:, idx, :]
    out = {}
    for name, X in [("tau_cmd", Ztau), *Zr_dict.items()]:
        Xp, Xm = isotypic_split(X, _RES_REP)                            # residual-rep signs == tau signs (12 rows)
        var_row = Xm[:K_cal].var(axis=0).mean(axis=1)                    # (12,) mean over phase of the cycle variance
        shift = Xm[K_cal:].mean(axis=0) - Xm[:K_cal].mean(axis=0)        # (12, N)
        snr_row = (shift ** 2).sum(axis=1) / (var_row + 1e-12)
        var_plus = Xp[:K_cal].var(axis=0).mean(axis=1)
        out[name] = {"var_minus": var_row, "snr_minus": snr_row, "var_plus": var_plus}
    # whole raw element (all channels), standardized: total antisymmetric SNR of everything the raw test sees
    repZ = C2Rep(_worker_man)
    Zs = repZ.apply("s", Z); Dm = 0.5 * (Z - Zs)
    var_all = Dm[:K_cal].var(axis=0).mean(axis=1); shift_all = Dm[K_cal:].mean(0) - Dm[:K_cal].mean(0)
    out["raw_all"] = {"snr_total": float(((shift_all ** 2).sum(axis=1) / (var_all + 1e-12)).sum())}
    return out


_worker_man = None


# ------------------------------------------------------------------------------------ stage a
def _maha_gate(cfg, res_dir, quick):
    """Mahalanobis gate fitted on a separate nominal training set (e07 protocol)."""
    mt = cfg["stage_a"]["mahalanobis_train"]; nro = 3 if quick else mt["rollouts"]; ncy = 30 if quick else mt["cycles"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    args = [(_sim(cfg, mt["seed_base"] + i), ncy, N, df0, cfg["observer"], False) for i in range(nro)]
    res = pmap(_worker, args, cfg["workers"])
    F = np.concatenate([cycle_features(r["Z"]) for r in res])
    gate = MahalanobisGate().fit(F)
    print(f"  [a] Mahalanobis gate fitted on {F.shape[0]} nominal cycles ({F.shape[1]} features)", flush=True)
    return gate


def stage_a(cfg, res_dir, quick=False):
    global _worker_man
    sa = cfg["stage_a"]; R = 4 if quick else sa["R"]; K_cal, K_post = sa["K_cal"], (30 if quick else sa["K_post"])
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha, window = det["alpha"], det["window_rminus"]
    oc = cfg["observer"]; t_on = _onset_time(K_cal, df0)
    tags = _load_models([cfg["delan"]["equivariant"], cfg["delan"]["plain"]]); eq, pl = cfg["delan"]["equivariant"], cfg["delan"]["plain"]
    gate = _maha_gate(cfg, res_dir, quick)
    cells = []
    for ftype, mags in sa["faults"].items():
        mags = mags[-1:] if quick else mags
        for mag in mags:
            for joint in sa["joints"]:
                magnitude = {"actuator_gain": mag - 1.0, "friction_scale": mag - 1.0}.get(ftype, mag)
                cells.append((ftype, mag, joint, magnitude))
    rminus_names = {"raw": None, "res_an": "Zr_an", "res_eq": eq, "res_pl": pl}
    all_p = {k: [] for k in rminus_names}; nom_p = {k: [] for k in rminus_names}
    rows_run = []; var_rows = []
    for ci, (ftype, mag, joint, magnitude) in enumerate(cells):
        fault = dict(type=ftype, t_onset=t_on, leg=sa["leg"], joint=joint, magnitude=float(magnitude))
        args = [(_sim(cfg, sa["seed_base"] + 100 * ci + r, faults=[fault]), K_cal + K_post, N, df0, oc, True) for r in range(R)]
        res = pmap(_worker, args, cfg["workers"])
        for r_i, out in enumerate(res):
            _worker_man = out["man"]; K = out["K"]; seed = sa["seed_base"] + 90000 + 100 * ci + r_i
            Zr = _delan_cycles(out, tags)
            Zx = {"raw": out["Z"], "res_an": out["Zr_an"], "res_eq": Zr.get(eq), "res_pl": Zr.get(pl)}
            rec = {"cell": ci, "fault": ftype, "magnitude": mag, "joint": joint, "rep": r_i, "K": K}
            for k, Zk in Zx.items():
                if Zk is None:
                    continue
                rep = C2Rep(out["man"]) if k == "raw" else _RES_REP
                ps = _rminus_pvals(Zk, rep, det, seed); all_p[k].append(ps); nom_p[k].append(ps[:K_cal // window])
                rec[f"_p_{k}"] = ps
            # magnitude channels (conformal e-process; unified protocol)
            sc = {"track": tracking_scores(out["Z"], out["chans"], out["Zq"]), "maha": gate.score(cycle_features(out["Z"]))}
            for k, Zk in (("res_an", out["Zr_an"]), ("res_eq", Zr.get(eq)), ("res_pl", Zr.get(pl))):
                if Zk is None:
                    continue
                sp, sm = _sym_scores(Zk); sc[f"{k}_full"] = residual_scores(Zk); sc[f"{k}_sym"] = sp
            for k, s in sc.items():
                d, _ = _score_alarm(s, K_cal, alpha); rec[f"Rplus_{k}"] = d
            vd = _variance_decomposition(out["Z"], out["chans"], {"res_an": out["Zr_an"], **({"res_eq": Zr[eq]} if eq in Zr else {}), **({"res_pl": Zr[pl]} if pl in Zr else {})}, K_cal)
            for name, v in vd.items():
                if name == "raw_all":
                    var_rows.append({"cell": ci, "fault": ftype, "magnitude": mag, "joint": joint, "rep": r_i, "signal": name, "row": "all", "var_minus": np.nan, "var_plus": np.nan, "snr_minus": v["snr_total"]})
                    continue
                for j, rn in enumerate([f"{l}-{jj}" for l in LEGS for jj in JOINTS]):
                    var_rows.append({"cell": ci, "fault": ftype, "magnitude": mag, "joint": joint, "rep": r_i, "signal": name, "row": rn,
                                     "var_minus": float(v["var_minus"][j]), "var_plus": float(v["var_plus"][j]), "snr_minus": float(v["snr_minus"][j])})
            rows_run.append(rec)
        print(f"  [a] cell {ci + 1}/{len(cells)} {ftype} {mag} {joint} done", flush=True)
    # e-CUSUM thresholds per R^- variant from the pooled nominal windows
    h = {k: calibrate_ecusum_threshold(nom_p[k], det["ecusum_horizon_windows"], far=alpha, n_boot=det["ecusum_boot"], rng=np.random.default_rng(3)) for k in nom_p if nom_p[k]}
    pd.DataFrame([{"variant": k, "ecusum_h": v} for k, v in h.items()]).to_csv(res_dir / "e13a_ecusum_thresholds.csv", index=False)
    w0 = K_cal // window
    for rec in rows_run:
        for k in rminus_names:
            ps = rec.pop(f"_p_{k}", None)
            if ps is None:
                continue
            E, al = eprocess_alarm(ps, alpha, start=w0); rec[f"Rminus_{k}_eproc"] = None if al is None else (al - w0 + 1) * window
            S, al = ecusum(ps, h[k], start=w0); rec[f"Rminus_{k}_ecusum"] = None if al is None else (al - w0 + 1) * window
            rec[f"Rminus_{k}_winrej"] = float(np.mean(ps[w0:] <= alpha))
    runs = pd.DataFrame(rows_run); runs.to_csv(res_dir / "e13a_runs.csv", index=False)
    dets = [c for c in runs.columns if c.startswith(("Rminus_", "Rplus_")) and not c.endswith("_winrej")]
    rows = []
    for (ftype, mag, joint), g in runs.groupby(["fault", "magnitude", "joint"]):
        for dname in dets:
            dl = g[dname].astype(float).to_numpy()
            rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": dname, "R": len(g), "det_rate_100": float(np.mean(dl <= K_post)),
                         "det_rate_20": float(np.mean(dl <= 20)), "delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                         "delay_q90": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan,
                         "ci_lo": binom_ci(int(np.sum(dl <= K_post)), len(g))[0], "ci_hi": binom_ci(int(np.sum(dl <= K_post)), len(g))[1]})
        for k in rminus_names:
            col = f"Rminus_{k}_winrej"
            if col in g:
                rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": col, "R": len(g), "det_rate_100": float(g[col].mean()), "det_rate_20": np.nan,
                             "delay_median": np.nan, "delay_q90": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e13a_power.csv", index=False)
    var = pd.DataFrame(var_rows); var.to_csv(res_dir / "e13a_variance_runs.csv", index=False)
    vsum = var[var.row != "all"].groupby(["signal", "row"])[["var_minus", "var_plus"]].mean().reset_index(); vsum.to_csv(res_dir / "e13a_variance_decomposition.csv", index=False)
    snr = var.groupby(["fault", "magnitude", "joint", "signal"]).snr_minus.sum().reset_index()
    snr["snr_minus"] = snr["snr_minus"] / R                                # sum over rows and reps / R = mean over reps of the row-sum
    snr.to_csv(res_dir / "e13a_snr.csv", index=False)
    # minimal detectable magnitude table (smallest magnitude with det100 >= criterion; gain: largest 1-kappa ... i.e. smallest 1-kappa)
    crit = sa["detect_criterion"]; mdm = []
    main = ["Rminus_raw_ecusum", "Rminus_res_an_ecusum", "Rminus_res_eq_ecusum", "Rminus_raw_eproc", "Rminus_res_an_eproc", "Rminus_res_eq_eproc",
            "Rplus_maha", "Rplus_res_eq_sym", "Rplus_res_eq_full", "Rplus_res_an_sym", "Rplus_res_an_full", "Rplus_track"]
    for ftype in sa["faults"]:
        for joint in sa["joints"]:
            for dname in main:
                sub = tab[(tab.fault == ftype) & (tab.joint == joint) & (tab.detector == dname)].copy()
                if sub.empty:
                    continue
                sub["sev"] = ((1 - sub.magnitude) if ftype == "actuator_gain" else (sub.magnitude - 1 if ftype == "friction_scale" else sub.magnitude)).round(6)
                sub = sub.sort_values("sev"); ok = sub[sub.det_rate_100 >= crit]
                mdm.append({"fault": ftype, "joint": joint, "detector": dname, "min_detectable_severity": float(ok.sev.iloc[0]) if len(ok) else np.inf,
                            "severity_unit": {"actuator_gain": "1-kappa", "actuator_bias": "N m", "friction_scale": "scale-1"}[ftype],
                            "det100_by_severity": json.dumps({float(s): round(float(d), 2) for s, d in zip(sub.sev, sub.det_rate_100)})})
    pd.DataFrame(mdm).to_csv(res_dir / "e13a_min_detectable.csv", index=False)
    _plot_a(cfg, res_dir, tab, vsum, snr, K_post)
    # reading
    lines = []
    for ftype in sa["faults"]:
        for joint in sa["joints"]:
            g = lambda d: tab[(tab.fault == ftype) & (tab.joint == joint) & (tab.detector == d)].sort_values("magnitude").det_rate_100.round(2).tolist()
            lines.append(f"{ftype} {joint}: raw R- e-CUSUM {g('Rminus_raw_ecusum')} | res_an {g('Rminus_res_an_ecusum')} | res_eq {g('Rminus_res_eq_ecusum')} | Maha {g('Rplus_maha')} | R+ res_eq(Pi+) {g('Rplus_res_eq_sym')}")
    ratio = vsum.pivot(index="row", columns="signal", values="var_minus")
    _conclude(res_dir, "[e13a] det100 by magnitude (ascending magnitude value as listed in config): " + " || ".join(lines)
              + f" || nominal Var(Pi- r)/Var(Pi- tau_cmd) per row (median over rows): res_an {float((ratio['res_an'] / ratio['tau_cmd']).median()):.3g}, res_eq {float((ratio['res_eq'] / ratio['tau_cmd']).median()) if 'res_eq' in ratio else float('nan'):.3g}"
              + f" | e-CUSUM h: { {k: round(v, 3) for k, v in h.items()} }")
    return {"tab": tab}


def _plot_a(cfg, res_dir, tab, vsum, snr, K_post):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sa = cfg["stage_a"]; ftypes = list(sa["faults"].keys())
    lines = [("Rminus_raw_ecusum", "R⁻ raw signal (S1), e-CUSUM", "C3", "o"), ("Rminus_res_an_ecusum", "R⁻ analytic residual, e-CUSUM", "C1", "s"),
             ("Rminus_res_eq_ecusum", "R⁻ equivariant-DeLaN residual, e-CUSUM", "C0", "^"), ("Rplus_maha", "Mahalanobis (magnitude ref.), e-process", "0.4", "x")]
    fig, axes = plt.subplots(2, len(ftypes), figsize=(3.4 * len(ftypes), 5.6), squeeze=False)
    xlab = {"actuator_gain": "gain κ (LF)", "actuator_bias": "bias [N m] (LF)", "friction_scale": "friction scale (LF)"}
    for j, ftype in enumerate(ftypes):
        for row, joint in enumerate(sa["joints"]):
            ax = axes[row, j]
            for dname, lab, c, mk in lines:
                sub = tab[(tab.fault == ftype) & (tab.detector == dname) & (tab.joint == joint)].sort_values("magnitude")
                if len(sub):
                    ax.errorbar(sub.magnitude, sub.det_rate_100, yerr=[sub.det_rate_100 - sub.ci_lo, sub.ci_hi - sub.det_rate_100], marker=mk, color=c, capsize=2, lw=1.2, label=lab if (row == 0 and j == 0) else None)
            ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3); ax.set_title(f"{ftype} — LF-{joint}", fontsize=8); ax.set_xlabel(xlab[ftype], fontsize=8)
            if ftype == "actuator_gain":
                ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel(f"power: alarm within {K_post} cycles", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.suptitle("e13a — residual R⁻ vs raw R⁻ on the low-SNR grid (same flip test, same FAR-calibrated e-CUSUM; R = %d)" % int(tab.R.max()), fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e13a_power_curves.png", dpi=140); plt.close(fig)
    # delay figure
    fig, axes = plt.subplots(1, len(ftypes), figsize=(3.4 * len(ftypes), 3.2), squeeze=False)
    for j, ftype in enumerate(ftypes):
        ax = axes[0, j]
        for dname, lab, c, mk in lines[:3]:
            for joint, ls in zip(sa["joints"], ("-", "--")):
                sub = tab[(tab.fault == ftype) & (tab.detector == dname) & (tab.joint == joint)].sort_values("magnitude")
                ax.plot(sub.magnitude, sub.delay_median, ls, marker=mk, color=c, label=f"{lab} {joint}" if j == 0 else None)
        ax.set_ylim(0, K_post + 5); ax.grid(alpha=0.3); ax.set_title(f"{ftype}: median delay [cycles]", fontsize=8); ax.set_xlabel(xlab[ftype], fontsize=8)
        if ftype == "actuator_gain":
            ax.invert_xaxis()
    axes[0, 0].legend(fontsize=5); fig.tight_layout(); fig.savefig(res_dir / "e13a_delay.png", dpi=140); plt.close(fig)
    # variance decomposition
    piv = vsum.pivot(index="row", columns="signal", values="var_minus")
    rows = [f"{l}-{jj}" for l in LEGS for jj in JOINTS]; piv = piv.reindex(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(rows)); w = 0.27
    for i, (sig, lab, c) in enumerate((("tau_cmd", "raw τ_cmd rows: Var(Π⁻ y)", "C3"), ("res_an", "analytic residual: Var(Π⁻ r)", "C1"), ("res_eq", "equivariant DeLaN residual: Var(Π⁻ r)", "C0"))):
        if sig in piv:
            axes[0].bar(x + (i - 1) * w, piv[sig], width=w, color=c, label=lab)
    axes[0].set_yscale("log"); axes[0].set_xticks(x); axes[0].set_xticklabels(rows, rotation=60, fontsize=7); axes[0].set_ylabel("nominal variance of the sign component [N² m²]")
    axes[0].legend(fontsize=6); axes[0].set_title("N3-2 (b): nominal antisymmetric variance per torque row", fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    # SNR ratio residual/raw per cell (tau rows) for the analytic and equivariant residual
    s = snr.pivot_table(index=["fault", "magnitude", "joint"], columns="signal", values="snr_minus")
    if "tau_cmd" in s:
        for sig, c, mk in (("res_an", "C1", "s"), ("res_eq", "C0", "^")):
            if sig in s:
                axes[1].scatter(s["tau_cmd"], s[sig], color=c, marker=mk, s=28, label=f"SNR⁻({sig}) vs SNR⁻(τ_cmd rows)")
        cols = [c for c in ("tau_cmd", "res_an", "res_eq") if c in s]
        lim = [max(1.0, s[cols].min().min() * 0.7), s[cols].max().max() * 1.5]
        axes[1].plot(lim, lim, "k:", lw=1, label="equal"); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlim(lim); axes[1].set_ylim(lim)
        # null baseline of the estimator: E sum_rows ||Delta^-||^2 / Var ~ 12 rows x N (1/K_cal + 1/K_post) with no fault
        K_cal = int(cfg["stage_a"]["K_cal"]); N = int(cfg["registration"]["N"]); base = 12 * N * (1.0 / K_cal + 1.0 / K_post)
        axes[1].axvline(base, color="0.6", ls="--", lw=1); axes[1].axhline(base, color="0.6", ls="--", lw=1, label=f"null baseline of the SNR estimate (≈{base:.0f})")
        axes[1].set_xlabel("antisymmetric SNR of the raw τ_cmd rows (Σ_c ‖Δ⁻_c‖²/Var_c, 12 rows)"); axes[1].set_ylabel("antisymmetric SNR of the residual rows"); axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3, which="both")
        axes[1].set_title("N3-2: fault SNR, residual vs raw torque rows (one point per grid cell)", fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / "e13a_variance_snr.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------ stage b
def stage_b(cfg, res_dir, quick=False):
    sb = cfg["stage_b"]; R = 12 if quick else sb["R"]; K_cal, K_test = sb["K_cal"], (60 if quick else sb["K_test"])
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha, M = det["alpha"], det["M"]; oc = cfg["observer"]
    tags = _load_models(cfg["delan"]["ladder_plain"] + cfg["delan"]["ladder_equiv"])
    defects = _model_defects(tags)
    Ks = sorted({int(k) for k in sb.get("Ks", [60, K_test]) if int(k) <= K_test}) or [K_test]
    args = [(_sim(cfg, sb["seed_base"] + r), K_cal + K_test, N, df0, oc, True) for r in range(R)]
    rows = []
    # process in chunks to bound memory
    chunk = cfg["workers"] * 2
    for c0 in range(0, R, chunk):
        res = pmap(_worker, args[c0:c0 + chunk], cfg["workers"])
        for r_i, out in enumerate(res):
            ri = c0 + r_i; K = out["K"]; Zr = _delan_cycles(out, tags)
            variants = {"raw": (out["Z"], C2Rep(out["man"])), "res_an": (out["Zr_an"], _RES_REP)}
            for t in tags:
                variants[t] = (Zr[t], _RES_REP)
            for vi, (vname, (Zx, rep)) in enumerate(variants.items()):
                cal = Zx[:K_cal]; test = Zx[K_cal:K_cal + K_test]
                mean_cal = cal.mean(0)
                for Kk in Ks:
                    if Kk > test.shape[0]:
                        continue
                    # mode: plain (H0 on the residual element) | centred (naive: subtract the calibration mean profile;
                    # Lemma centring (iii) — the estimated common offset is NOT harmless for an energy statistic) |
                    # h0prime (asymmetry-change test: X_k = Z^mon_k - Z^cal_k, exactly sign-symmetric under H0' whatever
                    # the model's defect; costs K calibration cycles and doubles the variance)
                    modes = [("plain", test[:Kk]), ("centred", test[:Kk] - mean_cal)]
                    if Kk <= cal.shape[0]:
                        modes.append(("h0prime", test[:Kk] - cal[:Kk]))
                    for mi, (mode, X) in enumerate(modes):
                        p, obs = hg_permutation_test(X, rep, statistic="paired_energy", M=M, rng=np.random.default_rng([sb["seed_base"], ri, Kk, mi, vi]))
                        rows.append({"rep": ri, "variant": vname, "K": Kk, "mode": mode, "p": p, "obs": obs})
        print(f"  [b] {min(c0 + chunk, R)}/{R} runs done", flush=True)
    runs = pd.DataFrame(rows); runs.to_csv(res_dir / "e13b_runs.csv", index=False)
    summ = []
    for (v, Kk, ce), g in runs.groupby(["variant", "K", "mode"]):
        k = int((g.p <= alpha).sum()); n = len(g); band = nominal_band(alpha, n)
        d = defects.get(v, {})
        summ.append({"variant": v, "K": Kk, "mode": ce, "size": k / n, "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1], "band_lo": band[0], "band_hi": band[1],
                     "in_band": bool(band[0] <= k / n <= band[1]), "n": n, "delta_f_q95": d.get("delta_q95", 0.0 if v in ("raw", "res_an") else np.nan),
                     "delta_f_q50": d.get("delta_q50", 0.0 if v in ("raw", "res_an") else np.nan), "equivariant": d.get("equivariant", v in ("raw", "res_an")),
                     "n_train_per_leg": d.get("n_train_per_leg"), "kind": ("plain DeLaN" if (v in defects and not defects[v]["equivariant"]) else ("equivariant DeLaN" if v in defects else v))})
    S = pd.DataFrame(summ); S.to_csv(res_dir / "e13b_size_vs_defect.csv", index=False)
    _plot_b(res_dir, S, alpha, Ks)
    plain = S[(S.kind == "plain DeLaN") & (S["mode"] == "plain")]; eqv = S[(S.kind == "equivariant DeLaN") & (S["mode"] == "plain")]
    h0p = S[S["mode"] == "h0prime"]; cen = S[S["mode"] == "centred"]
    _conclude(res_dir, "[e13b] size of the residual flip test under H0 (alpha 0.05): "
              + "; ".join(f"{r.variant} K={r.K} {r['mode']}: {r['size']:.3f}{'' if r.in_band else ' OUT'} (delta_q95 {r.delta_f_q95:.3g})" for _, r in S.sort_values(["kind", "delta_f_q95", "K", "mode"]).iterrows())
              + f" | band [{S.band_lo.iloc[0]:.3f}, {S.band_hi.iloc[0]:.3f}] | equivariant all in band: {bool(eqv.in_band.all())}; plain out of band at K={max(Ks)}: {plain[(plain.K == max(Ks))].loc[~plain[(plain.K == max(Ks))].in_band, 'variant'].tolist()}"
              + f" | H0' differenced test in band for all variants: {bool(h0p.in_band.all())} (sizes {h0p['size'].round(3).tolist()}); naive centring in band: {bool(cen.in_band.all())} (sizes {cen['size'].round(3).tolist()})")
    return {"S": S}


def _plot_b(res_dir, S, alpha, Ks):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Kmain = [k for k in Ks if k in (60, 200)] or Ks[-2:]
    fig, axes = plt.subplots(1, len(Kmain), figsize=(5.2 * len(Kmain), 3.8), squeeze=False)
    jit = {"equiv_full": 0.020, "equiv_n50k": 0.024, "equiv_n10k": 0.029, "equiv_n2k": 0.035}
    for ax, Kk in zip(axes[0], Kmain):
        s = S[S.K == Kk]
        band = (s.band_lo.iloc[0], s.band_hi.iloc[0]); ax.axhspan(band[0], band[1], color="0.9", label=f"binomial 95 % band (n={int(s.n.iloc[0])})"); ax.axhline(alpha, color="k", ls=":", lw=1)
        for kind, c, mk in (("plain DeLaN", "C3", "o"), ("equivariant DeLaN", "C0", "s")):
            for mode, ls, alp, lab in (("plain", "-", 1.0, ""), ("h0prime", "-.", 0.7, " (H0′ differenced test)"), ("centred", ":", 0.45, " (naive centring)")):
                g = s[(s.kind == kind) & (s["mode"] == mode)].sort_values("delta_f_q95")
                if g.empty:
                    continue
                x = g.delta_f_q95.to_numpy().astype(float); x = np.where(x > 0, x, [jit.get(v, 0.02) for v in g.variant])   # delta = 0 models: plotted at 0.02-0.035 (jittered)
                ax.errorbar(x, g["size"], yerr=[g["size"] - g.ci_lo, g.ci_hi - g["size"]], ls=ls, marker=mk, color=c, alpha=alp, capsize=2, label=f"{kind}{lab}")
        for v, c, mk in (("res_an", "C1", "D"), ("raw", "0.3", "x")):
            g = s[(s.variant == v) & (s["mode"] == "plain")]
            if len(g):
                ax.errorbar([0.012 if v == "res_an" else 0.008], g["size"], yerr=[g["size"] - g.ci_lo, g.ci_hi - g["size"]], marker=mk, color=c, capsize=2, ls="none", label={"res_an": "analytic residual (δ_f = 0)", "raw": "raw signal Z"}[v])
        ax.set_xscale("log"); ax.set_xlabel("δ_f^(0.95) of the nominal model [N m]  (δ_f = 0 models at x ≤ 0.035)", fontsize=8); ax.set_ylabel(f"empirical size at α={alpha}"); ax.set_title(f"K = {Kk} cycles per test", fontsize=9)
        ax.set_ylim(-0.02, min(1.02, max(0.3, float(s["size"].max()) * 1.15))); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6)
    fig.suptitle("e13b — size of the residual flip test under H0 vs the model's equivariance defect (nominal go2_urdf_sym world)", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e13b_size_vs_defect.png", dpi=140); plt.close(fig)
    # size vs K per model (uncentred): the onset of the contamination is ordered by delta_f
    if len(Ks) > 2:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        s = S[S["mode"] == "plain"]
        order = s.groupby("variant").delta_f_q95.first().sort_values()
        cmap = plt.get_cmap("Reds"); n_pl = int((s.groupby("variant").kind.first() == "plain DeLaN").sum())
        i_pl = 0
        for v in order.index:
            g = s[s.variant == v].sort_values("K"); kind = g.kind.iloc[0]
            if kind == "plain DeLaN":
                c = cmap(0.35 + 0.6 * i_pl / max(1, n_pl - 1)); i_pl += 1; ls, mk = "-", "o"
            elif kind == "equivariant DeLaN":
                c = "C0"; ls, mk = "-", "s"
            elif v == "res_an":
                c = "C1"; ls, mk = "-", "D"
            else:
                c = "0.3"; ls, mk = "--", "x"
            lab = f"{v} (δ_f^(0.95) = {g.delta_f_q95.iloc[0]:.2g} N m)" if kind == "plain DeLaN" else v
            ax.errorbar(g.K, g["size"], yerr=[g["size"] - g.ci_lo, g.ci_hi - g["size"]], ls=ls, marker=mk, color=c, capsize=2, label=lab, alpha=0.9 if kind == "plain DeLaN" else 0.75)
        # the two H0'-route variants (Lemma centring iii/iv), thin lines for all models
        for mode, ls, c, lab in (("centred", ":", "0.45", "naive centring (all models; grows with K/K_cal)"), ("h0prime", "-.", "C2", "H0′ differenced test (all models)")):
            first = True
            for v in order.index:
                g = S[(S["mode"] == mode) & (S.variant == v)].sort_values("K")
                if g.empty:
                    continue
                ax.plot(g.K, g["size"], ls=ls, color=c, lw=1, alpha=0.8, label=lab if first else None); first = False
        band = (s.band_lo.iloc[0], s.band_hi.iloc[0]); ax.axhspan(band[0], band[1], color="0.9"); ax.axhline(alpha, color="k", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("K cycles per test (window length)"); ax.set_ylabel(f"empirical size at α={alpha}"); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)
        ax.set_title("e13b — size vs test length: plain-DeLaN residuals saturate at K ≥ 10 for every δ_f of the ladder;\nequivariant / analytic / raw stay in band; naive centring grows with K, the differenced H0′ test does not", fontsize=8)
        fig.tight_layout(); fig.savefig(res_dir / "e13b_size_vs_K.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------ stage c
def _isolate(out, Zr_eq, K_cal, base_z, variant):
    """Three-channel isolation with fixed rules. variant: 'raw_track' (S2: R^- raw pair-joint + tracking-error left/right,
    no payload channel) | 'analytic_rows' (R^- raw pair-joint + analytic joint rows left/right + base rows payload) |
    'equiv_rows' (R^- on the equivariant residual element pair-joint + equivariant joint rows left/right + analytic base rows).
    Returns the predicted class label 'LEG-JOINT' or 'payload'."""
    Z, chans, man = out["Z"], out["chans"], out["man"]
    if variant in ("analytic_rows", "equiv_rows"):
        Zb = out["Zb"]; fz = Zb[:, 2, :].mean(axis=1)                            # per-cycle mean base f_z row
        z_fz = (fz[K_cal:].mean() - fz[:K_cal].mean()) / (fz[:K_cal].std() + 1e-9)
        if abs(z_fz) >= base_z:
            return "payload", {"z_fz": float(z_fz)}
    else:
        z_fz = np.nan
    if variant == "equiv_rows":
        Rz = Zr_eq; rep = _RES_REP; names = RES_COLS; groups = ("res",)
    else:
        Rz = Z; rep = C2Rep(man); names = chans; groups = ("q", "dq", "tau_cmd", "tau_meas")
    cal, post = Rz[:K_cal], Rz[K_cal:]
    e = channel_projection_energy(post, rep, names, swing_condition=True, groups=groups, Z_cal=cal)
    (pair, joint), _ = rank_groups(e["per_pair"])[0]
    legs = ("LF", "RF") if pair == "F" else ("LH", "RH")
    if variant == "raw_track":
        sc = tracking_scores(Z, chans, out["Zq"], per_leg=True); dev = leg_magnitude_deviation(sc[:K_cal], sc[K_cal:])
    elif variant == "analytic_rows":
        sc = residual_scores(out["Zr_an"], per_leg=True); dev = (sc[K_cal:].mean(0) - sc[:K_cal].mean(0)) / (sc[:K_cal].std(0) + 1e-9)
    else:
        sc = residual_scores(Zr_eq, per_leg=True); dev = (sc[K_cal:].mean(0) - sc[:K_cal].mean(0)) / (sc[:K_cal].std(0) + 1e-9)
    leg = max(legs, key=lambda l: dev[LEGS.index(l)])
    return f"{leg}-{joint}", {"z_fz": float(z_fz), "pair": pair, "joint": joint}


def stage_c(cfg, res_dir, quick=False):
    sc_ = cfg["stage_c"]; R = 6 if quick else sc_["R"]; K_cal, K_post = sc_["K_cal"], (40 if quick else sc_["K_post"])
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha, window = det["alpha"], det["window_rminus"]; oc = cfg["observer"]
    t_on = _onset_time(K_cal, df0); eq = cfg["delan"]["equivariant"]; tags = _load_models([eq])
    # ---- (1) e04c groups on the residual
    rows_run = []; nom_p = {"raw": [], "res_an": [], "res_eq": []}
    for gi, (gname, spec) in enumerate(sc_["groups"].items()):
        faults = [dict(type="actuator_gain", t_onset=t_on, leg=s["leg"], joint=s["joint"], magnitude=s["kappa"] - 1.0) for s in spec]
        args = [(_sim(cfg, sc_["seed_base"] + 100 * gi + r, faults=faults), K_cal + K_post, N, df0, oc, True) for r in range(R)]
        res = pmap(_worker, args, cfg["workers"])
        for r_i, out in enumerate(res):
            Zr = _delan_cycles(out, tags); K = out["K"]; seed = sc_["seed_base"] + 90000 + 100 * gi + r_i
            rec = {"group": gname, "rep": r_i, "K": K}
            for k, Zx, rep in (("raw", out["Z"], C2Rep(out["man"])), ("res_an", out["Zr_an"], _RES_REP), ("res_eq", Zr[eq], _RES_REP)):
                ps = _rminus_pvals(Zx, rep, det, seed); nom_p[k].append(ps[:K_cal // window]); rec[f"_p_{k}"] = ps
                # projection-energy shares of the mean post-onset deviation (standardized by the calibration pooled scale)
                Zs = rep.apply("s", Zx)
                scl = pooled_scale(Zx[:K_cal], Zs[:K_cal]); delta = ((Zx[K_cal:] - Zx[:K_cal].mean(0)) / scl).mean(0)
                dm = rep.apply("s", delta[None])[0]; anti = 0.5 * (delta - dm); sym = 0.5 * (delta + dm)
                rec[f"anti_share_{k}"] = float((anti ** 2).sum() / ((anti ** 2).sum() + (sym ** 2).sum() + 1e-12))
            sc = {"track": tracking_scores(out["Z"], out["chans"], out["Zq"])}
            for k, Zk in (("res_an", out["Zr_an"]), ("res_eq", Zr[eq])):
                sp, sm = _sym_scores(Zk); sc[f"{k}_full"] = residual_scores(Zk); sc[f"{k}_sym"] = sp; sc[f"{k}_anti"] = sm
            for k, s in sc.items():
                d, _ = _score_alarm(s, K_cal, alpha); rec[f"Rplus_{k}"] = d
            rows_run.append(rec)
        print(f"  [c] group {gname} done", flush=True)
    h = {k: calibrate_ecusum_threshold(nom_p[k], det["ecusum_horizon_windows"], far=alpha, n_boot=det["ecusum_boot"], rng=np.random.default_rng(4)) for k in nom_p}
    w0 = K_cal // window
    for rec in rows_run:
        for k in nom_p:
            ps = rec.pop(f"_p_{k}"); E, al = eprocess_alarm(ps, alpha, start=w0); rec[f"Rminus_{k}_eproc"] = None if al is None else (al - w0 + 1) * window
            S, al = ecusum(ps, h[k], start=w0); rec[f"Rminus_{k}_ecusum"] = None if al is None else (al - w0 + 1) * window
            rec[f"Rminus_{k}_winrej"] = float(np.mean(ps[w0:] <= alpha)); rec[f"_pw_{k}"] = ps[w0:]
    rows = []; energies = []
    for gname in sc_["groups"]:
        g = [r for r in rows_run if r["group"] == gname]
        for dname in [c for c in g[0] if c.startswith(("Rminus_", "Rplus_")) and not c.endswith("_winrej") and not c.startswith("_")]:
            dl = np.array([np.nan if r[dname] is None else r[dname] for r in g], dtype=float); k = int(np.sum(dl <= K_post))
            rows.append({"group": gname, "detector": dname, "power_100": k / len(g), "ci_lo": binom_ci(k, len(g))[0], "ci_hi": binom_ci(k, len(g))[1],
                         "delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan, "R": len(g)})
        for k in nom_p:
            pw = np.concatenate([r[f"_pw_{k}"] for r in g]); n = len(pw); band = nominal_band(alpha, n); kk = int(np.sum(pw <= alpha))
            rows.append({"group": gname, "detector": f"Rminus_{k}_window_rejection", "power_100": kk / n, "ci_lo": binom_ci(kk, n)[0], "ci_hi": binom_ci(kk, n)[1], "delay_median": np.nan, "R": len(g), "band_lo": band[0], "band_hi": band[1]})
        energies.append({"group": gname, **{f"anti_share_{k}": float(np.mean([r[f"anti_share_{k}"] for r in g])) for k in nom_p}})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e13c_isotypic_power.csv", index=False)
    en = pd.DataFrame(energies); en.to_csv(res_dir / "e13c_projection_energy.csv", index=False)
    pd.DataFrame([{"variant": k, "ecusum_h": v} for k, v in h.items()]).to_csv(res_dir / "e13c_ecusum_thresholds.csv", index=False)
    # ---- (2) isolation confusion
    iso = sc_["isolation"]; Ri = 3 if quick else iso["R"]; classes = iso["classes"] if not quick else iso["classes"][:2] + iso["classes"][-1:]
    conf_rows = []
    for ci, cl in enumerate(classes):
        if cl["type"] == "payload_asymmetric":
            over = dict(nuisance=[dict(type="payload_asymmetric", magnitude=cl["magnitude"], params=cl.get("params", {}), t_onset=t_on)])
        else:
            over = dict(faults=[dict(type=cl["type"], leg=cl["leg"], joint=cl["joint"], magnitude=cl["magnitude"], t_onset=t_on)])
        args = [(_sim(cfg, iso["seed_base"] + 100 * ci + r, **over), iso["K_cal"] + (40 if quick else iso["K_post"]), N, df0, oc, True) for r in range(Ri)]
        res = pmap(_worker, args, cfg["workers"])
        truth = "payload" if cl["type"] == "payload_asymmetric" else f"{cl['leg']}-{cl['joint']}"
        for r_i, out in enumerate(res):
            Zr = _delan_cycles(out, tags)
            for variant in ("raw_track", "analytic_rows", "equiv_rows"):
                pred, info = _isolate(out, Zr[eq], iso["K_cal"], iso["base_z_payload"], variant)
                conf_rows.append({"class": cl["name"], "truth": truth, "rep": r_i, "variant": variant, "pred": pred, "correct": pred == truth, **info})
        print(f"  [c] isolation class {cl['name']} done", flush=True)
    conf = pd.DataFrame(conf_rows); conf.to_csv(res_dir / "e13c_isolation_runs.csv", index=False)
    acc = conf.groupby("variant").correct.mean().reset_index(); acc.to_csv(res_dir / "e13c_isolation_accuracy.csv", index=False)
    for variant in conf.variant.unique():
        cm = pd.crosstab(conf[conf.variant == variant].truth, conf[conf.variant == variant].pred); cm.to_csv(res_dir / f"e13c_confusion_{variant}.csv")
    _plot_c(cfg, res_dir, tab, en, conf, alpha)
    def pw_(g, d): return float(tab[(tab.group == g) & (tab.detector == d)].power_100.iloc[0])
    def wr(g, k):
        r = tab[(tab.group == g) & (tab.detector == f"Rminus_{k}_window_rejection")].iloc[0]; return f"{r.power_100:.3f}[{r.band_lo:.3f},{r.band_hi:.3f}]"
    p_res = {}
    # P2 (blind): 5-cycle windows have 16 sign patterns, a single window can never reject at 0.05 (protocol_params.md), so
    # the per-window rejection rate is not the right metric; use the FAR-calibrated e-CUSUM alarm fraction (binomial band
    # of the calibrated FAR = alpha) and the plain e-process alarm fraction (<= alpha).
    band_R = nominal_band(alpha, R)
    for k in ("res_eq", "res_an", "raw"):
        p1 = pw_("single", f"Rminus_{k}_ecusum") >= 0.9
        p2 = (pw_("bilateral_equal", f"Rminus_{k}_ecusum") <= band_R[1] + 1e-9) and pw_("bilateral_equal", f"Rminus_{k}_eproc") <= alpha + 1e-9
        e_ = en.set_index("group")[f"anti_share_{k}"]
        p3 = bool(e_["bilateral_equal"] < e_["bilateral_unequal"] < e_["single"])            # projection-energy ordering (pre-registered)
        p_res[k] = (p1, p2, p3)
    rplus_ok = pw_("bilateral_equal", "Rplus_res_eq_sym") >= 0.9 and pw_("single", "Rplus_res_eq_sym") >= 0.9 and pw_("bilateral_unequal", "Rplus_res_eq_sym") >= 0.9
    _conclude(res_dir, "[e13c] isotypic on the residual (R- = flip test on Pi- r): " + "; ".join(f"{k}: single detect {p_res[k][0]}, bilateral-equal blind (e-CUSUM alarm <= band {band_R[1]:.2f} & e-process <= alpha) {p_res[k][1]}, anti-share ordering equal<unequal<single {p_res[k][2]}" for k in p_res)
              + " | e-CUSUM alarm fraction bilateral_equal: " + ", ".join(f"{k} {pw_('bilateral_equal', f'Rminus_{k}_ecusum'):.2f}" for k in p_res)
              + " | R- e-CUSUM median delay single/unequal: " + ", ".join(f"{k} {tab[(tab.group=='single')&(tab.detector==f'Rminus_{k}_ecusum')].delay_median.iloc[0]:.0f}/{tab[(tab.group=='bilateral_unequal')&(tab.detector==f'Rminus_{k}_ecusum')].delay_median.iloc[0]:.0f}" for k in p_res)
              + f" | R+ on Pi+ r_eq detects all three groups: {rplus_ok} ({pw_('single','Rplus_res_eq_sym'):.2f}/{pw_('bilateral_equal','Rplus_res_eq_sym'):.2f}/{pw_('bilateral_unequal','Rplus_res_eq_sym'):.2f})"
              + " | window rejection single/equal/unequal: " + ", ".join(f"{k} {wr('single', k)}/{wr('bilateral_equal', k)}/{wr('bilateral_unequal', k)}" for k in ("raw", "res_an", "res_eq"))
              + " | anti share: " + "; ".join(f"{r.group}: raw {r.anti_share_raw:.3f} res_an {r.anti_share_res_an:.3f} res_eq {r.anti_share_res_eq:.3f}" for r in en.itertuples())
              + " | isolation accuracy: " + ", ".join(f"{r.variant} {r.correct:.2f}" for r in acc.itertuples()))
    return {"tab": tab}


def wr_num(g, k, tab):
    return float(tab[(tab.group == g) & (tab.detector == f"Rminus_{k}_window_rejection")].power_100.iloc[0])


def _plot_c(cfg, res_dir, tab, en, conf, alpha):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    groups = list(cfg["stage_c"]["groups"].keys()); x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    dets = [("Rminus_raw_ecusum", "R⁻ raw (S1)", "C3"), ("Rminus_res_an_ecusum", "R⁻ analytic residual", "C1"), ("Rminus_res_eq_ecusum", "R⁻ equivariant residual (Π⁻r)", "C0"),
            ("Rplus_track", "R⁺ tracking (S2)", "0.5"), ("Rplus_res_eq_sym", "R⁺ equivariant residual (Π⁺r)", "C2")]
    w = 0.16
    for i, (d, lab, c) in enumerate(dets):
        v = [tab[(tab.group == g) & (tab.detector == d)].power_100.iloc[0] for g in groups]
        lo = [tab[(tab.group == g) & (tab.detector == d)].ci_lo.iloc[0] for g in groups]; hi = [tab[(tab.group == g) & (tab.detector == d)].ci_hi.iloc[0] for g in groups]
        axes[0].bar(x + (i - 2) * w, v, width=w, color=c, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=2, label=lab)
    axes[0].axhline(alpha, color="k", ls="--", lw=1); axes[0].set_xticks(x); axes[0].set_xticklabels(groups, fontsize=8); axes[0].set_ylabel("power (alarm within 100 cycles)"); axes[0].legend(fontsize=6); axes[0].set_title("e13c — power per channel", fontsize=9)
    for i, (k, lab, c) in enumerate((("raw", "raw Z", "C3"), ("res_an", "analytic residual", "C1"), ("res_eq", "equivariant residual", "C0"))):
        axes[1].bar(x + (i - 1) * 0.26, en[f"anti_share_{k}"], width=0.26, color=c, label=lab)
    axes[1].set_xticks(x); axes[1].set_xticklabels(groups, fontsize=8); axes[1].set_ylim(0, 1); axes[1].set_ylabel("antisymmetric energy share ‖Δ⁻‖²/(‖Δ⁻‖²+‖Δ⁺‖²)"); axes[1].legend(fontsize=6); axes[1].set_title("projection energy of the fault signature", fontsize=9)
    # confusion for the equivariant variant
    v = "equiv_rows"; cm = pd.crosstab(conf[conf.variant == v].truth, conf[conf.variant == v].pred)
    im = axes[2].imshow(cm.to_numpy(), cmap="Blues", aspect="auto"); axes[2].set_xticks(range(cm.shape[1])); axes[2].set_xticklabels(cm.columns, rotation=60, fontsize=6); axes[2].set_yticks(range(cm.shape[0])); axes[2].set_yticklabels(cm.index, fontsize=6)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, int(cm.iloc[i, j]), ha="center", va="center", fontsize=6, color="w" if cm.iloc[i, j] > cm.to_numpy().max() / 2 else "k")
    accs = conf.groupby("variant").correct.mean()
    axes[2].set_title("isolation confusion, equivariant residual rows\n" + ", ".join(f"{k}: {v:.2f}" for k, v in accs.items()), fontsize=8); axes[2].set_xlabel("predicted"); axes[2].set_ylabel("truth")
    fig.tight_layout(); fig.savefig(res_dir / "e13c_isotypic_isolation.png", dpi=140); plt.close(fig)


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
    stages = ["b", "a", "c"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"a": stage_a, "b": stage_b, "c": stage_c}[s](cfg, res_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E13 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
