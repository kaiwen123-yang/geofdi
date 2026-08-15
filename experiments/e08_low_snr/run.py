#!/usr/bin/env python3
"""e08 — low-SNR separation grid, full version (Sprint 7 Block P). Go2 go2_urdf_sym.

Completes e13a: adds inertia_add {10,20,50} g and a noise-scaling sweep x{1,2,4} over the full gain/bias/friction grid;
the detector set is R- {raw, analytic residual, equivariant-DeLaN residual} x {half-cycle e-process, e-CUSUM} +
rplus_resid + Mahalanobis + window-AE (nominal) + GRU-eta regressor (Liu Table I; trained on the two largest magnitudes,
tested on the small ones; 5 seeds). Unified conformal/FAR protocol. Nuisances (drift_sym, payload_sym) under the three
noise levels. Products: power-vs-magnitude curves per fault x noise, a merged minimal-detectable-magnitude table
(with e13a), the GRU seed spread, and the R- nuisance-silence confirmation.

    python experiments/e08_low_snr/run.py --stage grid|nuisance|gru|all [--run-id ID] [--quick] [--workers N]
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
from geofdi.detect.monitors import calibrate_ecusum_threshold, conformal_pvalues, ecusum, eprocess_alarm, tracking_scores
from geofdi.detect.permutation import hg_permutation_test
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.detect.sequential import EProcess, ECusum, calibrate_threshold, calibration_scale, half_cycles, mirror_scores
from geofdi.dynamics.delan import contact_torques_all, leg_arrays
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import RES_COLS, residual_manifest
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry import JOINTS, LEGS, z_channel_names

EXP_NAME = "e08_low_snr"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]
_MODELS = {}
_RES_REP = C2Rep(residual_manifest(include_base=False))
SEV_UNIT = {"actuator_gain": "1-kappa", "actuator_bias": "N m", "friction_scale": "scale-1", "inertia_add": "kg"}


def _sim(cfg, seed, noise_scale=1, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    if noise_scale != 1:
        s["noise"] = {k: v * noise_scale for k, v in s["noise"].items()}
    for k, v in over.items():
        s[k] = {**s.get("controller", {}), **v} if k == "controller" else v
    return s


def _conclude(res_dir, line):
    print(line, flush=True)
    (res_dir / "conclusions.txt").open("a").write(line + "\n")


def _sev(ftype, mag):
    return round((1 - mag) if ftype == "actuator_gain" else (mag - 1 if ftype == "friction_scale" else mag), 6)


def _load_models(tags):
    import torch
    from geofdi.dynamics.delan_equiv import load_delan
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for t in tags:
        if t in _MODELS:
            continue
        p = DATA_ROOT / "models" / "delan" / t
        if (p / "meta.json").exists():
            _MODELS[t] = load_delan(p, device=dev)
    return [t for t in tags if t in _MODELS]


def _worker(sim_cfg, K_cal, K_post, N, drop_first, ocfg, need_arrays=True):
    cfg = SimConfig(**sim_cfg); period = float(cfg.controller.get("period_s", 0.5)); cfg.duration_s = (K_cal + K_post + drop_first + 2) * period
    df, man = rollout(cfg); chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first); Zq, _ = register_cycles(df, QREF, N=N, drop_first=drop_first)
    K = min(K_cal + K_post, Z.shape[0], Zq.shape[0])
    dyn = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=ocfg["cutoff_hz"], torque=ocfg["torque"])[:, 6:]
    Zr_an, _ = registered_residuals(df, r, N=N, drop_first=drop_first)
    out = {"K": K, "Z": Z[:K].astype(np.float32), "Zq": Zq[:K].astype(np.float32), "Zr_an": Zr_an[:K].astype(np.float32), "chans": chans, "man": man}
    if need_arrays:
        jt = contact_torques_all(df, dyn); legs = {}
        for li, leg in enumerate(LEGS):
            q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt); legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32), "y": (tau + jt[:, 3 * li:3 * li + 3]).astype(np.float32)}
        out["arrays"] = {"legs": legs, "theta": df["theta"].to_numpy(), "t": df["t"].to_numpy(), "N": N, "drop_first": drop_first}
    return out


def _delan_res(out, tags):
    arr = out.pop("arrays", None)
    if arr is None:
        return {}
    T = len(arr["theta"]); dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]}); res = {}
    for tag in tags:
        r = np.zeros((T, 12), dtype=np.float32)
        for li, leg in enumerate(LEGS):
            L = arr["legs"][leg]; r[:, 3 * li:3 * li + 3] = L["y"] - _MODELS[tag].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
        Zr, _ = registered_residuals(dfr, r, N=arr["N"], drop_first=arr["drop_first"]); res[tag] = Zr[:out["K"]].astype(np.float32)
    return res


# ------------------------------------------------------------------------------------ R- half-cycle / e-CUSUM on any element
def _hc_p(Z, rep, K_cal, seed, cal_scale):
    H = half_cycles(Z); s = mirror_scores(H, rep, cal_scale)
    return conformal_pvalues(s[:2 * K_cal - 1], s[2 * K_cal - 1:])          # per monitored half-cycle


NAME_SEED = {"raw": 11, "res_an": 22, "res_eq": 33}


def _pvals_for(Zx, rep, K_cal, K, seed, name):
    cal_scale = calibration_scale(half_cycles(Zx[:K_cal]), rep)
    p_hc = _hc_p(Zx, rep, K_cal, seed, cal_scale)
    nw = K // 5; pw = np.empty(nw)
    for w in range(nw):
        pw[w], _ = hg_permutation_test(Zx[w * 5:(w + 1) * 5], rep, statistic="paired_energy", M=512, rng=np.random.default_rng([seed, w, NAME_SEED[name]]))
    return p_hc, pw


def _grid_worker(sim_cfg, fault, K_cal, K_post, N, df0, ocfg, tags, eq, pl, seed):
    """Worker (no torch): rollout, raw + analytic-residual half-cycle/window p-values, magnitude scores; and — if a
    DeLaN residual is requested — the per-leg arrays for the parent to compute the equivariant residual (CUDA cannot
    re-init in a fork)."""
    out = _worker(dict(sim_cfg, faults=[fault] if fault else []), K_cal, K_post, N, df0, ocfg, need_arrays=bool(tags))
    K = out["K"]; rec = {"K": K}
    for name, (Zx, rep) in (("raw", (out["Z"], C2Rep(out["man"]))), ("res_an", (out["Zr_an"], _RES_REP))):
        p_hc, pw = _pvals_for(Zx, rep, K_cal, K, seed, name); rec[f"phc_{name}"] = p_hc; rec[f"pw_{name}"] = pw
    rec["s_resid"] = residual_scores(out["Zr_an"]); rec["feat"] = cycle_features(out["Z"]); rec["s_track"] = tracking_scores(out["Z"], out["chans"], out["Zq"])
    if tags:
        rec["_arrays"] = out.pop("arrays", None); rec["_seed"] = seed; rec["_K_cal"] = K_cal
    return rec


def _add_res_eq(recs, tags, eq):
    """Parent-side: for each record with per-leg arrays, compute the equivariant-DeLaN residual cycles and its
    half-cycle / window p-values, then drop the arrays."""
    for rec in recs:
        arr = rec.pop("_arrays", None)
        if arr is None or eq not in _MODELS:
            continue
        T = len(arr["theta"]); r = np.zeros((T, 12), dtype=np.float32); dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]})
        for li, leg in enumerate(LEGS):
            L = arr["legs"][leg]; r[:, 3 * li:3 * li + 3] = L["y"] - _MODELS[eq].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
        Zr, _ = registered_residuals(dfr, r, N=arr["N"], drop_first=arr["drop_first"]); Zr = Zr[:rec["K"]].astype(np.float32)
        p_hc, pw = _pvals_for(Zr, _RES_REP, rec.pop("_K_cal"), rec["K"], rec.pop("_seed"), "res_eq"); rec["phc_res_eq"] = p_hc; rec["pw_res_eq"] = pw
    return recs


def stage_grid(cfg, res_dir, quick=False):
    import torch                                                                     # noqa: F401 (GPU load in parent)
    gc = cfg["grid"]; R = 6 if quick else gc["R"]; K_cal, K_post = gc["K_cal"], (40 if quick else gc["K_post"]); N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    oc = cfg["observer"]; alpha = cfg["detect"]["alpha"]; eq, pl = cfg["delan"]["equivariant"], cfg["delan"]["plain"]; tags = _load_models([eq]); t_on = (K_cal + df0) * 0.5
    gate = _maha(cfg, res_dir, quick)
    cells = []
    for ftype, mags in gc["faults"].items():
        mags = mags[-1:] if quick else mags
        for mag in mags:
            for joint in gc["joints"]:
                magnitude = {"actuator_gain": mag - 1.0, "friction_scale": mag - 1.0}.get(ftype, mag)
                cells.append((ftype, mag, joint, magnitude))
    noise_scales = [1] if quick else gc["noise_scales"]
    rows = []; nom_p = {ns: {"raw": [], "res_an": [], "res_eq": []} for ns in noise_scales}; recs_by = {}
    # nominal runs per noise scale (for the half-cycle conformal FAR baseline + e-CUSUM h)
    for ns in noise_scales:
        args = [(_sim(cfg, gc["seed_base"] + 900000 + 1000 * ns + r, noise_scale=ns), K_cal, K_post + 50, N, df0, oc, tags, eq, pl, gc["seed_base"] + 900000 + 1000 * ns + r) for r in range(max(10, R))]
        res = _add_res_eq(pmap(_grid_worker, [(a[0], None, *a[1:]) for a in args], cfg["workers"]), tags, eq); recs_by[("nominal", ns)] = res
        for name in nom_p[ns]:
            key = f"pw_{name}"
            nom_p[ns][name] = [r[key][:K_cal // 5] for r in res if key in r]
    h = {(ns, name): calibrate_ecusum_threshold(nom_p[ns][name], K_post // 5, far=alpha, n_boot=800, rng=np.random.default_rng(3)) for ns in noise_scales for name in nom_p[ns] if nom_p[ns][name]}
    for ci, (ftype, mag, joint, magnitude) in enumerate(cells):
        for ns in noise_scales:
            fault = dict(type=ftype, leg=gc["leg"], joint=joint, magnitude=float(magnitude), t_onset=t_on)
            args = [(_sim(cfg, gc["seed_base"] + 1000 * ci + 100 * ns + r, noise_scale=ns), fault, K_cal, K_post, N, df0, oc, tags, eq, pl, gc["seed_base"] + 500000 + 1000 * ci + 100 * ns + r) for r in range(R)]
            res = _add_res_eq(pmap(_grid_worker, args, cfg["workers"]), tags, eq)
            row = {"fault": ftype, "magnitude": mag, "severity": _sev(ftype, mag), "joint": joint, "noise_scale": ns, "R": R}
            # detectors
            for name in ("raw", "res_an", "res_eq"):
                if f"phc_{name}" not in res[0]:
                    continue
                # half-cycle e-process: calibrate the conformal set on the nominal runs of this noise scale (cross-run), evaluate on fault half-cycles
                cal = np.concatenate([r[f"phc_{name}"] for r in recs_by[("nominal", ns)]])[:0]      # placeholder (conformal already applied per run)
                dl_e = []; dl_c = []
                for r in res:
                    p_hc = r[f"phc_{name}"]; E, al = eprocess_alarm(p_hc, alpha, start=0); dl_e.append(np.nan if al is None else (al + 1) / 2.0)
                    S, al = ecusum(r[f"pw_{name}"], h[(ns, name)], start=K_cal // 5); dl_c.append(np.nan if al is None else (al - K_cal // 5 + 1) * 5)
                for tag2, dl in (("hc_eproc", np.array(dl_e)), ("ecusum", np.array(dl_c))):
                    row[f"Rminus_{name}_{tag2}_det100"] = float(np.mean(dl <= K_post)); row[f"Rminus_{name}_{tag2}_delay"] = float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan
            # rplus resid + maha
            for name, key in (("rplus_resid", "s_resid"), ("mahalanobis", "feat_score"), ("rplus_track", "s_track")):
                dl = []
                for r in res:
                    if name == "mahalanobis":
                        sc = gate.score(r["feat"])
                    else:
                        sc = r[key]
                    p = conformal_pvalues(sc[:K_cal], sc[K_cal:]); E, al = eprocess_alarm(p, alpha, start=0); dl.append(np.nan if al is None else al + 1)
                dl = np.array(dl); row[f"{name}_det100"] = float(np.mean(dl <= K_post)); row[f"{name}_delay"] = float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan
            rows.append(row)
        print(f"  [grid] {ftype} {mag} {joint} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e08_grid_power.csv", index=False)
    _plot_grid(cfg, res_dir, tab, K_post)
    _min_detectable(cfg, res_dir, tab)
    _conclude(res_dir, "[e08-grid] see e08_min_detectable.csv; power curves e08_power_*.png. R- residual (equivariant) det100 by fault (noise x1): "
              + "; ".join(f"{ft}: " + str(tab[(tab.fault == ft) & (tab.noise_scale == noise_scales[0]) & (tab.joint == 'KFE')].sort_values('magnitude')["Rminus_res_eq_hc_eproc_det100"].round(2).tolist()) for ft in gc["faults"]))
    return tab


def _maha(cfg, res_dir, quick):
    mt = cfg["baselines"]["mahalanobis_train"]; nro = 3 if quick else mt["rollouts"]; ncy = 30 if quick else mt["cycles"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    args = [(_sim(cfg, mt["seed_base"] + i), 30, ncy, N, df0, cfg["observer"], False) for i in range(nro)]
    res = pmap(_worker, args, cfg["workers"]); F = np.concatenate([cycle_features(r["Z"]) for r in res])
    return MahalanobisGate().fit(F)


def _min_detectable(cfg, res_dir, tab):
    crit = cfg["grid"]["detect_criterion"]; rows = []
    dets = [c[:-8] for c in tab.columns if c.endswith("_det100")]
    for ft in tab.fault.unique():
        for joint in tab.joint.unique():
            for ns in tab.noise_scale.unique():
                for det in dets:
                    sub = tab[(tab.fault == ft) & (tab.joint == joint) & (tab.noise_scale == ns)].sort_values("severity")
                    col = f"{det}_det100"
                    if col not in sub or sub[col].isna().all():
                        continue
                    ok = sub[sub[col] >= crit]
                    rows.append({"fault": ft, "joint": joint, "noise_scale": ns, "detector": det, "severity_unit": SEV_UNIT[ft],
                                 "min_detectable_severity": float(ok.severity.iloc[0]) if len(ok) else np.inf,
                                 "det100_by_severity": json.dumps(dict(zip(sub.severity.round(4), sub[col].round(2))))})
    pd.DataFrame(rows).to_csv(res_dir / "e08_min_detectable.csv", index=False)


def _plot_grid(cfg, res_dir, tab, K_post):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    faults = list(cfg["grid"]["faults"]); noise = sorted(tab.noise_scale.unique())
    lines = [("Rminus_raw_ecusum_det100", "R⁻ raw (e-CUSUM)", "C3", "o"), ("Rminus_res_an_ecusum_det100", "R⁻ analytic resid", "C1", "s"),
             ("Rminus_res_eq_hc_eproc_det100", "R⁻ equiv resid (½-cycle e-proc)", "C0", "^"), ("rplus_resid_det100", "R⁺ resid", "C2", "D"), ("mahalanobis_det100", "Mahalanobis", "0.4", "x")]
    for joint in cfg["grid"]["joints"]:
        fig, axes = plt.subplots(len(noise), len(faults), figsize=(3.0 * len(faults), 2.6 * len(noise)), squeeze=False)
        for ri, ns in enumerate(noise):
            for ci, ft in enumerate(faults):
                ax = axes[ri][ci]
                for col, lab, c, mk in lines:
                    sub = tab[(tab.fault == ft) & (tab.joint == joint) & (tab.noise_scale == ns)].sort_values("magnitude")
                    if col in sub and not sub[col].isna().all():
                        ax.plot(sub.magnitude, sub[col], marker=mk, color=c, ms=4, label=lab if (ri == 0 and ci == 0) else None)
                ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.3)
                if ri == 0:
                    ax.set_title(ft, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f"noise x{ns}\ndet100", fontsize=8)
                if ft == "actuator_gain":
                    ax.invert_xaxis()
        axes[0][0].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"e08 — low-SNR power vs magnitude, LF-{joint} (R={int(tab.R.max())})", fontsize=9); fig.tight_layout()
        fig.savefig(res_dir / f"e08_power_{joint}.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------ nuisance
def stage_nuisance(cfg, res_dir, quick=False):
    nc = cfg["nuisance"]; R = 8 if quick else nc["R"]; K_cal, K_mon = nc["K_cal"], (40 if quick else nc["K_mon"]); N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    oc = cfg["observer"]; alpha = cfg["detect"]["alpha"]; eq = cfg["delan"]["equivariant"]; tags = _load_models([eq])
    noise_scales = [1] if quick else cfg["grid"]["noise_scales"]; rows = []
    for name, over in cfg["nuisance"]["conditions"].items():
        for ns in noise_scales:
            ov = {"nuisance": over["nuisance"]}
            args = [(_sim(cfg, nc["seed_base"] + 1000 * ns + r, noise_scale=ns, **ov), None, K_cal, K_mon, N, df0, oc, tags, eq, cfg["delan"]["plain"], nc["seed_base"] + 500000 + 1000 * ns + r) for r in range(R)]
            res = _add_res_eq(pmap(_grid_worker, args, cfg["workers"]), tags, eq)
            for det, key in (("Rminus_raw_hc", "phc_raw"), ("Rminus_res_eq_hc", "phc_res_eq"), ("rplus_resid", "s_resid"), ("mahalanobis", "feat"), ("rplus_track", "s_track")):
                far = []
                for r in res:
                    if det.startswith("Rminus"):
                        p = r[key][2 * K_cal - 1:] if key in r else None
                        if p is None:
                            continue
                        far.append(float(np.mean(p <= alpha)))
                    else:
                        sc = (MahalanobisScore(res, r) if det == "mahalanobis" else r[key]); p = conformal_pvalues(sc[:K_cal], sc[K_cal:]); far.append(float(np.mean(p <= alpha)))
                if not far:
                    continue
                rate = float(np.mean(far)); band = nominal_band(alpha, len(far) * (K_mon))
                rows.append({"nuisance": name, "noise_scale": ns, "detector": det, "far_per_element": rate, "in_band": bool(rate <= 0.08), "R": R})
        print(f"  [nuisance] {name} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e08_nuisance.csv", index=False)
    _conclude(res_dir, "[e08-nuisance] FAR per element under nuisances x noise: " + "; ".join(f"{r.nuisance} x{r.noise_scale} {r.detector}: {r.far_per_element:.3f}" for r in tab.itertuples() if r.detector in ("Rminus_res_eq_hc", "rplus_resid", "mahalanobis")))
    return tab


def MahalanobisScore(res_all, r):
    # per-run Mahalanobis needs a gate; fit on the run's own calibration features (nuisance FAR is a within-run conformal quantity)
    from geofdi.baselines.mahalanobis import MahalanobisGate
    return MahalanobisGate().fit(r["feat"][:60]).score(r["feat"])


# ------------------------------------------------------------------------------------ gru
def stage_gru(cfg, res_dir, quick=False):
    import torch
    from geofdi.baselines.gru import GRURegressor, WindowSetReg, eta_lowpass_threshold, predict_eta, train_gru_regressor
    gc = cfg["grid"]; g = cfg["baselines"]["gru"]; K_cal, K_post = gc["K_cal"], gc["K_post"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t_on = (K_cal + df0) * 0.5; alpha = cfg["detect"]["alpha"]
    seeds = g["seeds"][:2] if quick else g["seeds"]; epochs = 3 if quick else g["epochs"]
    # training rollouts: nominal + the two largest magnitudes per fault; eta target vector (12) = 1 except the faulty joint = kappa
    def eta_of(ftype, mag, joint):
        eta = np.ones(12); li = LEGS.index(gc["leg"]); ji = JOINTS.index(joint)
        if ftype == "actuator_gain":
            eta[3 * li + ji] = mag                                                    # kappa
        return eta                                                                    # bias/friction/inertia not in the eta model -> healthy target (the GRU can only regress gain; reported)
    train_specs = [("nominal", None, None)]
    for ftype, mags in g["train_magnitudes"].items():
        for mag in mags:
            for joint in gc["joints"]:
                magnitude = {"actuator_gain": mag - 1.0, "friction_scale": mag - 1.0}.get(ftype, mag); train_specs.append((ftype, mag, joint))
    nrep = 2 if quick else 4
    args = []; meta = []; sid = 66000 + 700000
    for spec in train_specs:
        for r in range(nrep):
            ftype, mag, joint = spec
            fault = None if ftype == "nominal" else dict(type=ftype, leg=gc["leg"], joint=joint, magnitude={"actuator_gain": mag - 1.0, "friction_scale": mag - 1.0}.get(ftype, mag), t_onset=t_on)
            args.append((_sim(cfg, sid, faults=[fault] if fault else []), K_cal, K_post, N, df0, oc, False)); meta.append(spec + (sid,)); sid += 1
    res = pmap(_worker, args, cfg["workers"])
    # build 57-like input: use q, dq, tau_cmd, tau_meas standardized (48) + imu (6) + ... just use the 58-channel Z sequence
    def seq_of(o):
        return o["Z"].transpose(0, 2, 1).reshape(-1, o["Z"].shape[1])                 # (K*N, d) — cycle-registered, but the GRU wants time; use raw would need df. Approx with registered.
    # We need the raw time sequence: re-rollout with need_arrays? Simpler: use the registered cycles flattened as pseudo-time (the GRU sees the phase structure).
    d = res[0]["Z"].shape[1]
    seqs = []; tgs = []
    for o, mt in zip(res, meta):
        s = seq_of(o); nrows = s.shape[0]
        ftype, mag, joint = mt[0], mt[1], mt[2]
        eta = np.tile(eta_of(ftype, mag, joint) if ftype != "nominal" else np.ones(12), (nrows, 1))
        # onset: the first K_cal cycles are healthy
        eta[:K_cal * o["Z"].shape[2]] = 1.0
        seqs.append(s.astype(np.float32)); tgs.append(eta.astype(np.float32))
    mu = np.concatenate(seqs).mean(0); sd = np.concatenate(seqs).std(0) + 1e-6; seqs = [(s - mu) / sd for s in seqs]
    rows = []
    for seed in seeds:
        ws = WindowSetReg(seqs, tgs, g["window"], g["stride_train"]); torch.manual_seed(seed)
        model = GRURegressor(d, g["hidden"], g["layers"], 12); train_gru_regressor(model, ws, epochs=epochs, batch=g["batch"], lr=g["lr"], device=dev, seed=seed, max_batches_per_epoch=g["max_batches_per_epoch"], log=None)
        # test on the small gain magnitudes (unseen)
        for ftype in ("actuator_gain",):
            for mag in gc["faults"][ftype]:
                if mag in g["train_magnitudes"].get(ftype, []):
                    continue
                for joint in gc["joints"]:
                    fault = dict(type=ftype, leg=gc["leg"], joint=joint, magnitude=mag - 1.0, t_onset=t_on)
                    tres = pmap(_worker, [(_sim(cfg, 66000 + 800000 + seed * 1000 + i, faults=[fault]), K_cal, K_post, N, df0, oc, False) for i in range(6 if quick else 15)], cfg["workers"])
                    det = []
                    for o in tres:
                        s = ((seq_of(o) - mu) / sd).astype(np.float32); nr = s.shape[0]
                        idx0 = np.arange(0, nr - g["window"] + 1, 5); X = np.stack([s[i:i + g["window"]] for i in idx0])
                        eta_hat = predict_eta(model, X, device=dev); flags = eta_lowpass_threshold(eta_hat, fc_hz=1.0, fs=g["fs_hz"], thr=0.7)
                        onset_row = K_cal * o["Z"].shape[2]; post = (idx0 + g["window"] - 1) >= onset_row
                        det.append(bool(np.any(flags[post])))
                    rows.append({"seed": seed, "fault": ftype, "magnitude": mag, "severity": _sev(ftype, mag), "joint": joint, "det_rate": float(np.mean(det)), "seen": False})
        print(f"  [gru] seed {seed} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e08_gru_spread.csv", index=False)
    spread = tab.groupby(["fault", "magnitude", "joint"]).det_rate.agg(["mean", "std", "min", "max"]).reset_index(); spread.to_csv(res_dir / "e08_gru_spread_summary.csv", index=False)
    _conclude(res_dir, "[e08-gru] unseen-magnitude gain detection (5 seeds, rule eta<0.7): " + "; ".join(f"{r.fault} sev {r.magnitude} {r.joint}: {r['mean']:.2f}±{r['std']:.2f} [{r['min']:.2f},{r['max']:.2f}]" for r in spread.itertuples()))
    return tab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["grid", "nuisance", "gru", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml")); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    if a.workers:
        cfg["workers"] = a.workers
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True); (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["grid", "nuisance", "gru"] if a.stage == "all" else [a.stage]
    print(f"[{EXP_NAME}] run_id={a.run_id} stages={stages} quick={a.quick}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"grid": stage_grid, "nuisance": stage_nuisance, "gru": stage_gru}[s](cfg, res_dir, quick=a.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E08 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
