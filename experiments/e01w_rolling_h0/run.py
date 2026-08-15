#!/usr/bin/env python3
"""e01-W — H0 machinery for the wheeled M1 in rolling mode (Sprint 7, Block W2). Σ = G (pure reflection), data
elements = fixed-duration blocks of L seconds resampled on N points (phase.registration.register_blocks); the flip test,
statistics, e-process and H0' machinery of the trot are reused unchanged (C2Rep with delta_theta = 0).

  a  exactness: R replicates x 3 speeds; K = 60 blocks at L in {0.5, 1, 2} s cut from one rollout each -> p-value QQ (KS),
     size table at alpha in {0.01, 0.05, 0.1}, lag-1 autocorrelation of the per-block antisymmetric energy (the
     exchangeability diagnostic) -> minimal exchangeable L; original-world (chiral) column at 1 m/s = the eps_dyn effect.
  b  eps_ctrl: single-side wheel-rate gain 1.02 / single-side HIP kp 1.02 -> H0 size (inflated) vs the H0' differenced
     test (X_k = Z^mon_k - Z^cal_k, exact for any stable asymmetry, Part 2 Lemma centring (iv)) size; change run: the
     asymmetry doubles mid-monitoring -> alarm fraction / delay of the differenced e-process.
  c  nuisance / fault snapshot: payload sym 1 kg, lateral payload 0.5 kg, single wheel friction x0.7, single wheel motor
     kappa 0.8, single HIP kappa 0.8 -> R- e-process / e-CUSUM timelines (window 5 blocks) after 60 calibration blocks.
  d  e13d: equivariant DeLaN (per-leg 4-joint template pair, target tau_cmd, wheel angle zeroed) trained on nominal
     rolling data -> residual R- size under H0 and power on the wheel-motor fault; nuisance readings.

    python experiments/e01w_rolling_h0/run.py --stage a|b|c|d|all [--run-id ID] [--quick] [--workers N]
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
from scipy import stats

from geofdi.detect.monitors import calibrate_ecusum_threshold, ecusum, eprocess_alarm
from geofdi.detect.permutation import MirrorStatistics, hg_permutation_test, hg_permutation_tests, pooled_scale, random_flips
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_blocks, straight_mask
from geofdi.sim.env_m1 import SimConfigM1, rollout_m1
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry_m1 import JOINTS, LEGS, z_channel_names

EXP_NAME = "e01w_rolling_h0"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])


# ------------------------------------------------------------------------------------ helpers
def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed)
    for k, v in over.items():
        s[k] = {**s.get("controller", {}), **v} if k == "controller" else v
    return s


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


def _blocks(sim_cfg, K, L, N, warm=None):
    """Rollout long enough for K blocks of L s after warm-up; returns Z (K, d, N), manifest, chans, df."""
    cfg = SimConfigM1(**sim_cfg)
    warm = float(cfg.warmup_s if warm is None else warm)
    cfg.duration_s = warm + K * L + 2 * L
    df, man = rollout_m1(cfg)
    chans = z_channel_names(man)
    Z, meta = register_blocks(df, chans, L_s=L, N=N, mask=straight_mask(df, warmup_s=warm), max_blocks=K)
    return Z, man, chans, df


def _flip_p(Z, rep, M, seed, statistics=("paired_energy", "energy_distance")):
    r = hg_permutation_tests(Z, rep, statistics=statistics, M=M, rng=np.random.default_rng(seed))
    return {s: float(r[s]["p"]) for s in statistics}


def _lag1(Z, rep):
    """Lag-1 autocorrelation of the per-block antisymmetric energy ||Pi- Z_k||^2 (standardized channels)."""
    Zs = rep.apply("s", Z); sc = pooled_scale(Z, Zs); D = ((Z - Zs) / sc)
    e = (D ** 2).sum(axis=(1, 2))
    if len(e) < 3:
        return float("nan")
    e = e - e.mean(); return float((e[:-1] * e[1:]).sum() / ((e ** 2).sum() + 1e-12))


def _differenced_test(Zc, Zm, rep, M, seed):
    """H0' flip test on paired differences X_k = Z^mon_k - Z^cal_k (K = min); paired-energy statistic."""
    K = min(len(Zc), len(Zm)); X = Zm[:K] - Zc[:K]
    p, obs = hg_permutation_test(X, rep, statistic="paired_energy", M=M, rng=np.random.default_rng(seed))
    return float(p)


# ------------------------------------------------------------------------------------ stage a
def _rep_a(sim_cfg, K, L_list, N, M, seed, alphas):
    Lmax = max(L_list)
    cfg = SimConfigM1(**sim_cfg); warm = float(cfg.warmup_s); cfg.duration_s = warm + K * Lmax + 2 * Lmax
    df, man = rollout_m1(cfg); chans = z_channel_names(man); rep = C2Rep(man); out = {}
    mask = straight_mask(df, warmup_s=warm)
    for L in L_list:
        Z, meta = register_blocks(df, chans, L_s=L, N=N, mask=mask, max_blocks=K)
        out[L] = {"p": _flip_p(Z, rep, M, seed), "lag1": _lag1(Z, rep), "K": int(Z.shape[0])}
    return out


def stage_a(cfg, res_dir, quick=False):
    sa = cfg["stage_a"]; R = 12 if quick else sa["R"]; K = sa["K"]; N = cfg["registration"]["N"]; T = cfg["test"]
    L_list = [float(x) for x in sa["L_list"]]; rows = []; pvals = {}
    conds = [("m1_wheeled_sym", sp) for sp in sa["speeds"]] + [("m1_wheeled", sa["original_world_speed"])]
    for ci, (model, sp) in enumerate(conds):
        args = [(_sim(cfg, sa["seed_base"] + 100 * ci + r, model=model, speed=float(sp)), K, L_list, N, T["M"], sa["seed_base"] + 90000 + 100 * ci + r, T["alphas"]) for r in range(R)]
        res = pmap(_rep_a, args, cfg["workers"])
        for L in L_list:
            for stat in T["statistics"]:
                p = np.array([r[L]["p"][stat] for r in res]); pvals[(model, sp, L, stat)] = p
                ks = stats.kstest(p, "uniform").pvalue; lag = np.nanmean([r[L]["lag1"] for r in res])
                rec = {"model": model, "speed": sp, "L_s": L, "statistic": stat, "R": R, "K": K, "ks_p": float(ks), "lag1_anti_energy": float(lag), "block_min_K": int(min(r[L]["K"] for r in res))}
                for a in T["alphas"]:
                    k = int(np.sum(p <= a)); band = nominal_band(a, R)
                    rec[f"size_{a}"] = k / R; rec[f"in_band_{a}"] = bool(band[0] <= k / R <= band[1]); rec[f"band_{a}"] = f"[{band[0]:.3f},{band[1]:.3f}]"
                rows.append(rec)
        print(f"  [a] {model} {sp} m/s done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01w_a_size_table.csv", index=False)
    # QQ figure: uniform QQ of p-values per L (sym world, 1 m/s) + size vs L per speed
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    q = (np.arange(R) + 0.5) / R
    for L, c in zip(L_list, ("C0", "C1", "C2")):
        for stat, ls in (("paired_energy", "-"), ("energy_distance", "--")):
            p = np.sort(pvals[("m1_wheeled_sym", 1.0, L, stat)]); axes[0].plot(q, p, ls, color=c, label=f"L={L} s {stat}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1); axes[0].set_xlabel("uniform quantile"); axes[0].set_ylabel("p-value quantile"); axes[0].set_title("QQ under H0, m1_wheeled_sym 1 m/s (R=%d, K=%d)" % (R, K), fontsize=9); axes[0].legend(fontsize=6)
    for (model, sp), mk in zip(conds, "osd^"):
        sub = tab[(tab.model == model) & (tab.speed == sp) & (tab.statistic == "paired_energy")].sort_values("L_s")
        axes[1].plot(sub.L_s, sub["size_0.05"], marker=mk, label=f"{model} {sp} m/s")
    band = nominal_band(0.05, R); axes[1].axhspan(band[0], band[1], color="0.9"); axes[1].axhline(0.05, color="k", ls=":", lw=1)
    axes[1].set_xlabel("block length L [s]"); axes[1].set_ylabel("size at α = 0.05 (paired energy)"); axes[1].set_title("size vs block length (exchangeability); chiral world column", fontsize=9); axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3)
    for (model, sp), mk in zip(conds, "osd^"):
        sub = tab[(tab.model == model) & (tab.speed == sp) & (tab.statistic == "paired_energy")].sort_values("L_s")
        axes[2].plot(sub.L_s, sub.lag1_anti_energy, marker=mk, label=f"{model} {sp} m/s")
    axes[2].axhline(0, color="k", ls=":", lw=1); axes[2].set_xlabel("block length L [s]"); axes[2].set_ylabel("lag-1 autocorrelation of ‖Π⁻Z_k‖²"); axes[2].set_title("block-to-block dependence", fontsize=9); axes[2].legend(fontsize=6); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(res_dir / "e01w_a_qq_size.png", dpi=140); plt.close(fig)
    sym = tab[(tab.model == "m1_wheeled_sym") & (tab.statistic == "paired_energy")]
    okL = sorted({L for L in L_list if sym[sym.L_s == L]["in_band_0.05"].all()})
    minL = okL[0] if okL else None
    orig = tab[(tab.model == "m1_wheeled") & (tab.statistic == "paired_energy")]
    _conclude(res_dir, f"[e01w-a] size at 0.05 (paired energy), sym world by speed x L: " + "; ".join(f"{r.speed} m/s L={r.L_s}: {r['size_0.05']:.3f}{'' if r['in_band_0.05'] else ' OUT'} (KS {r.ks_p:.2f}, lag1 {r.lag1_anti_energy:+.2f})" for _, r in sym.iterrows())
              + f" | minimal L with all speeds in band: {minL} | original (chiral) world at {sa['original_world_speed']} m/s: " + "; ".join(f"L={r.L_s}: {r['size_0.05']:.3f}" for _, r in orig.iterrows()))
    return tab


# ------------------------------------------------------------------------------------ stage b
def _rep_b(sim_cfg_asym, K_cal, K_mon, L, N, M, seed, window, doubling_cfg=None):
    """One replicate: rollout with the asymmetry (constant, or doubling at monitored block `at_block`); returns H0 p on
    the monitoring blocks, differenced-test p, and the differenced e-process alarm index (windows) if doubling."""
    cfg = SimConfigM1(**sim_cfg_asym); warm = float(cfg.warmup_s); cfg.duration_s = warm + (K_cal + K_mon) * L + 2 * L
    if doubling_cfg is not None:
        t_change = warm + (K_cal + doubling_cfg["at_block"]) * L
        asym = []
        for a in cfg.controller.get("asymmetry", []):
            a1 = dict(a); a1["t_end"] = t_change; asym.append(a1)
            a2 = dict(a); a2["t_start"] = t_change
            for key in ("rate_gain", "kp_gain"):
                if key in a2:
                    a2[key] = 1.0 + doubling_cfg["factor"] * (a2[key] - 1.0)
            asym.append(a2)
        cfg.controller = {**cfg.controller, "asymmetry": asym}
    df, man = rollout_m1(cfg); chans = z_channel_names(man); rep = C2Rep(man)
    Z, meta = register_blocks(df, chans, L_s=L, N=N, mask=straight_mask(df, warmup_s=warm), max_blocks=K_cal + K_mon)
    Zc, Zm = Z[:K_cal], Z[K_cal:K_cal + K_mon]
    out = {"p_h0_mon": _flip_p(Zm, rep, M, seed, ("paired_energy",))["paired_energy"], "p_h0_cal": _flip_p(Zc, rep, M, seed + 1, ("paired_energy",))["paired_energy"],
           "p_diff": _differenced_test(Zc, Zm, rep, M, seed + 2)}
    # differenced e-process over windows of `window` monitored blocks vs the matching calibration blocks
    nw = min(K_cal, K_mon) // window; ps = []
    for w in range(nw):
        X = Zm[w * window:(w + 1) * window] - Zc[w * window:(w + 1) * window]
        p, _ = hg_permutation_test(X, rep, statistic="paired_energy", M=M, rng=np.random.default_rng([seed, 5, w])); ps.append(float(p))
    out["p_diff_windows"] = ps
    return out


def stage_b(cfg, res_dir, quick=False):
    sb = cfg["stage_b"]; R = 6 if quick else sb["R"]; L = sb["L"]; N = cfg["registration"]["N"]; T = cfg["test"]; alpha = T["alpha"]
    K_cal, K_mon, window = sb["K_cal"], sb["K_mon"], sb["window"]; rows = []
    conds = [("nominal", [])] + [(name, asym) for name, asym in sb["asymmetries"].items()]
    for ci, (name, asym) in enumerate(conds):
        for doubling in ([None] if name == "nominal" else [None, sb["doubling"]]):
            args = [(_sim(cfg, sb["seed_base"] + 100 * ci + r, controller={"asymmetry": asym}), K_cal, K_mon, L, N, T["M"], sb["seed_base"] + 90000 + 100 * ci + r, window, doubling) for r in range(R)]
            res = pmap(_rep_b, args, cfg["workers"])
            tag = name + ("_x2" if doubling else "")
            for key in ("p_h0_mon", "p_h0_cal", "p_diff"):
                p = np.array([r[key] for r in res]); k = int(np.sum(p <= alpha)); band = nominal_band(alpha, R)
                rows.append({"condition": tag, "test": key, "R": R, "size_or_power": k / R, "ci_lo": binom_ci(k, R)[0], "ci_hi": binom_ci(k, R)[1], "band_lo": band[0], "band_hi": band[1], "in_band": bool(band[0] <= k / R <= band[1])})
            # e-process on differenced windows
            alarms = []
            for r in res:
                E, al = eprocess_alarm(np.array(r["p_diff_windows"]), alpha, start=0); alarms.append(None if al is None else (al + 1) * window)
            dl = np.array([np.nan if a is None else a for a in alarms], dtype=float)
            rows.append({"condition": tag, "test": "diff_eprocess_alarm", "R": R, "size_or_power": float(np.mean(~np.isnan(dl))), "ci_lo": np.nan, "ci_hi": np.nan, "band_lo": np.nan, "band_hi": np.nan, "in_band": None,
                         "delay_median_blocks": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan})
            print(f"  [b] {tag} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01w_b_h0prime.csv", index=False)
    def g(c, t, col="size_or_power"): return float(tab[(tab.condition == c) & (tab.test == t)][col].iloc[0])
    lines = []
    for name in sb["asymmetries"]:
        lines.append(f"{name}: H0 size on monitoring blocks {g(name,'p_h0_mon'):.2f} (nominal {g('nominal','p_h0_mon'):.2f}), differenced H0' size {g(name,'p_diff'):.2f}; doubling run: differenced test power {g(name+'_x2','p_diff'):.2f}, e-process alarm fraction {g(name+'_x2','diff_eprocess_alarm'):.2f} (median delay {tab[(tab.condition==name+'_x2')&(tab.test=='diff_eprocess_alarm')].delay_median_blocks.iloc[0]:.0f} blocks; nominal-asymmetry alarm {g(name,'diff_eprocess_alarm'):.2f})")
    _conclude(res_dir, "[e01w-b] " + " | ".join(lines))
    return tab


# ------------------------------------------------------------------------------------ stage c
def _rep_c(sim_cfg, K_cal, K_post, L, N, M, seed, window):
    cfg = SimConfigM1(**sim_cfg); warm = float(cfg.warmup_s); cfg.duration_s = warm + (K_cal + K_post) * L + 2 * L
    df, man = rollout_m1(cfg); chans = z_channel_names(man); rep = C2Rep(man)
    Z, meta = register_blocks(df, chans, L_s=L, N=N, mask=straight_mask(df, warmup_s=warm), max_blocks=K_cal + K_post)
    nw = Z.shape[0] // window; ps = np.empty(nw)
    for w in range(nw):
        p, _ = hg_permutation_test(Z[w * window:(w + 1) * window], rep, statistic="paired_energy", M=M, rng=np.random.default_rng([seed, w])); ps[w] = p
    return {"p": ps, "K": int(Z.shape[0])}


def stage_c(cfg, res_dir, quick=False):
    sc = cfg["stage_c"]; R = 6 if quick else sc["R"]; L = sc["L"]; N = cfg["registration"]["N"]; T = cfg["test"]; alpha = T["alpha"]
    K_cal, K_post, window = sc["K_cal"], sc["K_post"], sc["window"]
    warm = float(cfg["sim"]["warmup_s"]); t_on = warm + K_cal * L
    conds = [("nominal", {})] + list(sc["conditions"].items())
    results = {}; nom_p = []
    for ci, (name, over) in enumerate(conds):
        ov = {}
        for key in ("faults", "nuisance"):
            if key in over:
                ov[key] = [dict(f, t_onset=t_on) for f in over[key]]
        args = [(_sim(cfg, sc["seed_base"] + 100 * ci + r, **ov), K_cal, K_post, L, N, T["M"], sc["seed_base"] + 90000 + 100 * ci + r, window) for r in range(R)]
        res = pmap(_rep_c, args, cfg["workers"]); results[name] = res
        nom_p += [r["p"][:K_cal // window] for r in res]
        print(f"  [c] {name} done", flush=True)
    h = calibrate_ecusum_threshold(nom_p, K_post // window, far=alpha, n_boot=1000, rng=np.random.default_rng(2))
    w0 = K_cal // window; rows = []; timelines = {}
    for name, res in results.items():
        d_e, d_c, wr = [], [], []
        for r in res:
            E, al = eprocess_alarm(r["p"], alpha, start=w0); d_e.append(np.nan if al is None else (al - w0 + 1) * window)
            S, al = ecusum(r["p"], h, start=w0); d_c.append(np.nan if al is None else (al - w0 + 1) * window)
            wr.append(np.mean(r["p"][w0:] <= alpha))
        d_e = np.array(d_e); d_c = np.array(d_c)
        rows.append({"condition": name, "R": R, "eproc_alarm_frac": float(np.mean(~np.isnan(d_e))), "eproc_delay_median_blocks": float(np.nanmedian(d_e)) if np.isfinite(d_e).any() else np.nan,
                     "ecusum_alarm_frac": float(np.mean(~np.isnan(d_c))), "ecusum_delay_median_blocks": float(np.nanmedian(d_c)) if np.isfinite(d_c).any() else np.nan,
                     "window_rejection_post": float(np.mean(wr)), "ecusum_h": h})
        timelines[name] = np.mean([np.log(np.maximum(1e-3, r["p"])) for r in res], axis=0)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01w_c_snapshot.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name, tl in timelines.items():
        ax.plot((np.arange(len(tl)) + 1) * window - K_cal, tl, label=name)
    ax.axvline(0, color="k", ls="--", lw=1); ax.axhline(np.log(alpha), color="0.5", ls=":", lw=1); ax.set_xlabel("blocks after onset (1 s each)"); ax.set_ylabel("mean log p (5-block windows)")
    ax.set_title("e01w-c — R⁻ window p timelines on the rolling M1 (mean over R=%d)" % R, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(res_dir / "e01w_c_timelines.png", dpi=140); plt.close(fig)
    _conclude(res_dir, "[e01w-c] R- alarm fraction (e-process / e-CUSUM at FAR 0.05, 100 monitored blocks) and median delay [blocks]: "
              + "; ".join(f"{r.condition}: {r.eproc_alarm_frac:.2f}/{r.ecusum_alarm_frac:.2f} (delay {r.eproc_delay_median_blocks:.0f}/{r.ecusum_delay_median_blocks:.0f})" for r in tab.itertuples()))
    return tab


# ------------------------------------------------------------------------------------ stage d (e13d)
def _leg_arrays_m1(df, leg, dt=0.005):
    from scipy.signal import savgol_filter
    q = df[[f"q_{leg}_{j}" for j in JOINTS]].to_numpy().copy(); q[:, 3] = 0.0            # wheel angle: rotational symmetry -> zeroed input
    dq = df[[f"dq_{leg}_{j}" for j in JOINTS]].to_numpy(); ddq = savgol_filter(dq, 7, 2, deriv=1, delta=dt, axis=0)
    a = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); tau = df[[f"tau_cmd_{leg}_{j}" for j in JOINTS]].to_numpy()
    return q, dq, ddq, a, tau


def _rollout_arrays_d(sim_cfg, seed):
    cfg = SimConfigM1(**sim_cfg); cfg.seed = int(seed); df, man = rollout_m1(cfg)
    df = df[df.t > cfg.warmup_s]
    out = {}
    for leg in LEGS:
        q, dq, ddq, a, tau = _leg_arrays_m1(df, leg, cfg.ctrl_dt); out[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32), "y": tau.astype(np.float32)}
    out["n"] = len(df); return out


def _rep_d(sim_cfg, K_cal, K_post, L, N, seed):
    """Rollout + registered raw blocks + per-leg arrays (DeLaN residual computed in the parent)."""
    cfg = SimConfigM1(**sim_cfg); warm = float(cfg.warmup_s); cfg.duration_s = warm + (K_cal + K_post) * L + 2 * L
    df, man = rollout_m1(cfg); chans = z_channel_names(man)
    mask = straight_mask(df, warmup_s=warm)
    Z, meta = register_blocks(df, chans, L_s=L, N=N, mask=mask, max_blocks=K_cal + K_post)
    legs = {}
    for leg in LEGS:
        q, dq, ddq, a, tau = _leg_arrays_m1(df, leg, cfg.ctrl_dt); legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32), "y": tau.astype(np.float32)}
    return {"Z": Z.astype(np.float32), "man": man, "chans": chans, "t": df["t"].to_numpy(), "mask": mask, "legs": legs, "K": int(Z.shape[0])}


def stage_d(cfg, res_dir, quick=False):
    import torch
    from geofdi.dynamics.delan_equiv import EquivariantDeLaN, equivariance_defect, load_delan, train_equivariant
    from geofdi.residuals.mirror_pairs import residual_manifest
    from geofdi.sim.telemetry_m1 import build_manifest as build_manifest_m1
    sd = cfg["stage_d"]; tr = sd["train"]; N = cfg["registration"]["N"]; T = cfg["test"]; alpha = T["alpha"]; L = sd["L"]
    K_cal, K_post, window = sd["K_cal"], sd["K_post"], sd["window"]; warm = float(cfg["sim"]["warmup_s"]); t_on = warm + K_cal * L
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = DATA_ROOT / "models" / "delan_m1" / ("equiv_rolling_quick" if quick else "equiv_rolling_v1")
    man_m1 = build_manifest_m1()
    if not (mdir / "meta.json").exists():
        nro = 2 if quick else tr["rollouts_per_speed"]; dur = 15.0 if quick else tr["duration_s"]
        args = []; sid = tr["seed_base"]
        for sp in tr["speeds"]:
            for i in range(nro):
                args.append((dict(_sim(cfg, sid, speed=float(sp)), duration_s=dur), sid)); sid += 1
        res = pmap(_rollout_arrays_d, args, cfg["workers"])
        nva = max(1, len(res) // 5); data = {}
        for leg in LEGS:
            data[leg] = {"train": {k: np.concatenate([r[leg][k] for r in res[nva:]]) for k in ("q", "dq", "ddq", "a", "y")},
                         "val": {k: np.concatenate([r[leg][k] for r in res[:nva]]) for k in ("q", "dq", "ddq", "a", "y")}}
        quad = EquivariantDeLaN.build(n_templates=2, hidden=128, depth=3, eps=1e-3, damping=0.05, frictionloss=0.0, device=dev, manifest=man_m1, robot="m1_wheeled")
        rep = train_equivariant(quad, data, epochs=(3 if quick else tr["epochs"]), batch=tr["batch"], lr=tr["lr"], device=dev, log=lambda m: print(m, flush=True),
                                q_sd_floor=0.1)     # rolling: the legs barely move (q std 0.002 rad) -> floor the input scale
        quad.meta.update({"tag": mdir.name, "robot": "m1_wheeled", "target": "tau_cmd (no contact term; quasi-static rolling)", "wheel_angle_input": "zeroed"})
        quad.save(mdir); v = data["LF"]["val"]
        d = equivariance_defect(quad, v["q"], v["dq"], v["ddq"], v["a"], pairs=[("LF", "RF")], maps=quad.maps)
        rep["defect_q95"] = d["q95"]; rep["n_train_per_leg"] = int(len(data["LF"]["train"]["q"]))
        (mdir / "report.json").write_text(json.dumps({k: (v if k != "templates" else {kk: {"final_val_rmse_per_joint": vv["final_val_rmse_per_joint"]} for kk, vv in v.items()}) for k, v in rep.items()}, indent=1, default=float))
        print(f"  [d] equivariant DeLaN (M1 rolling) trained: val rmse per leg " + str({l: np.round(rep['legs'][l]['final_val_rmse_per_joint'], 3).tolist() for l in LEGS}) + f"; delta_f q95 {d['q95']:.2e}", flush=True)
    quad = load_delan(mdir, device=dev)
    res_rep = C2Rep(residual_manifest_m1())
    # runs: nominal (size, R_size) + conditions (R_power)
    conds = [("nominal", {}, sd["R_size"] if not quick else 8)] + [(name, over, sd["R_power"] if not quick else 4) for name, over in sd["conditions"].items()]
    rows = []; nom_p_res = []; nom_p_raw = []; store = {}
    for ci, (name, over, R) in enumerate(conds):
        ov = {}
        for key in ("faults", "nuisance"):
            if key in over:
                ov[key] = [dict(f, t_onset=t_on) for f in over[key]]
        args = [(_sim(cfg, sd["seed_base"] + 100 * ci + r, **ov), K_cal, K_post, L, N, sd["seed_base"] + 90000 + 100 * ci + r) for r in range(R)]
        outs = pmap(_rep_d, args, cfg["workers"])
        recs = []
        for r_i, o in enumerate(outs):
            # residual per leg from the equivariant DeLaN (parent, GPU), registered into blocks with the raw mask
            r_all = np.zeros((len(o["t"]), 16), dtype=np.float32)
            for li, leg in enumerate(LEGS):
                Lg = o["legs"][leg]; r_all[:, 4 * li:4 * li + 4] = Lg["y"] - quad.predict(leg, Lg["q"], Lg["dq"], Lg["ddq"], Lg["a"])
            dfr = pd.DataFrame(r_all, columns=[f"res_{l}_{j}" for l in LEGS for j in JOINTS]); dfr["t"] = o["t"]
            Zr, _ = register_blocks(dfr, [f"res_{l}_{j}" for l in LEGS for j in JOINTS], L_s=L, N=N, mask=o["mask"], max_blocks=K_cal + K_post)
            rep_raw = C2Rep(o["man"]); seed = sd["seed_base"] + 90000 + 100 * ci + r_i
            nw = min(Zr.shape[0], o["Z"].shape[0]) // window; p_res = np.empty(nw); p_raw = np.empty(nw)
            for w in range(nw):
                p_res[w], _ = hg_permutation_test(Zr[w * window:(w + 1) * window], res_rep, statistic="paired_energy", M=T["M"], rng=np.random.default_rng([seed, w, 1]))
                p_raw[w], _ = hg_permutation_test(o["Z"][w * window:(w + 1) * window], rep_raw, statistic="paired_energy", M=T["M"], rng=np.random.default_rng([seed, w, 2]))
            recs.append({"p_res": p_res, "p_raw": p_raw, "p_res_cal60": float(hg_permutation_test(Zr[:K_cal], res_rep, statistic="paired_energy", M=T["M"], rng=np.random.default_rng([seed, 3]))[0]),
                         "p_raw_cal60": float(hg_permutation_test(o["Z"][:K_cal], rep_raw, statistic="paired_energy", M=T["M"], rng=np.random.default_rng([seed, 4]))[0])})
        store[name] = recs
        if name == "nominal":
            nom_p_res += [r["p_res"][:K_cal // window] for r in recs]; nom_p_raw += [r["p_raw"][:K_cal // window] for r in recs]
        print(f"  [d] {name} done", flush=True)
    h_res = calibrate_ecusum_threshold(nom_p_res, K_post // window, far=alpha, n_boot=1000, rng=np.random.default_rng(3))
    h_raw = calibrate_ecusum_threshold(nom_p_raw, K_post // window, far=alpha, n_boot=1000, rng=np.random.default_rng(4))
    w0 = K_cal // window
    for name, recs in store.items():
        R = len(recs)
        for chan, h in (("res", h_res), ("raw", h_raw)):
            d_e, d_c = [], []
            for r in recs:
                E, al = eprocess_alarm(r[f"p_{chan}"], alpha, start=w0); d_e.append(np.nan if al is None else (al - w0 + 1) * window)
                S, al = ecusum(r[f"p_{chan}"], h, start=w0); d_c.append(np.nan if al is None else (al - w0 + 1) * window)
            d_e = np.array(d_e); d_c = np.array(d_c)
            p60 = np.array([r[f"p_{chan}_cal60"] for r in recs]); k = int(np.sum(p60 <= alpha)); band = nominal_band(alpha, R)
            rows.append({"condition": name, "channel": {"res": "R- on equivariant-DeLaN residual", "raw": "R- on raw element"}[chan], "R": R,
                         "size_cal60": k / R, "band_lo": band[0], "band_hi": band[1], "eproc_alarm_frac": float(np.mean(~np.isnan(d_e))), "eproc_delay_median": float(np.nanmedian(d_e)) if np.isfinite(d_e).any() else np.nan,
                         "ecusum_alarm_frac": float(np.mean(~np.isnan(d_c))), "ecusum_delay_median": float(np.nanmedian(d_c)) if np.isfinite(d_c).any() else np.nan, "ecusum_h": h})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e13d_rolling_residual.csv", index=False)
    rep_json = json.loads((mdir / "report.json").read_text())
    _conclude(res_dir, f"[e13d] equivariant DeLaN (M1 rolling, {mdir.name}): val RMSE per leg " + str({l: np.round(rep_json['legs'][l]['final_val_rmse_per_joint'], 3).tolist() for l in LEGS})
              + f", delta_f q95 {rep_json.get('defect_q95', float('nan')):.2e} | " + "; ".join(f"{r.condition} [{r.channel}]: size(60 cal blocks) {r.size_cal60:.3f} band [{r.band_lo:.3f},{r.band_hi:.3f}], e-proc alarm {r.eproc_alarm_frac:.2f} (delay {r.eproc_delay_median:.0f}), e-CUSUM alarm {r.ecusum_alarm_frac:.2f} (delay {r.ecusum_delay_median:.0f})" for r in tab.itertuples()))
    return tab


def residual_manifest_m1():
    """Residual data element of the wheeled M1: 16 rows res_<leg>_<joint> with the torque signs (wheel row sign +1)."""
    from geofdi.sim.telemetry_m1 import JOINT_SIGN, MIRROR_LEG
    ch = [{"name": f"res_{l}_{j}", "group": "res", "leg": l, "joint": j, "kind": "scalar-signed", "partner": f"res_{MIRROR_LEG[l]}_{j}", "sign": JOINT_SIGN[j], "in_Z": True} for l in LEGS for j in JOINTS]
    return {"schema": "geofdi-residual-element-m1-v1", "leg_order": list(LEGS), "joint_order": list(JOINTS), "gait_group": {"G": "C2", "delta_theta": 0.0}, "channels": ch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c", "d", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["a", "b", "c", "d"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d}[s](cfg, res_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E01W ALL DONE", flush=True)


if __name__ == "__main__":
    main()
