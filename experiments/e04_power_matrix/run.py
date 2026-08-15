#!/usr/bin/env python3
"""e04 — power matrix (a), nuisance false-alarm table (b), isotypic prediction (c), noise sweep wrapped-vs-unfolded
(d), isolation ranking with swing-phase conditioning (e). Sprint 2, S2.

    python experiments/e04_power_matrix/run.py --stage a|b|c|d|e|all [--run-id ID] [--quick]

Calibration discipline: every detector (R^- mirror monitor, R^+ tracking-error conformal monitor, e-CUSUM threshold)
is calibrated on NOMINAL cycles only — the K_cal pre-onset cycles of each replicate (and, for the e-CUSUM threshold,
the pooled pre-onset windows of the replicates). Fault cycles are only ever evaluated. Alarm rules are fixed a priori:
e-process at 1/alpha, e-CUSUM at h(alpha) from the nominal calibration. Figures -> results/e04_power_matrix/<run_id>/,
first replicates' cycles -> data/processed/sim/e04_power_matrix/<run_id>/.
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

from geofdi.detect.monitors import (
    JOINTS,
    LEGS,
    MirrorMonitor,
    calibrate_ecusum_threshold,
    channel_projection_energy,
    conformal_pvalues,
    ecusum,
    eprocess_alarm,
    leg_magnitude_deviation,
    rank_groups,
    tracking_scores,
    unfolded_permutation_tests,
)
from geofdi.detect.permutation import hg_permutation_tests, pooled_scale
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles, write_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry import z_channel_names

EXP_NAME = "e04_power_matrix"
REPO = Path(__file__).resolve().parents[2]
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]


# ------------------------------------------------------------------------------------ helpers
def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        if k == "controller":
            s["controller"] = {**s.get("controller", {}), **v}
        else:
            s[k] = v
    return s


def _cycles(sim_cfg, n_cycles, N, drop_first):
    cfg = SimConfig(**sim_cfg)
    period = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (n_cycles + drop_first + 2) * period
    df, man = rollout(cfg)
    chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first)
    Zq, _ = register_cycles(df, QREF, N=N, drop_first=drop_first)
    K = min(n_cycles, Z.shape[0], Zq.shape[0])
    return Z[:K], Zq[:K], man, chans


def _onset_time(K_cal, drop_first, period=0.5):
    return (K_cal + drop_first) * period


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


def _detectors(Z, Zq, man, chans, K_cal, alpha, M, window, seed):
    """Run all monitors on one replicate; returns per-detector alarm delays (cycles after onset; None) + nominal p's."""
    rep = C2Rep(man)
    out = {}
    nom_p = {}
    for stat in ("paired_energy", "energy_distance"):
        mm = MirrorMonitor(rep, window=window, M=M, statistic=stat, alpha=alpha)
        ps = mm.window_pvalues(Z, seed=seed)
        w0 = K_cal // window
        nom_p[stat] = ps[:w0]
        E, al = eprocess_alarm(ps, alpha, start=w0)
        out[f"Rminus_{stat}_eproc"] = None if al is None else (al - w0 + 1) * window
        out[f"_p_{stat}"] = ps
    sc = tracking_scores(Z, chans, Zq)
    pc = conformal_pvalues(sc[:K_cal], sc[K_cal:])
    E, al = eprocess_alarm(pc, alpha, start=0)
    out["Rplus_track_eproc"] = None if al is None else al + 1
    out["_p_track"] = pc
    out["_nom_score"] = sc[:K_cal]
    out["_nom_p"] = nom_p
    return out


# ------------------------------------------------------------------------------------ workers
def _rep_fault(sim_cfg, K_cal, K_post, N, drop_first, alpha, M, window, seed, save_dir):
    Z, Zq, man, chans = _cycles(sim_cfg, K_cal + K_post, N, drop_first)
    if save_dir is not None:
        write_cycles(save_dir, Z.astype(np.float32), {"N": N, "channels": chans, "n_cycles": int(Z.shape[0]),
                                                      "t_start": [0.0] * int(Z.shape[0]), "row_start": [0] * int(Z.shape[0])}, man)
    d = _detectors(Z, Zq, man, chans, K_cal, alpha, M, window, seed)
    d["K"] = int(Z.shape[0])
    return d


def _rep_nuisance(sim_cfg, K_cal, K_mon, N, drop_first, alpha, M, window, seed, merged_with=None):
    Z, Zq, man, chans = _cycles(sim_cfg, K_cal + K_mon, N, drop_first)
    block = 1
    if merged_with is not None:                       # out-and-back: interleave with the mirrored-slope run -> pairs
        Z2, Zq2, _, _ = _cycles(merged_with, K_cal + K_mon, N, drop_first)
        K = min(Z.shape[0], Z2.shape[0]); Zi = np.empty((2 * K,) + Z.shape[1:]); Zqi = np.empty((2 * K,) + Zq.shape[1:])
        Zi[0::2] = Z[:K]; Zi[1::2] = Z2[:K]; Zqi[0::2] = Zq[:K]; Zqi[1::2] = Zq2[:K]
        Z, Zq = Zi, Zqi; K_cal = 2 * K_cal; block = 2
    rep = C2Rep(man); out = {}
    for stat in ("paired_energy", "energy_distance"):
        for b in sorted({1, block}):
            mm = MirrorMonitor(rep, window=window, M=M, statistic=stat, alpha=alpha, block_len=b)
            ps = mm.window_pvalues(Z, seed=seed + b)
            w0 = K_cal // window
            E, al = eprocess_alarm(ps, alpha, start=w0)
            out[f"Rminus_{stat}_b{b}"] = {"p_mon": ps[w0:].tolist(), "alarm": al is not None}
    sc = tracking_scores(Z, chans, Zq)
    pc = conformal_pvalues(sc[:K_cal], sc[K_cal:])
    E, al = eprocess_alarm(pc, alpha, start=0)
    out["Rplus_track"] = {"p_mon": pc.tolist(), "alarm": al is not None}
    return out


def _rep_iso(sim_cfg, K_cal, K_post, N, drop_first, alpha, M, window, seed, nominal_sim_cfg):
    Z, Zq, man, chans = _cycles(sim_cfg, K_cal + K_post, N, drop_first)
    d = _detectors(Z, Zq, man, chans, K_cal, alpha, M, window, seed)
    rep = C2Rep(man)
    # projection energies of the mean post-onset deviation from the nominal (pre-onset) mean, standardized
    Zs_all = rep.apply("s", Z); sc = pooled_scale(Z[:K_cal], Zs_all[:K_cal])
    delta = ((Z[K_cal:] - Z[:K_cal].mean(0)) / sc).mean(0)                     # (d,N)
    dm = rep.mirror_only(np.roll(delta, -delta.shape[-1] // 2, axis=-1)) if False else rep.apply("s", delta[None])[0]
    anti = 0.5 * (delta - dm); sym = 0.5 * (delta + dm)
    d["E_anti"] = float((anti ** 2).sum()); d["E_sym"] = float((sym ** 2).sum())
    d["post_window_p"] = {s: d[f"_p_{s}"][K_cal // window:].tolist() for s in ("paired_energy", "energy_distance")}
    for s in ("paired_energy", "energy_distance"):
        d.pop(f"_p_{s}")
    d.pop("_p_track"); d.pop("_nom_score"); d.pop("_nom_p")
    return d


def _rep_noise(sim_cfg, K, N, drop_first, M, seed):
    Z, Zq, man, chans = _cycles(sim_cfg, K, N, drop_first)
    rep = C2Rep(man)
    w = hg_permutation_tests(Z, rep, M=M, rng=np.random.default_rng([seed, 1]))
    u = unfolded_permutation_tests(Z, rep, M=M, rng=np.random.default_rng([seed, 2]))
    return {"wrap": {s: w[s]["p"] for s in w}, "unfolded": {s: u[s]["p"] for s in u}, "n_unf": u["paired_energy"]["n_elements"]}


def _rep_isolation(sim_cfg, K_cal, K_post, N, drop_first, seed):
    Z, Zq, man, chans = _cycles(sim_cfg, K_cal + K_post, N, drop_first)
    rep = C2Rep(man)
    cal, post = Z[:K_cal], Z[K_cal:]
    out = {}
    for kind, sw in (("all", False), ("swing", True)):
        r_nom = channel_projection_energy(post, rep, chans, swing_condition=sw, Z_cal=cal)
        r_pool = channel_projection_energy(post, rep, chans, swing_condition=sw, Z_cal=None)
        out[f"pair_{kind}_nominal"] = [(f"{p}-{j}", e) for (p, j), e in rank_groups(r_nom["per_pair"])]
        out[f"pair_{kind}_pooled"] = [(f"{p}-{j}", e) for (p, j), e in rank_groups(r_pool["per_pair"])]
        out[f"leg_{kind}_nominal"] = [(f"{l}-{j}", e) for (l, j), e in rank_groups(r_nom["per_group"])]
    sc = tracking_scores(Z, chans, Zq, per_leg=True)
    dev = leg_magnitude_deviation(sc[:K_cal], sc[K_cal:])
    out["leg_dev"] = {leg: float(dev[i]) for i, leg in enumerate(LEGS)}
    return out


# ------------------------------------------------------------------------------------ stages
def stage_a(cfg, res_dir, data_dir, quick=False):
    sa = cfg["stage_a"]; R = 6 if quick else sa["R"]; det = cfg["detect"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    K_cal, K_post = sa["K_cal"], sa["K_post"]; alpha, M, window = det["alpha"], det["M"], det["window_rminus"]
    t_on = _onset_time(K_cal, df0)
    configs = []
    for ftype, mags in sa["faults"].items():
        mags = mags[:2] if quick else mags
        for mag in mags:
            for joint in sa["joints"]:
                magnitude = {"actuator_gain": mag - 1.0, "friction_scale": mag - 1.0}.get(ftype, mag)
                configs.append((ftype, mag, joint, magnitude))
    rows, curves = [], {}
    all_nom_p = {"paired_energy": [], "energy_distance": []}
    results = {}
    for ci, (ftype, mag, joint, magnitude) in enumerate(configs):
        fault = dict(type=ftype, t_onset=t_on, leg=sa["leg"], joint=joint, magnitude=float(magnitude))
        args = [(_sim(cfg, sa["seed_base"] + r, faults=[fault]), K_cal, K_post, N, df0, alpha, M, window, sa["seed_base"] + 70000 + r,
                 (data_dir / "a" / f"{ftype}_{mag}_{joint}" / f"rep_{r:03d}") if r < cfg["outputs"]["store_cycles_first_n"] and ci < 2 else None) for r in range(R)]
        res = pmap(_rep_fault, args, cfg["workers"])
        results[(ftype, mag, joint)] = res
        for r in res:
            for s in all_nom_p:
                all_nom_p[s].append(r["_nom_p"][s])
        print(f"  [a] {ftype} {mag} {joint} done", flush=True)
    # e-CUSUM thresholds from pooled nominal (pre-onset) windows, horizon = 100 cycles
    h = {s: calibrate_ecusum_threshold(all_nom_p[s], det["ecusum_far_horizon_windows"], far=alpha, n_boot=1000, rng=np.random.default_rng(1)) for s in all_nom_p}
    for (ftype, mag, joint), res in results.items():
        w0 = K_cal // window
        for r in res:
            for s in all_nom_p:
                S, al = ecusum(r[f"_p_{s}"], h[s], start=w0)
                r[f"Rminus_{s}_ecusum"] = None if al is None else (al - w0 + 1) * window
        dets = [k for k in res[0] if not k.startswith("_") and k != "K"]
        for dname in dets:
            delays = np.array([r[dname] if r[dname] is not None else np.nan for r in res], dtype=float)
            det_rate = float(np.mean(~np.isnan(delays) & (delays <= K_post)))
            det20 = float(np.mean(~np.isnan(delays) & (delays <= 20)))
            med = float(np.nanmedian(delays)) if np.isfinite(delays).any() else np.nan
            q90 = float(np.nanquantile(delays, 0.9)) if np.isfinite(delays).any() else np.nan
            rows.append({"fault": ftype, "magnitude": mag, "joint": joint, "detector": dname, "R": R, "det_rate_100": det_rate,
                         "det_rate_20": det20, "delay_median": med, "delay_q90": q90})
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e04a_power_grid.csv", index=False)
    pd.DataFrame([{"stat": s, "ecusum_h": v} for s, v in h.items()]).to_csv(res_dir / "e04a_ecusum_thresholds.csv", index=False)
    # figures: delay vs magnitude per fault type (main detectors)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    main_dets = ["Rminus_paired_energy_eproc", "Rminus_paired_energy_ecusum", "Rplus_track_eproc"]
    ftypes = list(sa["faults"].keys())
    fig, axes = plt.subplots(1, len(ftypes), figsize=(4.0 * len(ftypes), 3.8), squeeze=False)
    for ax, ftype in zip(axes[0], ftypes):
        for dname in main_dets:
            for joint, ls in zip(sa["joints"], ("-", "--")):
                sub = tab[(tab.fault == ftype) & (tab.detector == dname) & (tab.joint == joint)].sort_values("magnitude")
                if len(sub):
                    ax.plot(sub.magnitude, sub.delay_median, ls, marker="o", label=f"{dname} {joint}")
        ax.set_title(f"{ftype}", fontsize=9); ax.set_xlabel("magnitude"); ax.set_ylabel("median delay (cycles)"); ax.set_ylim(0, K_post + 5)
        if ftype == "actuator_gain":
            ax.invert_xaxis()
        ax.legend(fontsize=6)
    fig.suptitle("e04a — detection delay vs fault magnitude (onset at cycle 60; e-process 1/α, e-CUSUM at h(α=0.05))", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e04a_delay_vs_magnitude.png", dpi=130); plt.close(fig)
    piv = tab[tab.detector.isin(main_dets)].pivot_table(index=["fault", "magnitude", "joint"], columns="detector", values="det_rate_100")
    piv.to_csv(res_dir / "e04a_detection_rate_heat.csv")
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(piv) + 1.5)); im = ax.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{a} {b} {c}" for a, b, c in piv.index], fontsize=7); ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7, rotation=20)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if piv.iloc[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=ax, label="detection rate within 100 cycles"); ax.set_title("e04a — detection rate heat table", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e04a_detection_rate_heat.png", dpi=130); plt.close(fig)
    # gate: monotone (non-increasing median delay with magnitude, main R^- e-process) and kappa<=0.7 -> det20 >= 0.9
    mono_ok = True
    for ftype in ftypes:
        for joint in sa["joints"]:
            sub = tab[(tab.fault == ftype) & (tab.detector == "Rminus_paired_energy_ecusum") & (tab.joint == joint)].sort_values("magnitude")
            d = sub.delay_median.to_numpy()
            if ftype == "actuator_gain":
                d = d[::-1]                        # magnitude ordering: kappa 0.9 (mild) ... 0.5 (severe) -> increasing severity is decreasing kappa
            d = np.nan_to_num(d, nan=K_post + 1)
            if np.any(np.diff(d) > 2.0):
                mono_ok = False
    # gate detector = the FAR-calibrated R- e-CUSUM (the plain e-process has a structural floor of ~5 windows = 25
    # cycles with 5-cycle windows: only 16 sign-distinct flip patterns -> p >= 1/16 -> e <= 2 per window)
    k07 = tab[(tab.fault == "actuator_gain") & (tab.magnitude <= 0.7) & (tab.detector == "Rminus_paired_energy_ecusum")]
    k_ok = bool((k07.det_rate_20 >= 0.9).all()) if len(k07) else False
    k07e = tab[(tab.fault == "actuator_gain") & (tab.magnitude <= 0.7) & (tab.detector == "Rminus_paired_energy_eproc")]
    _conclude(res_dir, f"[e04a] {'PASS' if (mono_ok and k_ok) else 'FAIL'}: monotone delay decrease with magnitude (tol 2 cycles): {mono_ok}; "
              f"kappa<=0.7 detected within 20 cycles >=0.9 (R- paired e-CUSUM at FAR 0.05): {k_ok} ({k07.det_rate_20.round(2).tolist()}; plain e-process {k07e.det_rate_20.round(2).tolist()}, floor 25 cycles); h_ecusum={ {k: round(v,3) for k,v in h.items()} }")
    return {"pass": mono_ok and k_ok}


def stage_b(cfg, res_dir, data_dir, quick=False):
    sb = cfg["stage_b"]; R = 6 if quick else sb["R"]; det = cfg["detect"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    K_cal, K_mon = sb["K_cal"], (60 if quick else sb["K_mon"]); alpha, M, window = det["alpha"], det["M"], det["window_rminus_far"]
    conds = [("nominal", {}, None)]
    for m in sb["nuisances"]["payload_symmetric"]:
        conds.append((f"payload_sym_{m}kg", dict(nuisance=[dict(type="payload_symmetric", magnitude=m)]), None))
    for m in sb["nuisances"]["payload_asymmetric"]:
        conds.append((f"payload_asym_{m}kg", dict(nuisance=[dict(type="payload_asymmetric", magnitude=m, params={"offset_y": 0.05})]), None))
    for m in sb["nuisances"]["drift_symmetric"]:
        conds.append((f"drift_sym_{m}", dict(nuisance=[dict(type="drift_symmetric", magnitude=m, params={"tau_s": 20.0})]), None))
    for v in sb["nuisances"]["speed"]:
        conds.append((f"speed_{v}", dict(speed=v), None))
    sd = sb["nuisances"]["slope_deg"]
    conds.append((f"slope_+{sd}_single", dict(terrain="slope", slope_deg=sd, slope_axis="lateral"), None))
    conds.append((f"slope_-{sd}_single", dict(terrain="slope", slope_deg=-sd, slope_axis="lateral"), None))
    conds.append((f"slope_pm{sd}_merged", dict(terrain="slope", slope_deg=sd, slope_axis="lateral"), dict(terrain="slope", slope_deg=-sd, slope_axis="lateral")))
    if quick:
        conds = [c for c in conds if c[0] in ("nominal", "payload_asym_1.0kg", "slope_+5.0_single", "slope_pm5.0_merged")]
    rows = []
    for name, over, merged in conds:
        args = [(_sim(cfg, sb["seed_base"] + r, **over), K_cal, K_mon, N, df0, alpha, M, window, sb["seed_base"] + 80000 + r,
                 (_sim(cfg, sb["seed_base"] + 40000 + r, **merged) if merged else None)) for r in range(R)]
        res = pmap(_rep_nuisance, args, cfg["workers"])
        for chan in res[0].keys():
            pm = np.concatenate([np.asarray(r[chan]["p_mon"]) for r in res]); n = len(pm)
            rate = float(np.mean(pm <= alpha)); k = int(np.sum(pm <= alpha)); ci = binom_ci(k, n); band = nominal_band(alpha, n)
            alarm = float(np.mean([r[chan]["alarm"] for r in res])); ka = int(sum(r[chan]["alarm"] for r in res)); cia = binom_ci(ka, R)
            rows.append({"nuisance": name, "channel": chan, "R": R, "n_tests": n, "far_per_test": rate, "far_ci_lo": ci[0], "far_ci_hi": ci[1],
                         "band_lo": band[0], "band_hi": band[1], "in_band": bool(band[0] <= rate <= band[1]),
                         "alarm_fraction": alarm, "alarm_ci_lo": cia[0], "alarm_ci_hi": cia[1], "alarm_le_alpha": bool(alarm <= alpha + 1e-12)})
        print(f"  [b] {name} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e04b_nuisance_far.csv", index=False)
    piv = tab.pivot_table(index="nuisance", columns="channel", values="far_per_test"); piv.to_csv(res_dir / "e04b_far_pivot.csv")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    chans = ["Rminus_paired_energy_b1", "Rminus_energy_distance_b1", "Rplus_track"]
    fig, ax = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(piv)); w = 0.25
    for i, ch in enumerate(chans):
        if ch in piv.columns:
            ax.bar(x + i * w, piv[ch].to_numpy(), width=w, label=ch)
    if "Rminus_paired_energy_b2" in piv.columns:
        m = piv["Rminus_paired_energy_b2"].to_numpy(); ax.scatter(x[~np.isnan(m)] + w, m[~np.isnan(m)], marker="*", s=90, color="k", zorder=5, label="R- paired, paired-direction blocks (b2)")
    ax.axhline(alpha, color="k", ls="--", lw=1); ax.set_xticks(x + w); ax.set_xticklabels(piv.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("false-alarm rate per test (α=0.05)"); ax.legend(fontsize=7); ax.set_title("e04b — nuisance × channel false-alarm table (nominal cycles, R^-: 20-cycle windows; R^+: per cycle)", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e04b_nuisance_far.png", dpi=130); plt.close(fig)
    def rate(n, ch):
        s = tab[(tab.nuisance == n) & (tab.channel == ch)]; return None if s.empty else (float(s.far_per_test.iloc[0]), bool(s.in_band.iloc[0]))
    sym_names = [n for n in piv.index if n.startswith(("payload_sym", "drift_sym", "speed", "nominal"))]
    sym_ok = all(rate(n, ch)[1] for n in sym_names for ch in ("Rminus_paired_energy_b1", "Rplus_track") if rate(n, ch))
    single_infl = any((rate(n, "Rminus_paired_energy_b1") or (0, True))[0] > tab[(tab.nuisance == n) & (tab.channel == "Rminus_paired_energy_b1")].band_hi.iloc[0] for n in piv.index if "single" in n)
    merged = rate("slope_pm5.0_merged", "Rminus_paired_energy_b2")
    merged_ok = bool(merged and merged[1])
    asym = [rate(n, "Rminus_paired_energy_b1") for n in piv.index if n.startswith("payload_asym")]
    _conclude(res_dir, f"[e04b] {'PASS' if (sym_ok and single_infl and merged_ok) else 'FAIL'}: symmetric nuisances in band on both channels: {sym_ok}; "
              f"single-direction slope inflates R-: {single_infl}; merged out-and-back (paired blocks) back in band: {merged_ok} ({merged}); "
              f"asymmetric payload R- rates (reported honestly): {asym}")
    return {"pass": bool(sym_ok and single_infl and merged_ok)}


def stage_c(cfg, res_dir, data_dir, quick=False):
    sc_ = cfg["stage_c"]; R = 6 if quick else sc_["R"]; det = cfg["detect"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    K_cal, K_post = sc_["K_cal"], sc_["K_post"]; alpha, M, window = det["alpha"], det["M"], det["window_rminus"]
    t_on = _onset_time(K_cal, df0)
    rows, energies = [], []
    for gname, spec in sc_["groups"].items():
        faults = [dict(type="actuator_gain", t_onset=t_on, leg=s["leg"], joint=s["joint"], magnitude=s["kappa"] - 1.0) for s in spec]
        args = [(_sim(cfg, sc_["seed_base"] + r, faults=faults), K_cal, K_post, N, df0, alpha, M, window, sc_["seed_base"] + 90000 + r, None) for r in range(R)]
        res = pmap(_rep_iso, args, cfg["workers"])
        for dname in ("Rminus_paired_energy_eproc", "Rminus_energy_distance_eproc", "Rplus_track_eproc"):
            delays = np.array([r[dname] if r[dname] is not None else np.nan for r in res], dtype=float)
            power = float(np.mean(~np.isnan(delays) & (delays <= K_post))); k = int(np.sum(~np.isnan(delays) & (delays <= K_post))); ci = binom_ci(k, R)
            rows.append({"group": gname, "detector": dname, "power_100": power, "ci_lo": ci[0], "ci_hi": ci[1], "delay_median": float(np.nanmedian(delays)) if np.isfinite(delays).any() else np.nan, "R": R})
        pw = np.concatenate([np.asarray(r["post_window_p"]["paired_energy"]) for r in res]); n = len(pw)
        rate = float(np.mean(pw <= alpha)); band = nominal_band(alpha, n)
        rows.append({"group": gname, "detector": "Rminus_paired_energy_window_rejection", "power_100": rate, "ci_lo": binom_ci(int(np.sum(pw <= alpha)), n)[0],
                     "ci_hi": binom_ci(int(np.sum(pw <= alpha)), n)[1], "delay_median": np.nan, "R": R, "band_lo": band[0], "band_hi": band[1]})
        Ea = np.array([r["E_anti"] for r in res]); Es = np.array([r["E_sym"] for r in res])
        energies.append({"group": gname, "E_anti_mean": float(Ea.mean()), "E_sym_mean": float(Es.mean()), "anti_share": float(np.mean(Ea / (Ea + Es + 1e-12)))})
        print(f"  [c] {gname} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e04c_isotypic_power.csv", index=False)
    en = pd.DataFrame(energies); en.to_csv(res_dir / "e04c_projection_energy.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = list(sc_["groups"].keys()); x = np.arange(len(groups)); w = 0.38
    for i, dname in enumerate(("Rminus_paired_energy_eproc", "Rplus_track_eproc")):
        v = [tab[(tab.group == g) & (tab.detector == dname)].power_100.iloc[0] for g in groups]
        lo = [tab[(tab.group == g) & (tab.detector == dname)].ci_lo.iloc[0] for g in groups]; hi = [tab[(tab.group == g) & (tab.detector == dname)].ci_hi.iloc[0] for g in groups]
        axes[0].bar(x + i * w, v, width=w, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=3, label=dname)
    axes[0].axhline(alpha, color="k", ls="--", lw=1); axes[0].set_xticks(x + w / 2); axes[0].set_xticklabels(groups); axes[0].set_ylabel("power (alarm within 100 cycles)"); axes[0].legend(fontsize=7); axes[0].set_title("e04c — power per channel", fontsize=9)
    axes[1].bar(x, en.anti_share, color="C3", label="antisymmetric share ‖Δ⁻‖²/(‖Δ⁻‖²+‖Δ⁺‖²)"); axes[1].set_xticks(x); axes[1].set_xticklabels(groups); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7); axes[1].set_title("e04c — projection energy of the fault signature", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e04c_isotypic.png", dpi=130); plt.close(fig)
    def pw_(g, d): return float(tab[(tab.group == g) & (tab.detector == d)].power_100.iloc[0])
    p1 = pw_("single", "Rminus_paired_energy_eproc") >= 0.9 and pw_("single", "Rplus_track_eproc") >= 0.9
    be = tab[(tab.group == "bilateral_equal") & (tab.detector == "Rminus_paired_energy_window_rejection")].iloc[0]
    p2 = bool(be.band_lo <= be.power_100 <= be.band_hi) and pw_("bilateral_equal", "Rminus_paired_energy_eproc") <= alpha + 1e-9 and pw_("bilateral_equal", "Rplus_track_eproc") >= 0.9
    # P3 with a graded R- strength (alarm fractions saturate at 1 for strong faults): post-onset window rejection rate,
    # and the median delay ordering single < unequal (pre-registered direction; only the metric's resolution differs)
    def wr(g): return float(tab[(tab.group == g) & (tab.detector == "Rminus_paired_energy_window_rejection")].power_100.iloc[0])
    def dl(g): return float(tab[(tab.group == g) & (tab.detector == "Rminus_paired_energy_eproc")].delay_median.iloc[0])
    p3 = (wr("bilateral_equal") < wr("bilateral_unequal") < wr("single")) and (np.isnan(dl("bilateral_equal")) or dl("single") <= dl("bilateral_unequal") <= dl("bilateral_equal")) \
         and pw_("bilateral_unequal", "Rplus_track_eproc") >= 0.9
    _conclude(res_dir, f"[e04c] {'PASS' if (p1 and p2 and p3) else 'FAIL'}: pre-registered predictions: single R-/R+ detect: {p1}; bilateral-equal R- blind (window rejection {be.power_100:.3f} in band [{be.band_lo:.3f},{be.band_hi:.3f}], alarm {pw_('bilateral_equal','Rminus_paired_energy_eproc'):.2f}) & R+ detects: {p2}; unequal intermediate: {p3} "
              f"| R- alarm powers: single {pw_('single','Rminus_paired_energy_eproc'):.2f}, equal {pw_('bilateral_equal','Rminus_paired_energy_eproc'):.2f}, unequal {pw_('bilateral_unequal','Rminus_paired_energy_eproc'):.2f}; R- window rejection: {wr('single'):.3f}/{wr('bilateral_equal'):.3f}/{wr('bilateral_unequal'):.3f}; R- median delay: {dl('single')}/{dl('bilateral_equal')}/{dl('bilateral_unequal')} cycles; R+: {pw_('single','Rplus_track_eproc'):.2f}/{pw_('bilateral_equal','Rplus_track_eproc'):.2f}/{pw_('bilateral_unequal','Rplus_track_eproc'):.2f} | anti share: {en.anti_share.round(3).tolist()}")
    return {"pass": bool(p1 and p2 and p3)}


def stage_d(cfg, res_dir, data_dir, quick=False):
    sd = cfg["stage_d"]; R = 20 if quick else sd["R"]; det = cfg["detect"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    K = sd["K"]; alpha, M = det["alpha"], det["M"]
    rows = []
    for a_std in sd["actuator_std"]:
        args = [(_sim(cfg, sd["seed_base"] + r, noise={**cfg["sim"]["noise"], "actuator_std": a_std}), K, N, df0, M, sd["seed_base"] + 60000 + r) for r in range(R)]
        res = pmap(_rep_noise, args, cfg["workers"])
        for kind in ("wrap", "unfolded"):
            for s in ("paired_energy", "energy_distance"):
                p = np.array([r[kind][s] for r in res]); k = int(np.sum(p <= alpha)); ci = binom_ci(k, R); band = nominal_band(alpha, R)
                rows.append({"actuator_std": a_std, "test": kind, "stat": s, "size": k / R, "ci_lo": ci[0], "ci_hi": ci[1], "band_lo": band[0], "band_hi": band[1],
                             "in_band": bool(band[0] <= k / R <= band[1]), "R": R, "n_elements": res[0]["n_unf"] if kind == "unfolded" else K})
        print(f"  [d] actuator_std {a_std} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e04d_noise_sweep.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for kind, ls in (("wrap", "-"), ("unfolded", "--")):
        for s, c in (("paired_energy", "C0"), ("energy_distance", "C1")):
            sub = tab[(tab.test == kind) & (tab.stat == s)]
            ax.errorbar(sub.actuator_std, sub["size"], yerr=[sub["size"] - sub.ci_lo, sub.ci_hi - sub["size"]], ls=ls, marker="o", color=c, capsize=3, label=f"{kind} {s}")
    b = tab.iloc[0]; ax.axhspan(b.band_lo, b.band_hi, color="0.9", label="binomial 95% band"); ax.axhline(alpha, color="k", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("actuator process noise std [N m]"); ax.set_ylabel(f"empirical size (α={alpha}, R={R}, K={K})"); ax.legend(fontsize=7)
    ax.set_title("e04d — wrapped (within-cycle) vs unfolded (true-shift) mirror test under H0", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e04d_noise_sweep.png", dpi=130); plt.close(fig)
    unf_ok = bool(tab[tab.test == "unfolded"].in_band.all())
    wrap_ok_levels = [a for a in sd["actuator_std"] if tab[(tab.test == "wrap") & (tab.actuator_std == a)].in_band.all()]
    _conclude(res_dir, f"[e04d] {'PASS' if unf_ok else 'FAIL'}: unfolded test in band at all noise levels: {unf_ok}; wrapped test in band at actuator_std in {wrap_ok_levels} (boundary for the wrapped version)")
    return {"pass": unf_ok, "wrap_ok_levels": wrap_ok_levels}


def stage_e(cfg, res_dir, data_dir, quick=False):
    se = cfg["stage_e"]; R = 6 if quick else se["R"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    K_cal, K_post = se["K_cal"], se["K_post"]; t_on = _onset_time(K_cal, df0)
    fault = dict(type="actuator_gain", t_onset=t_on, leg=se["fault"]["leg"], joint=se["fault"]["joint"], magnitude=se["fault"]["kappa"] - 1.0)
    args = [(_sim(cfg, se["seed_base"] + r, faults=[fault]), K_cal, K_post, N, df0, se["seed_base"] + 50000 + r) for r in range(R)]
    res = pmap(_rep_isolation, args, cfg["workers"])
    pair_of = {"LF": "F", "RF": "F", "LH": "H", "RH": "H"}
    target_leg, target_joint = se["fault"]["leg"], se["fault"]["joint"]; target_pair = f"{pair_of[target_leg]}-{target_joint}"; target = f"{target_leg}-{target_joint}"
    pairs = [f"{p}-{j}" for p in ("F", "H") for j in JOINTS]
    rows = []
    for kind in ("pair_all_nominal", "pair_swing_nominal", "pair_all_pooled", "pair_swing_pooled"):
        agg = {g: 0.0 for g in pairs}; top1 = {g: 0 for g in pairs}; ranks = []
        for r in res:
            for pos, (g, e) in enumerate(r[kind]):
                agg[g] += e / R
                if pos == 0: top1[g] += 1
                if g == target_pair: ranks.append(pos + 1)
        for pos, (g, e) in enumerate(sorted(agg.items(), key=lambda kv: -kv[1])):
            rows.append({"ranking": kind, "rank": pos + 1, "group": g, "mean_energy": e, "top1_count": top1[g], "is_target": g == target_pair, "target_rank_median": float(np.median(ranks))})
    # left/right resolution by the R+ per-leg magnitude deviation within the top pair
    resolved = []
    for r in res:
        top_pair = r["pair_swing_nominal"][0][0]; pj = top_pair.split("-")
        legs = ("LF", "RF") if pj[0] == "F" else ("LH", "RH")
        leg = max(legs, key=lambda l: r["leg_dev"][l]); resolved.append(f"{leg}-{pj[1]}")
    conf = pd.Series(resolved).value_counts()
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e04e_isolation_ranking.csv", index=False)
    conf.rename_axis("isolated (R- pair-joint, swing, nominal scale + R+ left/right)").to_csv(res_dir / "e04e_confusion_resolved.csv", header=["count"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, kind in zip(axes, ("pair_all_pooled", "pair_swing_pooled", "pair_all_nominal", "pair_swing_nominal")):
        sub = tab[tab.ranking == kind].sort_values("rank")
        ax.barh(sub.group, sub.mean_energy, color=["C3" if t else "C0" for t in sub.is_target]); ax.invert_yaxis(); ax.set_xlabel("mean R⁻ projection energy")
        ax.set_title(f"{kind}\ntop-1 hits {sub[sub.is_target].top1_count.iloc[0]}/{R}", fontsize=8)
    fig.suptitle(f"e04e — isolation ranking, LF-KFE κ=0.5 (R={R}); left/right resolved by R⁺: {dict(conf)}", fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / "e04e_isolation_ranking.png", dpi=130); plt.close(fig)
    top_sw = tab[(tab.ranking == "pair_swing_nominal") & (tab["rank"] == 1)].group.iloc[0]; top_all = tab[(tab.ranking == "pair_all_nominal") & (tab["rank"] == 1)].group.iloc[0]
    top_pool_sw = tab[(tab.ranking == "pair_swing_pooled") & (tab["rank"] == 1)].group.iloc[0]
    ok = (top_sw == target_pair) and (conf.idxmax() == target)
    _conclude(res_dir, f"[e04e] {'PASS' if ok else 'FAIL'}: nominal-scale R- pair-joint rank-1: all-phase {top_all}, swing {top_sw} (target pair {target_pair}); pooled-scale swing rank-1: {top_pool_sw}; "
              f"resolved (pair,joint)+R+ left/right: {dict(conf)} (target {target})")
    return {"pass": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["a", "b", "c", "d", "e", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO / "data" / "processed" / "sim" / EXP_NAME / args.run_id; data_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["a", "b", "c", "d", "e"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick}", flush=True)
    fn = {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d, "e": stage_e}
    for s in stages:
        t0 = _dt.datetime.now(); fn[s](cfg, res_dir, data_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)


if __name__ == "__main__":
    main()
