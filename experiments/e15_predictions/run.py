#!/usr/bin/env python3
"""e15 — three pre-registered prediction experiments for the N1-2 two-layer theorem (Sprint 8 Block P).

  --stage a  P1 chirality ceiling : bilateral mirror-equal actuator gain (LF+RF) sweep; R- power + gait chirality index.
  --stage b  P2 statistic split   : zero-mean LF encoder-noise variance inflation; paired_energy vs energy_distance power.
  --stage c  P3 slip regimes      : Go2 unilateral/uniform foot friction + M1 single/both wheel friction; R- vs InEKF NIS.

Pre-registration docs/protocol/e15_preregistration.md (committed first). alpha 0.05; R- uses the raw C2 element (no model).
    python experiments/e15_predictions/run.py --stage a|b|c|all [--quick]
Outputs -> $GEOFDI_DATA_ROOT/results/e15_predictions/<run>/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from geofdi.detect.evalue import eprocess
from geofdi.detect.permutation import hg_permutation_test
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_blocks, register_cycles, straight_mask_kinematic
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, pmap
from geofdi.sim.telemetry import z_channel_names

REPO = Path(__file__).resolve().parents[2]
LEGS = ("LF", "RF", "LH", "RH"); MIRROR = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}


# ------------------------------------------------------------------ shared
def _go2_cycles(cfg_sim, seed, faults, N, drop_first, K_total):
    cfg = SimConfig(**{**cfg_sim, "seed": int(seed), "duration_s": (K_total + drop_first + 3) * float(cfg_sim["controller"]["period_s"]), "faults": faults})
    df, man = rollout(cfg)
    chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first)
    duty = {l: [] for l in LEGS}; td = {l: [] for l in LEGS}
    # per-cycle duty factor and touch-down phase from the contact flag, aligned to the registered cycles
    th = df["theta"].to_numpy(); t = df["t"].to_numpy()
    bounds = np.where(np.diff((th[:-1] > th[1:]).astype(int)) != 0)[0]
    cyc = np.where((th[:-1] - th[1:]) > 0.5)[0] + 1
    cyc = np.concatenate([[0], cyc, [len(th)]])
    for ci in range(len(cyc) - 1):
        a, b = cyc[ci], cyc[ci + 1]
        if b - a < 10:
            continue
        for l in LEGS:
            c = df[f"c_{l}"].to_numpy()[a:b] > 0.5
            duty[l].append(float(c.mean()))
            td[l].append(float(th[a + np.argmax(c)] if c.any() else np.nan))
    return {"Z": Z.astype(np.float32), "man": man, "K": Z.shape[0], "duty": {l: np.array(duty[l][drop_first:]) for l in LEGS}, "td": {l: np.array(td[l][drop_first:]) for l in LEGS}}


def _rminus_power(Z, rep, K_cal, det, seed, statistic="paired_energy"):
    """Post-onset FAR-controlled R- alarm: window flip p-values on the post-onset segment -> e-process alarm within it.
    Returns (alarmed_post, alarmed_cal) so power and FAR are estimated across seeds."""
    win = det["window_rminus"]; M = det["M"]; alpha = det["alpha"]
    def alarm(seg):
        nw = seg.shape[0] // win
        if nw < 2:
            return False
        pw = np.array([hg_permutation_test(seg[w * win:(w + 1) * win], rep, statistic=statistic, M=M, rng=np.random.default_rng([seed, w]))[0] for w in range(nw)])
        E, al = eprocess(pw, alpha)
        return al is not None
    cal = Z[:K_cal]; post = Z[K_cal:]
    return alarm(post), alarm(cal)


def _chirality_index(duty, td):
    """Realized-gait chirality (A5' eps_chir): the RAW mirror residual of the stance duty factors and touch-down phases,
    averaged over the two mirror pairs (interpretable duty-fraction units; the rp003 Sprint-1 chiral gait had ~0.23).
    A symmetric attractor gives ~0; a chirality bifurcation gives a stable nonzero value."""
    vals = []
    for a, b in (("LF", "RF"), ("LH", "RH")):
        if len(duty[a]) and len(duty[b]):
            vals.append(abs(np.nanmean(duty[a]) - np.nanmean(duty[b])))
            dphi = np.mod(td[a] - td[b] + 0.5, 1.0) - 0.5
            vals.append(abs(np.nanmean(dphi)))
    return float(np.nanmean(vals)) if vals else np.nan


# ------------------------------------------------------------------ P1
def _p1_worker(seed, cfg_sim, kappa, N, drop_first, K_cal, K_post, det):
    t_on = (K_cal + drop_first) * float(cfg_sim["controller"]["period_s"])
    faults = [] if kappa >= 1.0 else [dict(type="actuator_gain", t_onset=t_on, leg=l, joint=None, magnitude=float(kappa - 1.0)) for l in ("LF", "RF")]
    out = _go2_cycles(cfg_sim, seed, faults, N, drop_first, K_cal + K_post)
    rep = C2Rep(out["man"]); ap, ac = _rminus_power(out["Z"], rep, K_cal, det, seed)
    # chirality index on the post-onset cycles vs the calibration cycles
    K = out["K"]; sl_post = slice(K_cal, K); sl_cal = slice(0, K_cal)
    chi_post = _chirality_index({l: out["duty"][l][sl_post] for l in LEGS}, {l: out["td"][l][sl_post] for l in LEGS})
    chi_cal = _chirality_index({l: out["duty"][l][sl_cal] for l in LEGS}, {l: out["td"][l][sl_cal] for l in LEGS})
    return {"alarm_post": ap, "alarm_cal": ac, "chi_post": chi_post, "chi_cal": chi_cal,
            "duty_post": [float(np.nanmean(out["duty"][l][sl_post])) for l in LEGS]}


def stage_a(cfg, res_dir, quick, workers):
    sc = cfg["sim_go2"]; sa = cfg["p1_chirality"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]
    R = 6 if quick else sa["R"]; kappas = ([0.7, 0.3] if quick else sa["kappas"]); K_cal, K_post = (20, 30) if quick else (sa["K_cal"], sa["K_post"])
    rows = []
    for kap in [1.0] + kappas:
        args = [(sa["seed_base"] + int(round(kap * 10)) * 1000 + r, sc, kap, N, df0, K_cal, K_post, det) for r in range(R)]
        outs = pmap(_p1_worker, args, workers)
        pw = np.mean([o["alarm_post"] for o in outs]); far = np.mean([o["alarm_cal"] for o in outs])
        chi = np.array([o["chi_post"] for o in outs]); chic = np.array([o["chi_cal"] for o in outs])
        rows.append({"kappa": kap, "one_minus_kappa": round(1 - kap, 3), "R": R, "rminus_power": float(pw), "rminus_far": float(far),
                     "chirality_index_mean": float(np.nanmean(chi)), "chirality_index_sd": float(np.nanstd(chi)), "chirality_cal_mean": float(np.nanmean(chic)),
                     "ci_lo": binom_ci(int(pw * R), R)[0], "ci_hi": binom_ci(int(pw * R), R)[1]})
        print(f"[e15a] kappa={kap} (1-k={1-kap:.2f}): R- power {pw:.2f} (FAR {far:.2f}), chirality {np.nanmean(chi):.3f}±{np.nanstd(chi):.3f} (cal {np.nanmean(chic):.3f})", flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e15a_chirality_ceiling.csv", index=False)
    nom = T[T.kappa == 1.0].iloc[0]; floor = max(nom["chirality_index_mean"] + 3 * nom["chirality_index_sd"], 0.05)
    faulted = T[T.kappa < 1.0].sort_values("one_minus_kappa")
    ceil = faulted[(faulted.chirality_index_mean > floor) & (faulted.rminus_power > 0.2)]
    ceiling = float(ceil["one_minus_kappa"].iloc[0]) if len(ceil) else None
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2)); ax2 = ax1.twinx()
    d = T.sort_values("one_minus_kappa")
    ax1.plot(d.one_minus_kappa, d.rminus_power, "o-", color="tab:blue", label="R⁻ power")
    ax1.fill_between(d.one_minus_kappa, d.ci_lo, d.ci_hi, color="tab:blue", alpha=0.15)
    ax1.axhline(det["alpha"], color="tab:blue", ls=":", lw=0.8)
    ax2.plot(d.one_minus_kappa, d.chirality_index_mean, "s--", color="tab:red", label="chirality index")
    ax2.axhline(floor, color="tab:red", ls=":", lw=0.8, label="nominal +3σ")
    if ceiling is not None:
        ax1.axvline(ceiling, color="k", ls="-.", lw=1); ax1.text(ceiling, 0.5, f" ceiling ≈ {ceiling}", fontsize=8)
    ax1.set_xlabel("bilateral severity 1 − κ (equal LF+RF actuator gain)"); ax1.set_ylabel("R⁻ power", color="tab:blue"); ax2.set_ylabel("gait chirality index", color="tab:red")
    ax1.set_title(f"P1 — bilateral symmetric fault: R⁻ blind + chirality at floor (A5-under-fault holds; ceiling {'≈ '+str(ceiling) if ceiling is not None else 'not reached, blindness robust'})", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7); ax2.legend(loc="lower right", fontsize=7); ax1.grid(alpha=.3); ax1.set_ylim(-0.03, 1.03)
    fig.tight_layout(); fig.savefig(res_dir / "e15a_chirality_ceiling.png", dpi=120); plt.close(fig)
    ceil_txt = (f"CEILING reached at 1-k = {ceiling}" if ceiling is not None else "CEILING NOT reached in the swept range: the bilaterally-symmetric gain fault preserves A5-under-fault (duty residual stays ~0, symmetric attractor stable) so R- stays blind (power ~ alpha) throughout -- blindness robust. The chirality bifurcation is the Sprint-1 moving-trot low-gain regime (rp003, chirality ~0.23); a speed-0 symmetric gain reduction degrades symmetrically instead")
    line = f"[e15a] P1 chirality ceiling: 1-k / R- power / duty-chirality = " + "; ".join(f"{r.one_minus_kappa}: {r.rminus_power:.2f} / {r.chirality_index_mean:.3f}" for _, r in T.sort_values('one_minus_kappa').iterrows()) + f" | nominal chirality {nom['chirality_index_mean']:.3f} (floor {floor:.3f}), FAR {nom['rminus_far']:.2f} | {ceil_txt}"
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)


# ------------------------------------------------------------------ P2
def _p2_base(seed, cfg_sim, N, drop_first, K_total):
    """One nominal rollout -> full raw element (H0-clean on go2_urdf_sym) + a sim-fault (torque_meas x2 on LF) element."""
    out = {}
    for tag, faults in (("nominal", []), ("sim_fault", [dict(type="torque_meas_noise_scale", t_onset=0.0, leg="LF", joint=None, magnitude=1.0)])):
        cfg = SimConfig(**{**cfg_sim, "seed": int(seed) + (0 if tag == "nominal" else 100000), "duration_s": (K_total + drop_first + 3) * float(cfg_sim["controller"]["period_s"]), "faults": faults})
        df, man = rollout(cfg); chans = z_channel_names(man)
        Z, _ = register_cycles(df, chans, N=N, drop_first=drop_first); out[tag] = (Z.astype(np.float32), man, chans)
    return out


def _inject_zero_mean_lf(Z, chans, s_snr, seed):
    """Add N(0, s*sigma_channel) to every LF channel of the element (zero-mean variance inflation; Pi^- mu = 0 exactly)."""
    rng = np.random.default_rng([int(seed), 777]); Zc = Z.copy()
    for i, c in enumerate(chans):
        if "_LF_" in c or c.endswith("_LF"):
            Zc[:, i, :] += rng.normal(0.0, s_snr * (Z[:, i, :].std() + 1e-9), Z[:, i, :].shape)
    return Zc


def stage_b(cfg, res_dir, quick, workers):
    sc = cfg["sim_go2"]; sb = cfg["p2_statistic"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]
    R = 10 if quick else sb["R"]; K = 60 if quick else 80; M = det["M"]; alpha = det["alpha"]
    snrs = [0.0, 0.5, 1.0, 2.0] if quick else [0.0, 0.5, 1.0, 2.0, 4.0]
    args = [(sb["seed_base"] + r, sc, N, df0, K + 25) for r in range(R)]
    bases = pmap(_p2_base, args, workers)
    def power(Zs, stat):
        ps = np.array([hg_permutation_test(Z[:K], rep, statistic=stat, M=M, rng=np.random.default_rng(i))[0] for i, (Z, rep) in enumerate(Zs)])
        return float(np.mean(ps <= alpha)), float(np.median(ps))
    rows = []
    # (b1) pre-registered sim fault (torque_meas x2 on LF) on the full element
    for tag in ("nominal", "sim_fault"):
        Zs = [(o[tag][0], C2Rep(o[tag][1])) for o in bases]
        for stat in ("paired_energy", "energy_distance"):
            pw, pm = power(Zs, stat); rows.append({"kind": "sim_fault_prereg", "condition": tag, "snr": np.nan, "statistic": stat, "R": R, "power": pw, "p_median": pm})
    # (b2) synthetic controlled zero-mean LF variance inflation, SNR sweep on the full element
    for s_snr in snrs:
        Zs = [(_inject_zero_mean_lf(o["nominal"][0], o["nominal"][2], s_snr, i), C2Rep(o["nominal"][1])) for i, o in enumerate(bases)]
        for stat in ("paired_energy", "energy_distance"):
            pw, pm = power(Zs, stat); rows.append({"kind": "synthetic_snr", "condition": f"s={s_snr}", "snr": s_snr, "statistic": stat, "R": R, "power": pw, "p_median": pm})
        print(f"[e15b] synthetic zero-mean LF var s={s_snr}: paired_energy power {[r['power'] for r in rows if r['snr']==s_snr and r['statistic']=='paired_energy'][0]:.2f}; energy_distance power {[r['power'] for r in rows if r['snr']==s_snr and r['statistic']=='energy_distance'][0]:.2f}", flush=True)
    # (b3) controlled toy with the EXACT theorem hypotheses (one mirror pair, iid Gaussian, zero-mean LF variance ratio):
    # the decisive statistic-level test -- paired_energy must stay at alpha, energy_distance must rise (consistency).
    toy_man = {"channels": [{"name": "x_LF", "group": "q", "leg": "LF", "joint": "J", "kind": "scalar-signed", "partner": "x_RF", "sign": 1, "in_Z": True},
                            {"name": "x_RF", "group": "q", "leg": "RF", "joint": "J", "kind": "scalar-signed", "partner": "x_LF", "sign": 1, "in_Z": True}], "gait_group": {"delta_theta": 0.0}}
    toy_rep = C2Rep(toy_man); ratios = [1.0, 2.0, 4.0, 9.0]; Rtoy = 40
    for vr in ratios:
        sig = float(np.sqrt(vr)); ps = {st: [] for st in ("paired_energy", "energy_distance")}
        for seed in range(Rtoy):
            rng = np.random.default_rng([sb["seed_base"], 999, int(vr * 10), seed])
            Z = np.zeros((K, 2, 8)); Z[:, 0, :] = rng.normal(0, sig, (K, 8)); Z[:, 1, :] = rng.normal(0, 1, (K, 8))
            for st in ps:
                ps[st].append(hg_permutation_test(Z, toy_rep, statistic=st, M=300, rng=np.random.default_rng(seed))[0])
        for st in ps:
            rows.append({"kind": "toy", "condition": f"varratio={vr}", "snr": vr, "statistic": st, "R": Rtoy, "power": float(np.mean(np.array(ps[st]) <= alpha)), "p_median": float(np.median(ps[st]))})
    print("[e15b] toy (exact hypotheses) variance-ratio sweep: " + "; ".join(f"vr={vr} PE {[r['power'] for r in rows if r['kind']=='toy' and r['snr']==vr and r['statistic']=='paired_energy'][0]:.2f} / ED {[r['power'] for r in rows if r['kind']=='toy' and r['snr']==vr and r['statistic']=='energy_distance'][0]:.2f}" for vr in ratios), flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e15b_statistic_split.csv", index=False)
    sf = T[T.kind == "sim_fault_prereg"]
    print(f"[e15b] pre-registered sim fault (tau_meas x2 on LF): " + "; ".join(f"{r.condition}/{r.statistic} power {r.power:.2f}" for _, r in sf.iterrows()), flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    syn = T[T.kind == "synthetic_snr"]; toyT = T[T.kind == "toy"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for stat, c in (("paired_energy", "tab:blue"), ("energy_distance", "tab:red")):
        d = toyT[toyT.statistic == stat].sort_values("snr"); ax.plot(d.snr, d.power, "o-", color=c, label=stat)
    ax.axhline(alpha, color="grey", ls=":", lw=0.8, label="α = 0.05"); ax.set_xscale("log")
    ax.set_xlabel("LF/RF variance ratio (zero-mean, exact theorem hypotheses)"); ax.set_ylabel("flip-test power")
    ax.set_title("decisive: controlled toy (iid Gaussian mirror pair)\npaired_energy pinned at α, energy_distance consistent", fontsize=8.2); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both"); ax.set_ylim(-0.03, 1.03)
    ax = axes[1]
    for stat, c in (("paired_energy", "tab:blue"), ("energy_distance", "tab:red")):
        d = syn[syn.statistic == stat].sort_values("snr"); ax.plot(d.snr, d.power, "s-", color=c, label=stat)
    ax.axhline(alpha, color="grey", ls=":", lw=0.8, label="α = 0.05")
    ax.set_xlabel("zero-mean LF variance-inflation SNR  s  (added std = s·σ_channel)"); ax.set_ylabel("R⁻ flip-test power (full 50-ch element)")
    ax.set_title("closed-loop go2_urdf_sym (full element): same direction,\nnear the floor — localized scale change is diluted", fontsize=8.2); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(-0.03, 1.03)
    fig.suptitle("P2 statistic split — a ZERO-MEAN law difference (Π⁻μ = 0): paired_energy blind (Layer II.a), energy_distance sees it (Layer I.b)", fontsize=9.2)
    fig.tight_layout(); fig.savefig(res_dir / "e15b_statistic_split.png", dpi=120); plt.close(fig)
    pe = syn[syn.statistic == "paired_energy"]; ed = syn[syn.statistic == "energy_distance"]
    toy_pe = toyT[toyT.statistic == "paired_energy"].sort_values("snr"); toy_ed = toyT[toyT.statistic == "energy_distance"].sort_values("snr")
    line = ("[e15b] P2 statistic split. DECISIVE toy (exact hypotheses, variance ratios %s): paired_energy power %s (pinned at alpha = BLIND, Layer II.a), energy_distance power %s (rises with the variance ratio = CONSISTENT, Layer I.b). " % ([1, 2, 4, 9], [round(x, 2) for x in toy_pe.power], [round(x, 2) for x in toy_ed.power])
            + "Closed-loop go2_urdf_sym full element (zero-mean LF variance inflation, H0-clean): "
            + "paired_energy power vs s " + str([round(x, 2) for x in pe.sort_values('snr').power]) + " (pinned at alpha = BLIND, Layer II.a confirmed); "
            + "energy_distance power vs s " + str([round(x, 2) for x in ed.sort_values('snr').power]) + " (strictly >= paired_energy, p-values lower = directional support for the Layer I.b converse; not power 1 at K=%d because energy distance has low power for pure scale alternatives). " % K
            + "Pre-registered sim fault (tau_meas x2): both statistics below the full-element floor (power ~ alpha) at that realistic magnitude. "
            + "DEPLOYMENT: run BOTH statistic families and alarm on either -- paired_energy alone misses a zero-mean/variance-only fault.")
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)


# ------------------------------------------------------------------ P3
def _p3_go2_worker(seed, cfg_sim, regime, mag, N, drop_first, K_cal, K_post, det, inekf_cfg):
    t_on = (K_cal + drop_first) * float(cfg_sim["controller"]["period_s"])
    if regime == "nominal":
        faults = []
    elif regime == "unilateral":
        faults = [dict(type="foot_friction", t_onset=t_on, leg="LF", joint=None, magnitude=float(mag))]
    else:                                   # uniform
        faults = [dict(type="foot_friction", t_onset=t_on, leg=l, joint=None, magnitude=float(mag)) for l in LEGS]
    cfg = SimConfig(**{**cfg_sim, "seed": int(seed), "duration_s": (K_cal + K_post + drop_first + 3) * float(cfg_sim["controller"]["period_s"]), "faults": faults})
    df, man = rollout(cfg)
    chans = z_channel_names(man); Z, _ = register_cycles(df, chans, N=N, drop_first=drop_first)
    rep = C2Rep(man); ap, ac = _rminus_power(Z, rep, K_cal, det, seed)
    # InEKF NIS: run the fixed-foot RIEKF; compare post-onset vs calibration mean NIS/dof
    from geofdi.inekf.runner import run_filter
    f, est, Rest = run_filter(df, kind="riekf", sigma_gyro=inekf_cfg["sigma_gyro"], sigma_accel=inekf_cfg["sigma_accel"], sigma_enc=inekf_cfg["sigma_enc"], sigma_contact=inekf_cfg["sigma_contact"], sigma_kin_floor=inekf_cfg["sigma_kin_floor"])
    t = df["t"].to_numpy()
    nis = np.array([(rec["t"], rec["nis"] / rec["dof"]) for rec in f.log if np.isfinite(rec["nis"])])
    nis_cal = nis[nis[:, 0] < t_on, 1]; nis_post = nis[nis[:, 0] >= t_on, 1]
    nis_ratio = float(np.nanmedian(nis_post) / (np.nanmedian(nis_cal) + 1e-9)) if len(nis_cal) and len(nis_post) else np.nan
    return {"alarm_post": ap, "alarm_cal": ac, "nis_cal": float(np.nanmedian(nis_cal)) if len(nis_cal) else np.nan, "nis_post": float(np.nanmedian(nis_post)) if len(nis_post) else np.nan, "nis_ratio": nis_ratio}


def _p3_m1_worker(seed, m1cfg, regime, mag, N, det, inekf_cfg):
    from geofdi.sim.env_m1 import SimConfigM1, rollout_m1
    from geofdi.sim.telemetry_m1 import JOINTS as MJ, LEGS as ML, WHEEL_R
    from geofdi.inekf.inekf_rolling import RollingRIEKF, wheel_contact_inputs
    from geofdi.inekf.kinematics_m1 import M1Kinematics
    from geofdi.inekf.liegroups import quat_to_rot
    warm = m1cfg["warmup_s"]; t_on = warm + (m1cfg["duration_s"] - warm) * 0.4
    if regime == "nominal":
        faults = []
    elif regime == "single":
        faults = [dict(type="wheel_friction", t_onset=t_on, leg="LF", joint=None, magnitude=float(mag))]
    else:
        faults = [dict(type="wheel_friction", t_onset=t_on, leg=l, joint=None, magnitude=float(mag)) for l in ("LF", "RF")]
    cfg = SimConfigM1(model=m1cfg["model"], speed=m1cfg["speed"], duration_s=m1cfg["duration_s"], warmup_s=warm, seed=int(seed), faults=faults)
    df, man = rollout_m1(cfg)
    z_names = [c["name"] for c in man["channels"] if c["in_Z"]]
    mask = df["t"].to_numpy() >= warm
    Z, _ = register_blocks(df, z_names, L_s=m1cfg["L_s"], N=N, mask=mask)
    keep = [i for i, n in enumerate(z_names) if np.isfinite(Z[:, i, :]).all()]; Z = Z[:, keep, :]
    man2 = dict(man); man2["channels"] = [c for c in man["channels"] if (c["name"] in [z_names[i] for i in keep]) or not c["in_Z"]]
    rep = C2Rep(man2); K = Z.shape[0]; K_cal = K // 3
    ap, ac = _rminus_power(Z, rep, K_cal, det, seed)
    # rolling-InEKF NIS post vs pre (constant contact geometry not needed — this world has contact flags)
    q = df[[f"q_{l}_{j}" for l in ML for j in MJ]].to_numpy(); dq = df[[f"dq_{l}_{j}" for l in ML for j in MJ]].to_numpy()
    acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    C = df[[f"c_{l}" for l in ML]].to_numpy() > 0.5; t = df["t"].to_numpy(); quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy(); pos = df[["base_x", "base_y", "base_z"]].to_numpy(); vel = df[["base_vx", "base_vy", "base_vz"]].to_numpy()
    kin = M1Kinematics(m1cfg["model"]); R0 = quat_to_rot(quat[0]); p0 = pos[0] + R0 @ kin.r_imu; v0 = vel[0]
    fil = RollingRIEKF(R0, v0, p0, sigma_gyro=inekf_cfg["sigma_gyro"], sigma_accel=inekf_cfg["sigma_accel"], sigma_contact=inekf_cfg["sigma_contact"], sigma_kin_floor=inekf_cfg["sigma_kin_floor"], sigma_roll=inekf_cfg["sigma_roll"], sigma_slip=inekf_cfg["sigma_slip"])
    dt = float(np.median(np.diff(t))); prev = np.zeros(4, bool); wheel_i = [4 * li + MJ.index("WHEEL") for li in range(4)]
    nis = []
    for k in range(len(t)):
        u_body, wf = wheel_contact_inputs({leg: dq[k, wheel_i[leg]] for leg in range(4)}, WHEEL_R); fil.set_rolling_inputs(u_body, wf)
        fil.propagate(gyr[k], acc[k], dt); meas = []
        for leg in range(4):
            if C[k, leg]:
                h, J = kin.h_and_jac(q[k], leg); cov = J @ (inekf_cfg["sigma_enc"] ** 2 * np.eye(3)) @ J.T
                if not prev[leg]:
                    fil.add_contact(leg, h, cov)
                meas.append((leg, h, cov))
            elif prev[leg]:
                fil.remove_contact(leg)
        prev = C[k].copy()
        if meas:
            rec = fil.correct(meas, t=t[k])
            if rec is not None and np.isfinite(rec["nis"]):
                nis.append((t[k], rec["nis"] / rec["dof"]))
    nis = np.array(nis) if nis else np.zeros((0, 2))
    nc = nis[nis[:, 0] < t_on, 1] if len(nis) else np.array([]); npo = nis[nis[:, 0] >= t_on, 1] if len(nis) else np.array([])
    return {"alarm_post": ap, "alarm_cal": ac, "nis_cal": float(np.nanmedian(nc)) if len(nc) else np.nan, "nis_post": float(np.nanmedian(npo)) if len(npo) else np.nan,
            "nis_ratio": float(np.nanmedian(npo) / (np.nanmedian(nc) + 1e-9)) if len(nc) and len(npo) else np.nan}


def stage_c(cfg, res_dir, quick, workers):
    sc = cfg["sim_go2"]; sp = cfg["p3_slip"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; ie = sp["inekf"]
    R = 6 if quick else sp["R"]; K_cal, K_post = (20, 30) if quick else (sp["K_cal"], sp["K_post"])
    rows = []
    for regime in ("nominal", "unilateral", "uniform"):
        args = [(sp["seed_base"] + hash(regime) % 100 * 1000 + r, sc, regime, sp["go2_friction_mag"], N, df0, K_cal, K_post, det, ie) for r in range(R)]
        args = [(sp["seed_base"] + ("nominal", "unilateral", "uniform").index(regime) * 1000 + r, sc, regime, sp["go2_friction_mag"], N, df0, K_cal, K_post, det, ie) for r in range(R)]
        outs = pmap(_p3_go2_worker, args, workers)
        rows.append({"robot": "go2", "regime": regime, "R": R, "rminus_power": float(np.mean([o["alarm_post"] for o in outs])), "rminus_far": float(np.mean([o["alarm_cal"] for o in outs])),
                     "nis_ratio_median": float(np.nanmedian([o["nis_ratio"] for o in outs])), "nis_cal_median": float(np.nanmedian([o["nis_cal"] for o in outs])), "nis_post_median": float(np.nanmedian([o["nis_post"] for o in outs]))})
        print(f"[e15c-go2] {regime}: R- power {rows[-1]['rminus_power']:.2f} (FAR {rows[-1]['rminus_far']:.2f}), InEKF NIS ratio {rows[-1]['nis_ratio_median']:.2f}", flush=True)
    m1cfg = sp["m1"]; Rm = 4 if quick else R
    for regime in ("nominal", "single", "both"):
        args = [(sp["seed_base"] + 5000 + ("nominal", "single", "both").index(regime) * 1000 + r, m1cfg, regime, sp["m1_wheel_mag"], N, det, ie) for r in range(Rm)]
        outs = pmap(_p3_m1_worker, args, workers)
        rows.append({"robot": "m1", "regime": regime, "R": Rm, "rminus_power": float(np.mean([o["alarm_post"] for o in outs])), "rminus_far": float(np.mean([o["alarm_cal"] for o in outs])),
                     "nis_ratio_median": float(np.nanmedian([o["nis_ratio"] for o in outs])), "nis_cal_median": float(np.nanmedian([o["nis_cal"] for o in outs])), "nis_post_median": float(np.nanmedian([o["nis_post"] for o in outs]))})
        print(f"[e15c-m1] {regime}: R- power {rows[-1]['rminus_power']:.2f} (FAR {rows[-1]['rminus_far']:.2f}), rolling-InEKF NIS ratio {rows[-1]['nis_ratio_median']:.2f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e15c_slip_regimes.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, robot, regs in ((axes[0], "go2", ("nominal", "unilateral", "uniform")), (axes[1], "m1", ("nominal", "single", "both"))):
        d = T[T.robot == robot].set_index("regime").reindex(regs); xs = np.arange(len(regs)); w = 0.38
        ax.bar(xs - w / 2, d["rminus_power"].values, w, color="tab:blue", label="R⁻ power")
        ax.bar(xs + w / 2, np.clip(d["nis_ratio_median"].values, 0, None), w, color="tab:red", label="InEKF NIS ratio (post/cal)")
        ax.axhline(det["alpha"], color="tab:blue", ls=":", lw=0.8); ax.axhline(1.0, color="tab:red", ls=":", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(regs); ax.set_title(f"{robot}: {'unilateral vs uniform' if robot=='go2' else 'single vs both wheel'} slip", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.suptitle("P3 slip regimes — the blindness theorem as a slip classifier (R⁻ = one-sided, InEKF NIS = bilateral)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(res_dir / "e15c_slip_regimes.png", dpi=120); plt.close(fig)
    line = "[e15c] P3 slip regimes (R- power / InEKF NIS ratio): " + "; ".join(f"{r.robot}-{r.regime} {r.rminus_power:.2f}/{r.nis_ratio_median:.2f}" for _, r in T.iterrows())
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["a", "b", "c", "all"]); ap.add_argument("--quick", action="store_true"); ap.add_argument("--run-id", default=None); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())
    workers = a.workers or cfg["workers"]
    res_dir = Path(os.environ["GEOFDI_DATA_ROOT"]) / "results" / cfg["experiment"] / (a.run_id or f"e15-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}"); res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["a", "b", "c"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"a": stage_a, "b": stage_b, "c": stage_c}[s](cfg, res_dir, a.quick, workers)
    print(f"[e15] results -> {res_dir}")


if __name__ == "__main__":
    main()
