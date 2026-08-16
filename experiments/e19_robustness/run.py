#!/usr/bin/env python3
"""e19 — robustness sweeps of the R- test size (Sprint 9 B2). Three reviewer probes, all under H0 (nominal world):

  phase : the phase estimate is WRONG by +-{2,5,10} % of the period (a linear clock-rate error). Does the flip test stay
          at its level when cycles are registered on a mis-scaled phase?
  kcal  : H0' calibration-set size K_cal in {60, 200, 400} -- how much nominal data does the recalibrated null need?
  block : block-flip length vs the true nuisance correlation time (ratio {0.5, 1, 2}) under an autocorrelated but
          mirror-symmetric-in-law nuisance (drift_lateral). Does a mismatched block length break the level?

    python experiments/e19_robustness/run.py [--stage phase|kcal|block|all] [--quick]
Outputs -> $GEOFDI_DATA_ROOT/results/e19_robustness/<run>/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.detect.h0prime import h0prime_test
from geofdi.detect.permutation import hg_permutation_test
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap
from geofdi.sim.telemetry import z_channel_names

DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
CFG = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())


def _sim(seed, **over):
    s = dict(CFG["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    s.update(over); return s


def _cycles(sim_cfg, K_total, N, drop_first, phase_err_pct=0.0):
    """One rollout -> registered cycles. `phase_err_pct` mis-scales the phase clock by that percentage (a systematic
    clock-rate error: theta' = (theta * (1+e)) mod 1), which is what a biased period estimate does."""
    cfg = SimConfig(**sim_cfg); per = float(cfg.controller.get("period_s", 0.5))
    # margin: a phase clock slowed by e % yields e % FEWER wraps, so ask for enough cycles that the slowest sweep point
    # still delivers K_total registered cycles (without it every worker silently returns None at negative errors).
    margin = 1.0 + abs(float(phase_err_pct)) / 100.0 + 0.15
    cfg.duration_s = (K_total * margin + drop_first + 3) * per
    df, man = rollout(cfg)
    if phase_err_pct:
        t = df["t"].to_numpy(); th = df["theta"].to_numpy()
        cyc = np.floor(np.cumsum(np.r_[0, np.diff(th) < -0.5]))            # cycle index
        th_lin = cyc + th                                                   # unwrapped phase in cycles
        df = df.copy(); df["theta"] = np.mod(th_lin * (1.0 + phase_err_pct / 100.0), 1.0)
    chans = z_channel_names(man)
    Z, _ = register_cycles(df, chans, N=N, drop_first=drop_first)
    return Z[:K_total].astype(np.float32), man


def _worker_phase(seed, err, K, N, df0, sim_cfg):
    Z, man = _cycles(sim_cfg, K, N, df0, phase_err_pct=err)
    if Z.shape[0] < K:
        return None
    rep = C2Rep(man)
    p_h0, _ = hg_permutation_test(Z, rep, statistic="paired_energy", M=CFG["detect"]["M"], rng=np.random.default_rng(int(seed)))
    Kh = K // 2
    r = h0prime_test(Z[:Kh], Z[Kh:2 * Kh], rep, M=CFG["detect"]["M"], rng=np.random.default_rng(int(seed) + 1))
    return {"p_h0": float(p_h0), "p_h0prime": float(r["p"]), "nu_cal": float(r["nu_cal"]), "nu_mon": float(r["nu_mon"])}


def stage_phase(res_dir, quick, workers):
    sp = CFG["sweep_phase"]; R = 10 if quick else sp["R"]; K = 30 if quick else sp["K"]
    N = CFG["registration"]["N"]; df0 = CFG["registration"]["drop_first"]; alpha = CFG["detect"]["alpha"]
    rows = []
    for err in sp["errors_pct"]:
        outs = [o for o in pmap(_worker_phase, [(sp["seed_base"] + 1000 * abs(int(err)) + (0 if err >= 0 else 500) + r, err, K, N, df0, _sim(sp["seed_base"] + r)) for r in range(R)], workers) if o]
        for tag, key in (("H0", "p_h0"), ("H0prime", "p_h0prime")):
            p = np.array([o[key] for o in outs]); k = int((p <= alpha).sum()); n = len(p)
            if n == 0:
                print(f"[e19 phase] error {err:+3.0f} %: NO usable run (all rollouts short of K) — reported as missing", flush=True)
                rows.append({"sweep": "phase", "phase_error_pct": err, "test": tag, "R": 0, "size": float("nan"),
                             "ci_lo": float("nan"), "ci_hi": float("nan"), "band_lo": float("nan"), "band_hi": float("nan"), "in_band": False})
                continue
            band = nominal_band(alpha, n)
            rows.append({"sweep": "phase", "phase_error_pct": err, "test": tag, "R": n, "size": k / n,
                         "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1], "band_lo": band[0], "band_hi": band[1],
                         "in_band": bool(band[0] <= k / n <= band[1])})
        print(f"[e19 phase] error {err:+3.0f} %: H0 size {rows[-2]['size']:.3f} ({'in' if rows[-2]['in_band'] else 'OUT'}), "
              f"H0' size {rows[-1]['size']:.3f} ({'in' if rows[-1]['in_band'] else 'OUT'})", flush=True)
    return pd.DataFrame(rows)


def _worker_kcal(seed, Kmax, N, df0, sim_cfg, kcals, K_mon):
    Z, man = _cycles(sim_cfg, Kmax, N, df0)
    if Z.shape[0] < Kmax:
        return None
    rep = C2Rep(man); out = {}
    for kc in kcals:
        if kc + K_mon > Z.shape[0]:
            continue
        r = h0prime_test(Z[:kc], Z[kc:kc + K_mon], rep, M=CFG["detect"]["M"], rng=np.random.default_rng([int(seed), kc]))
        out[kc] = {"p": float(r["p"]), "nu_cal": float(r["nu_cal"]), "nu_mon": float(r["nu_mon"])}
    return out


def stage_kcal(res_dir, quick, workers):
    sk = CFG["sweep_kcal"]; R = 10 if quick else sk["R"]; kcals = [60, 120] if quick else sk["K_cals"]
    K_mon = 30 if quick else sk["K_mon"]; Kmax = max(kcals) + K_mon
    N = CFG["registration"]["N"]; df0 = CFG["registration"]["drop_first"]; alpha = CFG["detect"]["alpha"]
    outs = [o for o in pmap(_worker_kcal, [(sk["seed_base"] + r, Kmax, N, df0, _sim(sk["seed_base"] + r), kcals, K_mon) for r in range(R)], workers) if o]
    rows = []
    for kc in kcals:
        p = np.array([o[kc]["p"] for o in outs if kc in o]); k = int((p <= alpha).sum()); n = len(p)
        band = nominal_band(alpha, n)
        nu_sd = float(np.std([o[kc]["nu_cal"] for o in outs if kc in o]))
        rows.append({"sweep": "kcal", "K_cal": kc, "K_mon": K_mon, "test": "H0prime", "R": n, "size": k / n,
                     "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1], "band_lo": band[0], "band_hi": band[1],
                     "in_band": bool(band[0] <= k / n <= band[1]), "nu_cal_sd": nu_sd})
        print(f"[e19 kcal] K_cal {kc}: H0' size {k/n:.3f} ({'in' if rows[-1]['in_band'] else 'OUT'}), nu_cal sd {nu_sd:.3f}", flush=True)
    return pd.DataFrame(rows)


def _worker_block(seed, K, N, df0, sim_cfg, ratios, tau_cycles):
    Z, man = _cycles(sim_cfg, K, N, df0)
    if Z.shape[0] < K:
        return None
    rep = C2Rep(man); out = {}
    for rt in ratios:
        B = max(1, int(round(rt * tau_cycles)))
        p, _ = hg_permutation_test(Z, rep, statistic="paired_energy", M=CFG["detect"]["M"],
                                   rng=np.random.default_rng([int(seed), B]), block=B) if _supports_block() else (np.nan, None)
        out[rt] = float(p)
    return out


def _supports_block():
    import inspect
    return "block" in inspect.signature(hg_permutation_test).parameters


def stage_block(res_dir, quick, workers):
    sb = CFG["sweep_block"]; R = 10 if quick else sb["R"]; K = 40 if quick else sb["K"]
    N = CFG["registration"]["N"]; df0 = CFG["registration"]["drop_first"]; alpha = CFG["detect"]["alpha"]
    tau = sb["tau_cycles"]
    if not _supports_block():
        print("[e19 block] hg_permutation_test has no `block` argument — running the sweep by BLOCKING THE ELEMENTS instead "
              "(averaging groups of B consecutive cycles), which is the equivalent construction.", flush=True)
    rows = []
    # nuisance: an autocorrelated but mirror-symmetric-in-law lateral drift, correlation time tau cycles
    per = float(CFG["sim"]["controller"]["period_s"]); tau_s = tau * per
    nuis = [dict(type="drift_lateral", t_onset=0.0, magnitude=0.02, params={"tau_s": tau_s})]
    for rt in sb["ratios"]:
        B = max(1, int(round(rt * tau)))
        ps = []
        args = [(sb["seed_base"] + 100 * int(rt * 10) + r, _sim(sb["seed_base"] + r, nuisance=nuis), K, N, df0, B) for r in range(R)]
        for o in pmap(_worker_block_avg, args, workers):
            if o is not None:
                ps.append(o)
        p = np.array(ps); k = int((p <= alpha).sum()); n = len(p); band = nominal_band(alpha, n)
        rows.append({"sweep": "block", "ratio_B_over_tau": rt, "B_cycles": B, "tau_cycles": tau, "test": "H0", "R": n,
                     "size": k / n, "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1], "band_lo": band[0],
                     "band_hi": band[1], "in_band": bool(band[0] <= k / n <= band[1])})
        print(f"[e19 block] B/tau {rt} (B={B} cycles): H0 size {k/n:.3f} ({'in' if rows[-1]['in_band'] else 'OUT'})", flush=True)
    return pd.DataFrame(rows)


def _worker_block_avg(seed, sim_cfg, K, N, df0, B):
    """Block construction: average B consecutive cycles into one element, then the ordinary flip test. B >= the nuisance
    correlation time makes the elements approximately independent; B < it leaves the correlation in."""
    Z, man = _cycles(sim_cfg, K, N, df0)
    if Z.shape[0] < K:
        return None
    nb = Z.shape[0] // B
    Zb = Z[:nb * B].reshape(nb, B, Z.shape[1], Z.shape[2]).mean(axis=1)
    if nb < 6:
        return None
    p, _ = hg_permutation_test(Zb, C2Rep(man), statistic="paired_energy", M=CFG["detect"]["M"], rng=np.random.default_rng(int(seed)))
    return float(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["phase", "kcal", "block", "all"])
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--run-id", default=None); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); workers = a.workers or CFG["workers"]
    res_dir = DATA / "results" / CFG["experiment"] / (a.run_id or f"e19-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}")
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(CFG, sort_keys=False))
    stages = ["phase", "kcal", "block"] if a.stage == "all" else [a.stage]
    frames = []
    for s in stages:
        frames.append({"phase": stage_phase, "kcal": stage_kcal, "block": stage_block}[s](res_dir, a.quick, workers))
    T = pd.concat(frames, ignore_index=True); T.to_csv(res_dir / "e19_robustness.csv", index=False)
    _plot(res_dir, T)
    def _x(r):
        for c in ("phase_error_pct", "K_cal", "ratio_B_over_tau"):
            if c in r and pd.notna(r[c]):
                return r[c]
        return "?"
    line = "[e19] " + " | ".join(
        f"{sw}: " + ", ".join(f"{_x(r)}→{r['size']:.3f}{'' if r['in_band'] else '*'}" for _, r in g.iterrows() if r["test"] in ("H0", "H0prime"))
        for sw, g in T.groupby("sweep")) + "  (* = outside the nominal band)"
    (res_dir / "conclusions.txt").write_text(line + "\n"); print(line)
    print(f"[e19] results -> {res_dir}")


def _plot(res_dir, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    alpha = CFG["detect"]["alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]; d = T[T.sweep == "phase"]
    for tag, c in (("H0", "tab:red"), ("H0prime", "tab:blue")):
        g = d[d.test == tag].sort_values("phase_error_pct")
        ax.errorbar(g.phase_error_pct, g["size"], yerr=[g["size"] - g.ci_lo, g.ci_hi - g["size"]], fmt="o-", color=c, capsize=3, label=tag)
    if len(d):
        ax.axhspan(d.band_lo.iloc[0], d.band_hi.iloc[0], color="green", alpha=0.12, label="nominal band")
    ax.axhline(alpha, color="k", ls=":", lw=0.8); ax.set_xlabel("phase-clock error [% of period]"); ax.set_ylabel("size at α")
    ax.set_title("B2a: mis-estimated gait phase", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1]; d = T[T.sweep == "kcal"]
    if len(d):
     ax.errorbar(d.K_cal, d["size"], yerr=[d["size"] - d.ci_lo, d.ci_hi - d["size"]], fmt="s-", color="tab:blue", capsize=3)
    if len(d):
        ax.axhspan(d.band_lo.iloc[0], d.band_hi.iloc[0], color="green", alpha=0.12)
    ax.axhline(alpha, color="k", ls=":", lw=0.8); ax.set_xlabel("K_cal (calibration cycles)"); ax.set_ylabel("H₀′ size at α")
    ax.set_title("B2b: how much nominal data H₀′ needs", fontsize=9); ax.grid(alpha=.3)
    ax = axes[2]; d = T[T.sweep == "block"]
    if len(d):
     ax.errorbar(d.ratio_B_over_tau, d["size"], yerr=[d["size"] - d.ci_lo, d.ci_hi - d["size"]], fmt="^-", color="tab:purple", capsize=3)
    if len(d):
        ax.axhspan(d.band_lo.iloc[0], d.band_hi.iloc[0], color="green", alpha=0.12)
    ax.axhline(alpha, color="k", ls=":", lw=0.8); ax.set_xlabel("block length / nuisance correlation time")
    ax.set_title("B2c: block-length mismatch under a correlated,\nmirror-symmetric-in-law nuisance", fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("e19 — robustness of the R⁻ level to the three standard reviewer probes", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e19_robustness.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
