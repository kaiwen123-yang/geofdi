#!/usr/bin/env python3
"""e13 weld addendum (Sprint 8 Block L3) — e13a/e13b on the WELD world (leg = 3-DoF arm), RA-L split material.

The floating-base e13 (run.py) established: residual R⁻ inherits H₀ (analytic + equivariant DeLaN in band); a plain
(non-equivariant) DeLaN contaminates H₀ at a rate that scales with its equivariance defect δ_f. This addendum reruns the
two headline stages on the welded-trunk world (`weld_base=True`, LF/RF and LH/RH are the mirror pairs, legs swing in the
air so the contact term is 0), reusing the e13 worker/residual machinery and the frozen weld DeLaN models
(`models/delan_weld/weld_equiv_v1` δ_f q95 = 0, `weld_plain_v1` δ_f q95 = 1.60; Sprint 6 Block Q):

  e13a_weld : power of the residual R⁻ flip test vs the raw R⁻ under an LF-KFE actuator-gain fault ladder (κ = 1 − mag),
              analytic residual + equivariant-DeLaN residual + plain-DeLaN residual; R rollouts each.
  e13b_weld : size of the residual flip test under H₀ (no fault) for the analytic / equivariant / plain residuals, plus
              the H₀′ differenced test and the naive-centred variant (Lemma centring) — the contamination table.

    experiments/e13_residual_symmetry/weld_addendum.py [--R 50] [--quick]
Outputs -> $GEOFDI_DATA_ROOT/results/e13_residual_symmetry/e13weld-<stamp>/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import run as e13                                              # reuse _sim/_worker/_delan_cycles/_RES_REP/etc.
from geofdi.detect.permutation import hg_permutation_test
from geofdi.groups.c2 import C2Rep
from geofdi.residuals.mirror_pairs import isotypic_split
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap

DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
WELD_TAGS = ["weld_equiv_v1", "weld_plain_v1"]


def _load_weld_models(tags):
    import torch
    from geofdi.dynamics.delan_equiv import load_delan
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for t in tags:
        if t not in e13._MODELS:
            e13._MODELS[t] = load_delan(DATA_ROOT / "models" / "delan_weld" / t, device=dev)
    return tags


def _weld_defects(tags):
    out = {}
    for t in tags:
        r = json.loads((DATA_ROOT / "models" / "delan_weld" / t / "report.json").read_text())
        d = r.get("defect", {}) or {}
        out[t] = {"delta_q95": d.get("q95", np.nan), "delta_q50": d.get("q50", np.nan), "equivariant": bool(r.get("equivariant", False))}
    return out


def _cycles_from_out(out):
    """Parent-side: turn an e13 worker output (raw cycles + per-leg arrays) into the residual variants (needs GPU models)."""
    Zr = e13._delan_cycles(out, WELD_TAGS)         # pops out['arrays']; runs in the parent (no CUDA-in-fork)
    return {"K": out["K"], "raw": out["Z"], "man": out["man"], "res_an": out["Zr_an"], **{f"res::{t}": Zr[t] for t in WELD_TAGS}}


def _rminus_p(Z, rep, seed, M=512):
    if Z.shape[0] < 6 or not np.isfinite(Z).all():
        return np.nan
    p, _ = hg_permutation_test(Z, rep, statistic="paired_energy", M=M, rng=np.random.default_rng(seed))
    return float(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--R", type=int, default=50); ap.add_argument("--quick", action="store_true"); ap.add_argument("--run-id", default=None)
    a = ap.parse_args(); R = 12 if a.quick else a.R
    cfg = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())
    oc = cfg["observer"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; det = cfg["detect"]; alpha = det["alpha"]; M = det["M"]
    res_dir = DATA_ROOT / "results" / e13.EXP_NAME / (a.run_id or f"e13weld-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}"); res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump({"stage": "weld_addendum", "R": R, "world": "weld_base", "models": WELD_TAGS, "sim": cfg["sim"], "observer": oc}, sort_keys=False))
    _load_weld_models(WELD_TAGS); defects = _weld_defects(WELD_TAGS)
    rep_res = e13._RES_REP; workers = cfg["workers"]
    variants = ["raw", "res_an", "res::weld_equiv_v1", "res::weld_plain_v1"]
    vlabel = {"raw": "raw", "res_an": "analytic residual", "res::weld_equiv_v1": "equivariant DeLaN residual (δ_f=0)", "res::weld_plain_v1": "plain DeLaN residual (δ_f=1.60)"}

    # ---------------- e13b_weld: size under H0 (no fault) ----------------
    K_cal, K_test = (30, 30) if a.quick else (60, 60)
    print(f"[e13b_weld] {R} nominal weld rollouts x {len(variants)} residual variants (K_test {K_test})", flush=True)
    argsN = [(e13._sim(cfg, 70000 + r, weld_base=True), K_cal + K_test, N, df0, oc, True) for r in range(R)]
    outsN = [_cycles_from_out(o) for o in pmap(e13._worker, argsN, workers)]
    rows = []
    for ri, o in enumerate(outsN):
        K = o["K"]
        for v in variants:
            Z = o[v]; rep = C2Rep(o["man"]) if v == "raw" else rep_res
            cal = Z[:K_cal]; test = Z[K_cal:K_cal + K_test]
            for mode, X in (("plain", test), ("centred", test - cal.mean(0)), ("h0prime", test[:min(len(test), len(cal))] - cal[:min(len(test), len(cal))])):
                rows.append({"rep": ri, "variant": v, "mode": mode, "p": _rminus_p(X, rep, [70000, ri, variants.index(v), ("plain", "centred", "h0prime").index(mode)], M)})
    runsB = pd.DataFrame(rows); runsB.to_csv(res_dir / "e13b_weld_runs.csv", index=False)
    summ = []
    for (v, mode), g in runsB.groupby(["variant", "mode"]):
        pv = g.p.dropna(); k = int((pv <= alpha).sum()); n = len(pv); band = nominal_band(alpha, n)
        d = defects.get(v.split("::")[-1], {}) if v.startswith("res::") else {"delta_q95": 0.0 if v in ("raw", "res_an") else np.nan, "equivariant": True}
        summ.append({"variant": v, "label": vlabel[v], "mode": mode, "size": k / n if n else np.nan, "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1],
                     "band_lo": band[0], "band_hi": band[1], "in_band": bool(band[0] <= k / n <= band[1]) if n else False, "n": n, "delta_f_q95": d.get("delta_q95", np.nan)})
    S = pd.DataFrame(summ); S.to_csv(res_dir / "e13b_weld_size.csv", index=False)

    # ---------------- e13a_weld: power under an LF-KFE actuator-gain ladder ----------------
    ladder = ([0.90] if a.quick else cfg["stage_a"]["faults"]["actuator_gain"])  # kappa
    K_cal_a, K_post = (30, 40) if a.quick else (40, 60)
    t_on = e13._onset_time(K_cal_a, df0)
    powrows = []
    for kappa in ladder:
        fault = dict(type="actuator_gain", t_onset=t_on, leg="LF", joint="KFE", magnitude=float(kappa - 1.0))
        print(f"[e13a_weld] LF-KFE actuator_gain kappa={kappa} (1-kappa={1-kappa:.2f}); {R} rollouts", flush=True)
        argsF = [(e13._sim(cfg, 71000 + int(kappa * 1000) + r, weld_base=True, faults=[fault]), K_cal_a + K_post, N, df0, oc, True) for r in range(R)]
        outsF = [_cycles_from_out(o) for o in pmap(e13._worker, argsF, workers)]
        for ri, o in enumerate(outsF):
            for v in variants:
                Z = o[v]; rep = C2Rep(o["man"]) if v == "raw" else rep_res
                # post-onset differenced test (H0': monitoring window minus calibration window) — exact under H0 whatever δ_f
                cal = Z[:K_cal_a]; post = Z[K_cal_a:]; K = min(len(cal), len(post))
                p_h0p = _rminus_p(post[:K] - cal[:K], rep, [71000, ri, variants.index(v), int(kappa * 100)], M)
                p_plain = _rminus_p(post, rep, [71001, ri, variants.index(v), int(kappa * 100)], M)
                powrows.append({"kappa": kappa, "one_minus_kappa": round(1 - kappa, 3), "rep": ri, "variant": v, "p_h0prime": p_h0p, "p_plain": p_plain})
    runsA = pd.DataFrame(powrows); runsA.to_csv(res_dir / "e13a_weld_runs.csv", index=False)
    powsum = []
    for (kappa, v), g in runsA.groupby(["kappa", "variant"]):
        for col in ("p_h0prime", "p_plain"):
            pv = g[col].dropna(); k = int((pv <= alpha).sum()); n = len(pv)
            powsum.append({"kappa": kappa, "one_minus_kappa": round(1 - kappa, 3), "variant": v, "label": vlabel[v], "test": col, "power": k / n if n else np.nan, "ci_lo": binom_ci(k, n)[0], "ci_hi": binom_ci(k, n)[1], "n": n})
    P = pd.DataFrame(powsum); P.to_csv(res_dir / "e13a_weld_power.csv", index=False)

    # ---------------- figure ----------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for v, c in zip(variants, ("k", "tab:green", "tab:blue", "tab:red")):
        d = P[(P.variant == v) & (P.test == "p_h0prime")].sort_values("one_minus_kappa")
        if len(d):
            ax.plot(d.one_minus_kappa, d.power, "o-", color=c, label=vlabel[v])
    ax.axhline(alpha, color="grey", ls=":", lw=0.8); ax.set_xlabel("fault severity 1 − κ (LF-KFE actuator gain)"); ax.set_ylabel("R⁻ H₀′ power (differenced test)")
    ax.set_title("e13a-weld: residual R⁻ power on the welded leg", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(-0.03, 1.03)
    ax = axes[1]
    plain = S[S["mode"] == "plain"].set_index("variant").reindex(variants)
    xs = np.arange(len(variants)); ax.bar(xs, plain["size"].values, color=["grey", "tab:green", "tab:blue", "tab:red"])
    for i, v in enumerate(variants):
        r = plain.loc[v]; ax.errorbar(i, r["size"], yerr=[[r["size"] - r["ci_lo"]], [r["ci_hi"] - r["size"]]], color="k", capsize=3, lw=1)
    b = S[S["mode"] == "plain"].iloc[0]; ax.axhspan(b["band_lo"], b["band_hi"], color="green", alpha=0.12, label=f"H₀ band [{b['band_lo']:.2f}, {b['band_hi']:.2f}]")
    ax.axhline(alpha, color="grey", ls=":", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(["raw", "analytic\nresidual", "equiv DeLaN\nδ_f=0", "plain DeLaN\nδ_f=1.60"], fontsize=8)
    ax.set_ylabel("R⁻ flip-test size under H₀"); ax.set_title("e13b-weld: plain DeLaN contaminates H₀ (Cor contamination)", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.suptitle("e13 weld addendum (leg = 3-DoF arm, LF/RF mirror pair) — RA-L split material", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e13weld_power_and_contamination.png", dpi=120); plt.close(fig)

    # ---------------- conclusions ----------------
    def sz(v, mode="plain"):
        r = S[(S.variant == v) & (S["mode"] == mode)]
        return (f"{r['size'].iloc[0]:.3f}{'' if r['in_band'].iloc[0] else ' OUT'}") if len(r) else "nan"
    lines = [
        f"[e13a-weld] R⁻ H₀′ power vs 1−κ (LF-KFE), " + "; ".join(
            f"{vlabel[v]}: " + "/".join(f"{P[(P.variant==v)&(P.test=='p_h0prime')&(P.kappa==k)]['power'].iloc[0]:.2f}" for k in ladder if len(P[(P.variant==v)&(P.test=='p_h0prime')&(P.kappa==k)])) for v in variants) + f" (at 1−κ = {[round(1-k,2) for k in ladder]})",
        f"[e13b-weld] H₀ size (plain flip test): raw {sz('raw')}, analytic residual {sz('res_an')}, equivariant DeLaN {sz('res::weld_equiv_v1')} (δ_f=0), plain DeLaN {sz('res::weld_plain_v1')} (δ_f={defects['weld_plain_v1']['delta_q95']:.2f}) | band [{S[S['mode']=='plain']['band_lo'].iloc[0]:.3f}, {S[S['mode']=='plain']['band_hi'].iloc[0]:.3f}]",
        f"[e13b-weld] H₀′ differenced test restores size for the plain model: plain DeLaN h0prime {sz('res::weld_plain_v1','h0prime')} vs plain {sz('res::weld_plain_v1','plain')}; naive centred plain DeLaN {sz('res::weld_plain_v1','centred')} (Lemma centring: centring does NOT fix it)",
    ]
    (res_dir / "conclusions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"[e13weld] results -> {res_dir}")


if __name__ == "__main__":
    main()
