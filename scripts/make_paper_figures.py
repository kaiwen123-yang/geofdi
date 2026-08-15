#!/usr/bin/env python3
"""make_paper_figures.py — one-shot regeneration of the T-RO paper's tables and figures from the review-pack run CSVs
(Sprint 7 Block F, the figure factory). See docs/paper/figure_plan.md for the F#/T# ↔ source-run map.

Every artifact is built from a CSV that lives in a review pack (the figure_plan rule: no paper artifact comes from a
run that is not in a pack). Output lands in $GEOFDI_DATA_ROOT/results/paper/{tables,figures}/ with:
  - tables/T*.csv            paper-ready CSVs (selected/renamed columns, a `source` tag where merged)
  - paper_tables.tex         all ten tables as booktabs tabulars (\\input-able)
  - figures/F*.pdf|.png      vector regenerations of the CSV-reproducible figures; the multi-panel merges that are
                             composed by hand in the manuscript are REGISTERED (source path recorded) not redrawn
  - coverage.md              every F#/T# with its source run, pack, and status (generated / registered / doc-derived)

Usage:
  make_paper_figures.py [--check] [--only T1,F6,...]
    --check : verify all source CSVs exist and print the coverage table; build nothing.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ["GEOFDI_DATA_ROOT"]); RES = DATA / "results"
OUT = RES / "paper"; TAB = OUT / "tables"; FIG = OUT / "figures"

# exp -> run_id (the run ids that are in the review packs; figure_plan.md is the source of truth)
RUN = {
    "e01": "s1-20260815-1422", "e02": "s3-20260815-1734", "e03": "e03-20260816", "e03e1": "e1-20260816",
    "e04": "s2-20260815-1652", "e05": "e05-20260815-1859", "e05a": "e05a-kcal200", "e06": "e06-20260815-2003",
    "e07": "e07-20260815-final", "e08": "e08-20260816", "e09": "e09-20260816b", "e10": "e10-20260816",
    "e11": "e11-20260816", "e13": "e13-20260815-2334", "e13b": "e13b-K-20260815", "ladder": "q-20260815",
}
EXPDIR = {"e01": "e01_h0_qq", "e02": "e02_inekf_money_figure", "e03": "e03_liu_a1_headtohead",
          "e03e1": "e03_liu_a1_headtohead", "e04": "e04_power_matrix", "e05": "e05_residual_channel",
          "e05a": "e05_residual_channel", "e06": "e06_n3_isolability", "e07": "e07_baselines",
          "e08": "e08_low_snr", "e09": "e09_three_channel", "e10": "e10_n2_signatures", "e11": "e11_sequential",
          "e13": "e13_residual_symmetry", "e13b": "e13_residual_symmetry", "ladder": "delan_ladder"}


def src(exp, fname):
    return RES / EXPDIR[exp] / RUN[exp] / fname


def _tex(df, caption, label, floatfmt="%.3g", maxcol=12):
    d = df.copy()
    if d.shape[1] > maxcol:
        d = d.iloc[:, :maxcol]
    cols = list(d.columns)
    body = " \\\\\n".join(" & ".join(_fmt(v, floatfmt) for v in row) for row in d.itertuples(index=False))
    align = "l" * len(cols)
    return (f"\\begin{{table}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{align}}}\\toprule\n"
            + " & ".join(str(c).replace('_', r'\_') for c in cols) + " \\\\\\midrule\n"
            + body + " \\\\\\bottomrule\n\\end{tabular}\\end{table}\n")


def _fmt(v, ff):
    if isinstance(v, float):
        return ("%.4g" % v) if np.isfinite(v) else "--"
    return str(v).replace("_", r"\_").replace("%", r"\%")


# --------------------------------------------------------------------------------- table builders
def T1(_):
    """nuisance × channel FAR (e04b) + residual R⁺ under drift (e05a)."""
    a = pd.read_csv(src("e04", "e04b_nuisance_far.csv")); a = a[["nuisance", "channel", "far_per_test", "band_lo", "band_hi", "in_band", "alarm_fraction"]].assign(source="e04b")
    b = pd.read_csv(src("e05a", "e05a_far.csv")); b = b[["condition", "detector", "far_per_cycle", "band_lo", "band_hi", "in_band", "alarm_fraction"]].rename(columns={"condition": "nuisance", "detector": "channel", "far_per_cycle": "far_per_test"}).assign(source="e05a-K200")
    return pd.concat([a, b], ignore_index=True)


def T2(_):
    """Σ-invariant (bilateral) vs single/unequal — raw (e04c) and residual (e13c) isotypic power."""
    a = pd.read_csv(src("e04", "e04c_isotypic_power.csv")).assign(source="e04c-raw")
    b = pd.read_csv(src("e13", "e13c_isotypic_power.csv")).assign(source="e13c-residual")
    return pd.concat([a, b], ignore_index=True)


def T3(_):
    return pd.read_csv(src("e07", "e07_table.csv"))


def T4(_):
    return pd.read_csv(src("e13", "e13a_min_detectable.csv"))


def T5(_):
    """e03 external Liu A1: four classes × detectors at equal FAR (ours + GRU)."""
    g = pd.read_csv(src("e03", "e03_gru_summary.csv")).assign(detector="GRU")
    # ours: per-class detection from the episodes table
    ep = pd.read_csv(src("e03", "e03_episodes.csv"))
    rows = []
    for cls, sub in ep.groupby("cls"):
        rows.append(dict(cls=cls, n=len(sub), det_rate=float(sub["detected"].mean()),
                         delay_median_s=float(sub.loc[sub["detected"], "delay_s"].median()) if sub["detected"].any() else np.nan,
                         loc_acc=float(sub["loc_pair_correct"].dropna().mean()) if sub["loc_pair_correct"].notna().any() else np.nan,
                         detector="Rminus_halfcycle"))
    ours = pd.DataFrame(rows)
    fa = pd.read_csv(src("e03", "e03_false_alarms.csv"))
    ours.attrs["fa"] = float(fa["fa_rate_per_gap"].iloc[0])
    return pd.concat([ours, g[["cls", "n", "det_rate", "delay_median_s", "loc_acc", "detector"]]], ignore_index=True)


def T7(_):
    return pd.read_csv(src("ladder", "ladder.csv"))[["tag", "kind", "n_train_per_leg", "seed", "val_rmse_all", "beta_hat_q95_mean", "delta_f_q95", "delta_f_q50", "delta_f_max"]]


def T8(_):
    v = pd.read_csv(src("e13", "e13a_variance_decomposition.csv")); s = pd.read_csv(src("e13", "e13a_snr.csv"))
    v = v.assign(snr_ratio=v["var_plus"] / v["var_minus"].replace(0, np.nan))
    return v


def T9(_):
    frames = []
    for tag, f in [("analytic_rows", "e13c_confusion_analytic_rows.csv"), ("equiv_rows", "e13c_confusion_equiv_rows.csv")]:
        p = src("e13", f)
        if p.exists():
            frames.append(pd.read_csv(p).assign(variant=tag))
    # also fold in the e09 three-channel confusion (accuracy summary) as the isolation-accuracy companion
    acc = src("e09", "e09_accuracy.csv")
    if acc.exists():
        a = pd.read_csv(acc); a.to_csv(TAB / "T9b_e09_isolation_accuracy.csv", index=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


TABLES = {
    "T1": ("nuisance FAR (per channel) with residual R+ under drift", T1),
    "T2": ("blind-spot: Sigma-invariant vs single/unequal isotypic power (raw & residual)", T2),
    "T3": ("baselines at unified FAR (det100/det20/delay per detector)", T3),
    "T4": ("minimal detectable magnitude per (fault,joint,detector)", T4),
    "T5": ("e03 external Liu-A1: per-class detection, delay, localisation (ours vs GRU)", T5),
    "T7": ("DeLaN ladders: val RMSE / beta / delta_f, plain vs equivariant", T7),
    "T8": ("variance decomposition Var(Pi- tau) vs Var(Pi- r) + SNR", T8),
    "T9": ("isolation confusion: analytic rows vs equivariant rows (+e09 accuracy)", T9),
}
# T6 (epsilon-budget/falsification) and T10 (M1 audit/manifest) are prose/theory tables — registered, not rebuilt here.
DOC_TABLES = {"T6": "theory/sections tab:degradation, tab:degradation2", "T10": "docs/protocol/m1_model_audit.md + legacy_aug_inventory.md"}


# --------------------------------------------------------------------------------- figure regenerations (CSV -> PDF)
def fig_F_tradeoff():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = pd.read_csv(src("e11", "e11_tradeoff.csv"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for det, sub in t.groupby("detector"):
        sub = sub.sort_values("arl0_measured"); ax.plot(sub["arl0_measured"], sub["delay_median_cycles"], "o-", label=det, ms=4)
    ax.set_xlabel("measured ARL$_0$ (cycles)"); ax.set_ylabel("detection delay (cycles)"); ax.legend(fontsize=7)
    ax.set_title("F4b sequential ARL$_0$–delay trade-off (e11)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "F4b_arl_delay_tradeoff.pdf"); plt.close(fig); return "F4b_arl_delay_tradeoff.pdf"


def fig_F_e10_signatures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = pd.read_csv(src("e10", "e10_signatures.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 3.0))
    ax[0].hist(s["cos_enc"], bins=12, color="C0", alpha=0.8); ax[0].axvline(s["cos_enc"].mean(), color="k", ls="--", lw=1)
    ax[0].set_title("encoder-bias signature\ncos($\\Delta z$, J$b$)", fontsize=8); ax[0].set_xlabel("cosine")
    ax[1].hist(s["benc_rec"], bins=12, color="C2", alpha=0.8); ax[1].axvline(0.05, color="r", ls="--", lw=1, label="true 0.05")
    ax[1].set_title("reconstructed encoder bias (rad)", fontsize=8); ax[1].set_xlabel("$\\hat b$ (rad)"); ax[1].legend(fontsize=7)
    fig.suptitle("F10b N2 estimator signatures (e10)", fontsize=9); fig.tight_layout()
    fig.savefig(FIG / "F10b_n2_signatures.pdf"); plt.close(fig); return "F10b_n2_signatures.pdf"


def fig_F_e08_lowsnr():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = pd.read_csv(src("e08", "e08_grid_power.csv"))
    sub = g[(g.fault == "actuator_gain") & (g.joint == "KFE")]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for ns_, m in sub.groupby("noise_scale"):
        m = m.sort_values("severity"); ax.plot(m["severity"], m["Rminus_res_an_hc_eproc_det100"], "o-", ms=3, label=f"noise×{ns_:g}")
    ax.set_xlabel("severity (1-$\\kappa$)"); ax.set_ylabel("det100 (analytic-residual R$^-$)"); ax.legend(fontsize=7, title="KFE gain")
    ax.set_title("F6b residual R$^-$ power vs SNR (e08)", fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "F6b_lowsnr_power.pdf"); plt.close(fig); return "F6b_lowsnr_power.pdf"


REGEN = {"F4b": fig_F_tradeoff, "F10b": fig_F_e10_signatures, "F6b": fig_F_e08_lowsnr}
# existing multi-panel figures composed in the manuscript — registered with their source PNG (paper = vector by hand)
REGISTER_FIG = {
    "F2": ("e04", "e04b_nuisance_far.png"), "F3": ("e04", "e04d_noise_sweep.png"),
    "F5": ("e13", "e13c_isotypic_isolation.png"), "F6": ("e13", "e13a_power_curves.png"),
    "F7": ("e13", "e13b_size_vs_defect.png"), "F8": ("e06", "e06iii_isolation_vs_dk.png"),
    "F9": ("e07", "e07_delay_pivot.csv"), "F10": ("e02", "e02a_money_kicked-perturbed.png"), "F11_e09": ("e09", "e09_confusion.png"),
    "F12_e11compl": ("e11", "e11_complementarity.png"),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args(); only = set(x.strip() for x in a.only.split(",") if x.strip())
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    cov = []; tex_parts = []
    # tables
    for tid, (cap, fn) in TABLES.items():
        if only and tid not in only:
            continue
        try:
            df = fn(None)
            if a.check:
                cov.append((tid, "table", "OK" if not df.empty else "EMPTY", cap)); continue
            df.to_csv(TAB / f"{tid}_{cap.split(':')[0].split('(')[0].strip().replace(' ', '_')[:40]}.csv", index=False)
            tex_parts.append(_tex(df, cap, f"tab:{tid}"))
            cov.append((tid, "table", f"generated ({df.shape[0]}x{df.shape[1]})", cap))
        except Exception as e:
            cov.append((tid, "table", f"FAILED: {type(e).__name__}: {e}", cap))
    for tid, s in DOC_TABLES.items():
        cov.append((tid, "table", "doc-derived (registered)", s))
    if not a.check and tex_parts:
        (OUT / "paper_tables.tex").write_text("% auto-generated by scripts/make_paper_figures.py\n" + "\n".join(tex_parts))
    # figures (regenerated)
    for fid, fn in REGEN.items():
        if only and fid not in only:
            continue
        try:
            name = "SKIP" if a.check else fn(); cov.append((fid, "figure", f"regenerated -> {name}" if not a.check else "OK", ""))
        except Exception as e:
            cov.append((fid, "figure", f"FAILED: {type(e).__name__}: {e}", ""))
    # figures (registered existing)
    for fid, (exp, fname) in REGISTER_FIG.items():
        p = src(exp, fname); cov.append((fid, "figure", f"registered {'OK' if p.exists() else 'MISSING'}: {p.relative_to(RES) if p.exists() else p}", ""))
    # coverage report
    lines = ["# Paper figure/table coverage (auto)\n", f"Generated into `{OUT}` by scripts/make_paper_figures.py.\n",
             "| id | kind | status | note |", "|----|------|--------|------|"]
    for i, k, s, n in cov:
        lines.append(f"| {i} | {k} | {s} | {n} |")
    rep = "\n".join(lines) + "\n"
    if a.check:
        print(rep)
    else:
        (OUT / "coverage.md").write_text(rep); print(rep); print(f"[make_paper_figures] wrote {OUT}")


if __name__ == "__main__":
    main()
