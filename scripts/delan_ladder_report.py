#!/usr/bin/env python3
"""Block Q report: plain vs equivariant DeLaN ladders — validation RMSE, beta_hat and the equivariance defect
delta_f^(0.95) vs training-sample size (from the report.json files under $GEOFDI_DATA_ROOT/models/delan*).

    python scripts/delan_ladder_report.py [--run-id ID]
    -> results/delan_ladder/<run_id>/{ladder.csv, delta_f_ladder.png, rmse_ladder.png, seed_replicates.csv}
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
LEGS = ("LF", "RF", "LH", "RH")

LADDER = [("full", "equiv_full", 400000), ("n50k", "equiv_n50k", 50000), ("n10k", "equiv_n10k", 10000), ("n2k", "equiv_n2k", 2000)]
EXTRA = [("delan", "equiv1_full", "equivariant, 1 template (ablation)"), ("delan_weld", "weld_plain_v1", "welded trunk, plain"),
         ("delan_weld", "weld_equiv_v1", "welded trunk, equivariant"), ("delan_weld", "weld_full", "welded trunk, plain (e06 original)")]
SEEDREP = [("n2k", "equiv_n2k", 2000), ("n10k", "equiv_n10k", 10000)]


def _load(sub, tag):
    p = DATA_ROOT / "models" / sub / tag / "report.json"
    return json.loads(p.read_text()) if p.exists() else None


def _row(sub, tag, r, n_train=None, kind=None):
    legs = r["legs"]
    rmse_leg = {l: float(np.sqrt(np.mean(np.square(legs[l]["final_val_rmse_per_joint"])))) for l in LEGS}
    rmse_all = float(np.sqrt(np.mean([v ** 2 for v in rmse_leg.values()])))
    beta = float(np.mean([legs[l]["beta_hat"]["global_q"] for l in LEGS]))
    de = r.get("defect", {})
    return {"model_dir": sub, "tag": tag, "kind": kind or ("equivariant" if r.get("equivariant") else "plain"),
            "n_templates": r.get("n_templates", 4), "n_train_per_leg": n_train if n_train is not None else r.get("n_train_per_leg"),
            "seed": r.get("seed", 0), "epochs": r.get("epochs"), "val_rmse_all": rmse_all, **{f"val_rmse_{l}": rmse_leg[l] for l in LEGS},
            "beta_hat_q95_mean": beta, "delta_f_q95": de.get("q95", np.nan), "delta_f_q50": de.get("q50", np.nan),
            "delta_f_max": de.get("max", np.nan), "delta_f_rms": de.get("rms", np.nan), "train_seconds": r.get("total_seconds")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    out = REPO / "results" / "delan_ladder" / args.run_id; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for plain, equiv, n in LADDER:
        for tag in (plain, equiv):
            r = _load("delan", tag)
            if r is not None:
                rows.append(_row("delan", tag, r, n))
    for sub, tag, kind in EXTRA:
        r = _load(sub, tag)
        if r is not None:
            rows.append(_row(sub, tag, r, kind=kind))
    tab = pd.DataFrame(rows); tab.to_csv(out / "ladder.csv", index=False)
    # seed replicates of the small cells
    srows = []
    for plain, equiv, n in SEEDREP:
        for s in (0, 1, 2):
            for base, kind in ((plain, "plain"), (equiv, "equivariant")):
                tag = base if s == 0 else f"{base}_s{s}"
                r = _load("delan", tag)
                if r is not None:
                    srows.append(_row("delan", tag, r, n, kind))
    srep = pd.DataFrame(srows)
    if len(srep):
        srep.to_csv(out / "seed_replicates.csv", index=False)
        agg = srep.groupby(["n_train_per_leg", "kind"]).val_rmse_all.agg(["mean", "std", "min", "max", "count"]).reset_index()
        agg.to_csv(out / "seed_replicates_summary.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lad = tab[tab.tag.isin([t for p, e, _ in LADDER for t in (p, e)])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for kind, mk, c in (("plain", "o-", "C3"), ("equivariant", "s-", "C0")):
        sub = lad[lad.kind == kind].sort_values("n_train_per_leg")
        y = sub.delta_f_q95.to_numpy(); y_plot = np.where(y > 0, y, np.nan)
        axes[0].plot(sub.n_train_per_leg, y_plot, mk, color=c, label=f"{kind} DeLaN: δ_f^(0.95)")
        if kind == "equivariant":
            axes[0].plot(sub.n_train_per_leg, np.full(len(sub), 1e-3), "s--", color=c, alpha=0.6, label="equivariant: δ_f = 0 exactly (plotted at 1e-3)")
        y2 = np.where(sub.delta_f_q50.to_numpy() > 0, sub.delta_f_q50.to_numpy(), np.nan)
        if kind == "plain":
            axes[0].plot(sub.n_train_per_leg, y2, "o:", color=c, alpha=0.7, label="plain: δ_f^(0.5)")
        axes[1].plot(sub.n_train_per_leg, sub.val_rmse_all, mk, color=c, label=f"{kind}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("training samples per leg"); axes[0].set_ylabel("equivariance defect [N m] (nominal validation)")
    axes[0].set_title("δ_f vs sample size: plain per-leg nets vs mirror weight sharing", fontsize=9); axes[0].legend(fontsize=6); axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("training samples per leg"); axes[1].set_ylabel("validation RMSE, all legs/joints [N m]")
    axes[1].set_title("validation RMSE (same rollout split)", fontsize=9); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, which="both")
    if len(srep):
        for kind, c in (("plain", "C3"), ("equivariant", "C0")):
            s2 = srep[srep.kind == kind]
            axes[1].scatter(s2.n_train_per_leg * (1.08 if kind == "plain" else 0.92), s2.val_rmse_all, marker="x", color=c, s=25, alpha=0.7, label=None)
        axes[1].text(0.02, 0.03, "x: seed replicates (seeds 0,1,2) of the 2k / 10k cells", transform=axes[1].transAxes, fontsize=6)
    fig.tight_layout(); fig.savefig(out / "delta_f_ladder.png", dpi=150); plt.close(fig)
    with open(out / "summary.txt", "w") as fh:
        for _, r in tab.iterrows():
            fh.write(f"{r.model_dir}/{r.tag:14s} {r.kind:32s} n={r.n_train_per_leg!s:>7} rmse_all={r.val_rmse_all:.3f} beta={r.beta_hat_q95_mean:.3f} "
                     f"delta_q95={r.delta_f_q95:.4f} delta_q50={r.delta_f_q50:.4f} delta_max={r.delta_f_max:.3f} secs={r.train_seconds}\n")
        if len(srep):
            fh.write("\nseed replicates (val RMSE all legs):\n" + agg.to_string() + "\n")
    print((out / "summary.txt").read_text())
    print(f"[ladder] wrote {out}")


if __name__ == "__main__":
    main()
