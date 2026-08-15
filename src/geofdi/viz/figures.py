"""Figure factory (matplotlib, Agg): p-value QQ plots, size tables, e-process trajectories, type-I curves."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DPI = 130          # 1600 px long edge at figsize width <= 12.3 in


def qq_pvalues(pvals_by_stat: dict, out_png, title: str = "", level: float = 0.95):
    n_stats = len(pvals_by_stat)
    fig, axes = plt.subplots(1, n_stats, figsize=(5.2 * n_stats, 5.0), squeeze=False)
    for ax, (name, p) in zip(axes[0], pvals_by_stat.items()):
        p = np.sort(np.asarray(p)); R = len(p)
        i = np.arange(1, R + 1)
        u = i / (R + 1)
        lo = stats.beta.ppf((1 - level) / 2, i, R - i + 1); hi = stats.beta.ppf(1 - (1 - level) / 2, i, R - i + 1)
        ax.fill_between(u, lo, hi, color="0.85", label=f"{int(level*100)}% band (order-statistic)")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(u, p, ".", ms=4, color="C0")
        ks = stats.kstest(p, "uniform").pvalue
        ax.set_title(f"{name}: KS p = {ks:.3f} (R={R})", fontsize=10)
        ax.set_xlabel("uniform quantile"); ax.set_ylabel("sorted p-value"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def size_bars(table_rows: list[dict], out_png, title: str = ""):
    """table_rows: dicts with keys stat, alpha, size, ci_lo, ci_hi, band_lo, band_hi, label(optional)."""
    labels = sorted({(r.get("label", ""), r["stat"]) for r in table_rows})
    alphas = sorted({r["alpha"] for r in table_rows})
    fig, ax = plt.subplots(figsize=(1.6 + 2.2 * len(alphas) * max(1, len(labels) / 2), 4.5))
    w = 0.8 / max(1, len(labels))
    for j, (lab, st) in enumerate(labels):
        xs, ys, err_lo, err_hi = [], [], [], []
        for i, a in enumerate(alphas):
            rr = [r for r in table_rows if r.get("label", "") == lab and r["stat"] == st and r["alpha"] == a]
            if not rr:
                continue
            r = rr[0]; xs.append(i + j * w); ys.append(r["size"]); err_lo.append(r["size"] - r["ci_lo"]); err_hi.append(r["ci_hi"] - r["size"])
        ax.bar(xs, ys, width=w, yerr=[err_lo, err_hi], capsize=3, label=f"{lab} {st}".strip())
    for i, a in enumerate(alphas):
        ax.hlines(a, i - 0.1, i + 0.8, colors="k", linestyles="--", lw=1)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(alphas))]); ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylabel("empirical size (95% CP CI)"); ax.legend(fontsize=7, ncol=2)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def typeI_vs_m(curves: dict, bounds: dict, out_png, alpha: float, title: str = ""):
    """curves: {label: (m_list, rate_list, lo_list, hi_list)}; bounds: {label: (m_list, bound_list)}."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, (ms, r, lo, hi) in curves.items():
        ax.errorbar(ms, r, yerr=[np.array(r) - np.array(lo), np.array(hi) - np.array(r)], marker="o", capsize=3, label=k)
    for k, (ms, b) in bounds.items():
        ax.plot(ms, np.minimum(b, 1.05), ":", lw=1.5, label=k)
    ax.axhline(alpha, color="k", ls="--", lw=1, label=f"α={alpha}")
    ax.set_xscale("log"); ax.set_ylim(0, 1.08); ax.set_xlabel("monitoring window m (cycles)"); ax.set_ylabel("rejection rate")
    ax.legend(fontsize=7); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def eprocess_trajectories(E_list: list, alpha: float, out_png, title: str = "", window_cycles: int | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for E in E_list:
        ax.plot(np.arange(1, len(E) + 1), E, lw=0.8, alpha=0.7)
    ax.axhline(1 / alpha, color="r", ls="--", label=f"1/α = {1/alpha:.0f}")
    ax.set_yscale("log"); ax.set_xlabel("window index" + (f" ({window_cycles} cycles each)" if window_cycles else "")); ax.set_ylabel("running e-value")
    ax.legend(); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)


def h0prime_timeline(p_windows: list, change_idx: int | None, alpha: float, out_png, title: str = ""):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    p = np.asarray(p_windows)
    ax.plot(np.arange(len(p)), p, "o-", ms=3, lw=0.8)
    ax.axhline(alpha, color="r", ls="--", lw=1, label=f"α={alpha}")
    if change_idx is not None:
        ax.axvline(change_idx - 0.5, color="k", ls=":", label="δ doubled")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.1); ax.set_xlabel("monitoring window"); ax.set_ylabel("H0' p-value")
    ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=DPI); plt.close(fig)
