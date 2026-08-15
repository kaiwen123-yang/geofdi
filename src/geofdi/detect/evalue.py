"""p-to-e calibration and the multiplicative e-process (test martingale) for anytime-valid monitoring.

Calibrator (default): e = p^{-1/2} / 2. It is a valid calibrator because for p ~ Uniform(0,1),
E[e] = ∫_0^1 p^{-1/2}/2 dp = 1 (the family kappa * p^{kappa-1}, 0 < kappa < 1, at kappa = 1/2;
Vovk & Wang 2021, "E-values: calibration, combination and applications"). Since valid p-values are
super-uniform under H0, E[e] <= 1 for every window; the running product E_t = prod_{s<=t} e_s over
independent (or sequentially valid) windows is a nonnegative supermartingale with E_0 = 1, so Ville's
inequality gives P_H0(sup_t E_t >= 1/alpha) <= alpha — the anytime-valid alarm rule is E_t >= 1/alpha.
"""
from __future__ import annotations

import numpy as np


def p_to_e(p, kind: str = "sqrt", kappa: float = 0.5):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    if kind == "sqrt":
        return 0.5 * p ** (-0.5)
    if kind == "kappa":
        if not 0 < kappa < 1:
            raise ValueError("kappa must be in (0,1)")
        return kappa * p ** (kappa - 1)
    raise ValueError(kind)


def eprocess(pvals, alpha: float = 0.05, kind: str = "sqrt", kappa: float = 0.5):
    """Running product of window e-values; returns (E_t array, first alarm index or None)."""
    e = p_to_e(pvals, kind, kappa)
    E = np.cumprod(e)
    hits = np.where(E >= 1.0 / alpha)[0]
    return E, (int(hits[0]) if len(hits) else None)


def ville_frequency(alarm_indices) -> float:
    """Empirical P(sup_t E_t >= 1/alpha) across independent runs (None = never alarmed)."""
    a = list(alarm_indices)
    return float(np.mean([x is not None for x in a])) if a else float("nan")


def average_run_length(alarm_indices, horizon: int) -> dict:
    """Empirical ARL in windows: censored mean of min(alarm+1, horizon) plus the uncensored-only mean."""
    a = list(alarm_indices)
    lengths = np.array([horizon if x is None else x + 1 for x in a], dtype=float)
    uncens = np.array([x + 1 for x in a if x is not None], dtype=float)
    return {"arl_censored_mean": float(lengths.mean()) if len(lengths) else float("nan"),
            "arl_uncensored_mean": float(uncens.mean()) if len(uncens) else float("nan"),
            "n_runs": len(a), "n_alarms": len(uncens), "horizon": horizon}
