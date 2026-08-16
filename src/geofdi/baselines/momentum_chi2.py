"""Classical model-based FDI baseline (Sprint 9 B1): De Luca–Mattone generalised-momentum observer + chi-square test.

This is the textbook method a reviewer will ask about (De Luca & Mattone 2003/2005; Haddadin, De Luca & Albu-Schaffer,
*Robot Collisions: A Survey*, T-RO 2017): run the generalised-momentum observer to get the joint residual
r(t) in R^12, whiten it with a covariance estimated on nominal data, and alarm when the quadratic form

    q(t) = r(t)^T Sigma^{-1} r(t)   ~   chi^2_k  (under the nominal hypothesis, if r were i.i.d. Gaussian)

exceeds a FIXED threshold chi^2_{k, 1-alpha}. It needs no fault data (like R-), but its false-alarm rate rests on a
distributional assumption that a legged robot violates: r is gait-periodic and serially correlated, so the nominal q is
neither chi^2 nor independent across samples. We therefore report the method BOTH ways:

  * `fixed`       — the classical chi^2_{k,1-alpha} threshold. Its *measured* nominal exceedance rate is reported; the
                    gap to alpha is the point of the comparison.
  * `far_matched` — the threshold re-calibrated on the run's own nominal data so that the probability of raising ANY
                    alarm within a monitoring HORIZON equals alpha. This is the comparison the unified baseline protocol
                    requires (docs/protocol/baseline_protocol.md): detection is "first exceedance inside the horizon", so
                    a per-sample rate alpha is not the same guarantee -- over a 100-cycle horizon a per-sample rate of
                    0.05 alarms with probability ~1. Bootstrapped exactly like the e-CUSUM threshold h.

    det = MomentumChi2(alpha=0.05, debounce=3)
    det.fit(r_cal)                      # (T_cal, k) nominal residual rows
    out = det.score(r_mon)              # dict: q, thresholds, per-sample alarms, first-alarm index for both modes
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class MomentumChi2:
    def __init__(self, alpha: float = 0.05, debounce: int = 3, diagonal: bool = False, ridge: float = 1e-9,
                 horizon: int | None = None, n_boot: int = 400, rng=None):
        self.alpha = float(alpha); self.debounce = int(debounce); self.diagonal = bool(diagonal); self.ridge = float(ridge)
        self.horizon = horizon; self.n_boot = int(n_boot); self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, r_cal: np.ndarray):
        r = np.asarray(r_cal, float); r = r[np.isfinite(r).all(axis=1)]
        self.k_ = r.shape[1]; self.mu_ = r.mean(axis=0)
        d = r - self.mu_
        if self.diagonal:
            self.Sinv_ = np.diag(1.0 / (d.var(axis=0) + self.ridge))
        else:
            S = np.cov(d, rowvar=False) + self.ridge * np.eye(self.k_)
            self.Sinv_ = np.linalg.inv(S)
        q_cal = np.einsum("ij,jk,ik->i", d, self.Sinv_, d)
        self.q_cal_ = q_cal
        self.thr_fixed_ = float(stats.chi2.ppf(1 - self.alpha, self.k_))
        # FAR-matched: horizon-calibrated (probability of ANY debounced alarm within `horizon` samples <= alpha),
        # bootstrapped from the nominal statistic; falls back to the per-sample quantile when no horizon is given.
        self.thr_far_ = (float(_horizon_threshold(q_cal, self.alpha, self.debounce, int(self.horizon), self.n_boot, self.rng))
                         if self.horizon else float(_debounced_quantile(q_cal, self.alpha, self.debounce)))
        self.cal_exceed_fixed_ = float(np.mean(q_cal > self.thr_fixed_))
        return self

    def score(self, r_mon: np.ndarray) -> dict:
        r = np.asarray(r_mon, float)
        d = np.nan_to_num(r - self.mu_)
        q = np.einsum("ij,jk,ik->i", d, self.Sinv_, d)
        out = {"q": q, "thr_fixed": self.thr_fixed_, "thr_far_matched": self.thr_far_,
               "cal_exceed_rate_at_fixed_thr": self.cal_exceed_fixed_, "dof": self.k_}
        for tag, thr in (("fixed", self.thr_fixed_), ("far_matched", self.thr_far_)):
            a = _debounce(q > thr, self.debounce)
            idx = np.flatnonzero(a)
            out[f"alarm_{tag}"] = a
            out[f"first_alarm_{tag}"] = int(idx[0]) if len(idx) else None
            out[f"alarm_rate_{tag}"] = float(a.mean())
        return out


def _debounce(flag: np.ndarray, m: int) -> np.ndarray:
    """True where `flag` has been continuously True for m samples (the standard anti-chatter rule)."""
    if m <= 1:
        return flag.astype(bool)
    f = flag.astype(int)
    c = np.convolve(f, np.ones(m, int), mode="full")[m - 1:len(f) + m - 1]
    return c >= m


def _debounced_quantile(q_cal: np.ndarray, alpha: float, debounce: int) -> float:
    """Smallest threshold whose debounced alarm rate on the calibration statistic is <= alpha."""
    cands = np.quantile(q_cal, np.linspace(0.50, 0.9999, 400))
    for thr in cands:
        if _debounce(q_cal > thr, debounce).mean() <= alpha:
            return float(thr)
    return float(cands[-1])


def _horizon_threshold(q_cal: np.ndarray, alpha: float, debounce: int, horizon: int, n_boot: int, rng) -> float:
    """Smallest threshold whose probability of raising ANY debounced alarm inside a `horizon`-sample window of nominal
    data is <= alpha. Windows are drawn as contiguous blocks of the calibration statistic (so serial correlation, the
    very thing that breaks the i.i.d. chi-square assumption, is preserved in the bootstrap)."""
    n = len(q_cal)
    if n <= horizon:
        starts = np.zeros(n_boot, int); horizon = max(1, n - 1)
    else:
        starts = rng.integers(0, n - horizon, size=n_boot)
    wins = np.stack([q_cal[s:s + horizon] for s in starts])
    cands = np.quantile(q_cal, np.linspace(0.50, 0.99999, 300))
    for thr in cands:
        hit = np.mean([_debounce(w > thr, debounce).any() for w in wins])
        if hit <= alpha:
            return float(thr)
    return float(cands[-1])
