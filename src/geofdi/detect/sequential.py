"""Sequential layer redesign (Sprint 7 Block E1): half-cycle mirror scores + per-element conformal p-values + three
sequential aggregators behind one interface (e-process, e-CUSUM, conformal-CUSUM).

Why. The Sprint-2/4 R^- layer tested 5-cycle windows with the flip test (16 sign patterns -> p >= 1/16, e <= 2 per
window: alarm floor 25 cycles for the plain e-process, ~10 for the calibrated e-CUSUM). Target: R^- delay <= 2 cycles.
Two changes: (i) HALF-CYCLE elements from the unfolded definition — under H0 the half-cycle H_j has the law of the
mirror image of the previous half-cycle, Law(H_j) = Law(rho_g H_{j-1}) (Part 0 Remark rem:wrap), so the per-half-cycle
mirror score s_j = || (H_j - rho_g H_{j-1}) / scale ||^2 is a mirror-asymmetry statistic available twice per cycle;
(ii) a CALIBRATION SET of >= 400 nominal elements turns each score into a conformal p-value with p_min = 1/401, i.e. a
per-element e-value up to 0.5 * sqrt(401) ~ 10 — two extreme half-cycles reach 1/alpha = 20. The conformal p is exact per
element under exchangeability of the scores; the products/sums below are anytime-valid under independence and are
checked empirically on nominal streams (ARL / false-alarm-within-horizon).

    H = half_cycles(Z)                                  # (2K, d, N/2) from registered cycles (K, d, N)
    s = mirror_scores(H, rep, scale)                    # (2K-1,) half-cycle mirror scores
    det = EProcess(alpha) | ECusum(alpha, h) | ConformalCusum(alpha, h); det.calibrate(s_cal); det.run(s_mon) -> alarm index
The same interface serves any per-element score (R+ residual magnitude, Mahalanobis, GRU 1 - min eta_hat).
"""
from __future__ import annotations

import numpy as np

from .evalue import p_to_e
from .monitors import conformal_pvalues
from .permutation import pooled_scale


# ------------------------------------------------------------------------------ half-cycle mirror scores
def half_cycles(Z: np.ndarray) -> np.ndarray:
    """(K, d, N) registered cycles -> (2K, d, N//2) half-cycles in time order (A_1, B_1, A_2, B_2, ...)."""
    K, d, N = Z.shape; h = N // 2
    H = np.empty((2 * K, d, h))
    H[0::2] = Z[:, :, :h]; H[1::2] = Z[:, :, h:2 * h]
    return H


def mirror_scores(H: np.ndarray, rep, scale: np.ndarray | None = None) -> np.ndarray:
    """s_j = mean over (channels, samples) of ((H_j - rho_g H_{j-1}) / scale)^2 for j = 1..2K-1; rho_g = mirror only
    (the half-period shift is already the step to the previous half-cycle). `scale` (d,1): per-channel std from the
    calibration data (pooled over the flip-invariant multiset); default = pooled scale of H itself."""
    Hs = rep.mirror_only(H)                                     # rho_g applied to each half-cycle
    if scale is None:
        scale = pooled_scale(H, Hs)
    D = (H[1:] - Hs[:-1]) / scale
    return (D ** 2).mean(axis=(1, 2))


def calibration_scale(H_cal: np.ndarray, rep) -> np.ndarray:
    return pooled_scale(H_cal, rep.mirror_only(H_cal))


# ------------------------------------------------------------------------------ sequential aggregators
class SequentialDetector:
    """calibrate(scores_cal) -> conformal p per monitored score -> aggregator state; run(scores) returns
    (state array, first alarm index or None)."""

    name = "base"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha; self.cal = None

    def calibrate(self, scores_cal):
        self.cal = np.sort(np.asarray(scores_cal, dtype=float)); return self

    def pvalues(self, scores):
        return conformal_pvalues(self.cal, np.asarray(scores, dtype=float))

    def run(self, scores):
        raise NotImplementedError


class EProcess(SequentialDetector):
    """E_t = prod e(p_k), e = p^{-1/2}/2; alarm at E_t >= 1/alpha (Ville)."""
    name = "eprocess"

    def run(self, scores):
        p = self.pvalues(scores); E = np.cumprod(p_to_e(p)); hits = np.where(E >= 1.0 / self.alpha)[0]
        return E, (int(hits[0]) if len(hits) else None)


class ECusum(SequentialDetector):
    """S_t = max(0, S_{t-1} + log e(p_t)); alarm at S_t >= h. h is calibrated for a target ARL0 (or FAR over a horizon) on
    nominal streams by block bootstrap of calibration scores (see calibrate_threshold)."""
    name = "ecusum"

    def __init__(self, alpha: float = 0.05, h: float | None = None):
        super().__init__(alpha); self.h = h

    def run(self, scores):
        p = self.pvalues(scores); le = np.log(p_to_e(p)); S = np.empty(len(le)); s = 0.0
        for i, v in enumerate(le):
            s = max(0.0, s + v); S[i] = s
        hits = np.where(S >= self.h)[0]
        return S, (int(hits[0]) if len(hits) else None)


class ConformalCusum(SequentialDetector):
    """CUSUM on the conformal p-values themselves with the likelihood-ratio increment of a level shift of the p-law
    (Vovk-style conformal change detection): S_t = max(0, S_{t-1} + log(f(p_t))), f(p) = kappa p^(kappa-1) (the same
    calibrator family; kappa = 0.5 -> identical increment to e-CUSUM but with a threshold set for a target ARL0 through the
    conformal-martingale identity), alarm at S_t >= h. Kept as a separate object so that ARL targets can be set the two
    ways (bootstrap vs Ville-type bound h = log(ARL0 * alpha ...)); the increment differs from ECusum only through
    kappa (default 0.25: heavier weight on very small p, faster for large shifts)."""
    name = "conformal_cusum"

    def __init__(self, alpha: float = 0.05, h: float | None = None, kappa: float = 0.25):
        super().__init__(alpha); self.h = h; self.kappa = kappa

    def run(self, scores):
        p = self.pvalues(scores); le = np.log(p_to_e(p, kind="kappa", kappa=self.kappa)); S = np.empty(len(le)); s = 0.0
        for i, v in enumerate(le):
            s = max(0.0, s + v); S[i] = s
        hits = np.where(S >= self.h)[0]
        return S, (int(hits[0]) if len(hits) else None)


def calibrate_threshold(det: SequentialDetector, nominal_score_runs, horizon: int, far: float = 0.05, n_boot: int = 2000,
                        rng: np.random.Generator | None = None) -> float:
    """h such that P(max_{t <= horizon} S_t >= h) ~= far on nominal streams built by block bootstrap from the given
    nominal score runs (each run: scores NOT used for the conformal calibration set — held-out nominal segments)."""
    rng = np.random.default_rng() if rng is None else rng
    runs = [np.asarray(r, dtype=float) for r in nominal_score_runs if len(r) > 0]
    maxes = []
    hh = det.h; det.h = np.inf
    for _ in range(n_boot):
        seq = []
        while len(seq) < horizon:
            r = runs[rng.integers(len(runs))]; L = min(len(r), horizon - len(seq)); a = rng.integers(0, len(r) - L + 1); seq.extend(r[a:a + L])
        S, _ = det.run(np.array(seq)); maxes.append(S.max())
    det.h = hh
    return float(np.quantile(maxes, 1 - far))


def make_detector(kind: str, alpha: float = 0.05, h: float | None = None) -> SequentialDetector:
    if kind == "eprocess":
        return EProcess(alpha)
    if kind == "ecusum":
        return ECusum(alpha, h)
    if kind == "conformal_cusum":
        return ConformalCusum(alpha, h)
    raise ValueError(kind)
