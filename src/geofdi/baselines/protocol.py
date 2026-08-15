"""Unified alarm protocol for all detectors (ours and baselines) — docs/protocol/baseline_protocol.md.

Every detector produces a per-cycle SCORE (larger = more anomalous). On the SAME nominal calibration cycles of a run
(K_cal cycles after warm-up) the score is turned into a one-sided conformal p-value, p_k = (1 + #{cal >= s_k}) / (n+1),
and the monitored cycles are aggregated by the e-process E_t = prod e(p_k), e = p^{-1/2}/2, alarm at E >= 1/alpha
(Ville: P(false alarm ever) <= alpha under exchangeability). Delay = cycles from onset to the alarm; det100 = alarm
within 100 monitored cycles. Because the calibration set, the p-value map and the alarm rule are identical, differences
between detectors reflect only their scores. A naive rule (fixed 95 % quantile threshold, first exceedance) is
reported alongside with its actual per-cycle FAR, to make the FAR-control point explicit.

Ours (R-, R+) use only nominal data (calibration); the GRU is trained on FAULT rollouts (its protocol); the AE and
the Mahalanobis gate are fitted on nominal rollouts (a separate nominal training set) and calibrated per run like all.
"""
from __future__ import annotations

import numpy as np

from ..detect.monitors import conformal_pvalues, eprocess_alarm


def cycle_scores_from_windows(win_scores: np.ndarray, win_cycle: np.ndarray, K: int) -> np.ndarray:
    """Average window scores per cycle index (0..K-1); cycles without windows get nan."""
    out = np.full(K, np.nan)
    for k in range(K):
        m = win_cycle == k
        if m.any():
            out[k] = float(np.mean(win_scores[m]))
    return out


def alarm_from_scores(scores: np.ndarray, K_cal: int, alpha: float = 0.05):
    """scores: per-cycle scores (K_cal calibration + monitored). Returns dict with e-process delay (cycles, None if
    none), naive-threshold delay and the naive per-cycle exceedance rate on the monitored part."""
    s = np.asarray(scores, dtype=float); cal = s[:K_cal]; mon = s[K_cal:]
    ok = np.isfinite(cal); cal = cal[ok]
    p = conformal_pvalues(cal, np.where(np.isfinite(mon), mon, np.nanmedian(cal)))
    E, al = eprocess_alarm(p, alpha, start=0)
    thr = np.quantile(cal, 1 - alpha); exc = mon > thr
    naive = int(np.argmax(exc)) + 1 if exc.any() else None
    return {"delay_eproc": None if al is None else al + 1, "delay_naive": naive, "naive_rate": float(np.mean(exc)), "p": p}
