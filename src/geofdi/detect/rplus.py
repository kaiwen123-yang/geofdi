"""R+ (magnitude) channels: per-cycle mirror-invariant scores -> conformal p (calibrated on nominal cycles only) ->
e-process / e-CUSUM alarms (as in S2).

- rplus_track : tracking-error magnitude sqrt(mean (q - q_ref)^2) per cycle (S2 baseline; sim-only, needs q_ref)
- rplus_resid : generalized-momentum residual magnitude — the residual r_j(t) (12 joints, from the momentum observer
                driven by the COMMANDED torque) is phase-registered per cycle (N bins), and the score is the phase-binned
                L2 energy  s_k = sqrt( mean_{bins, joints} rbar_{k,j}(theta)^2 )  (per leg optional for isolation).
- rplus_delan : same score with the residual r = tau_cmd - DeLaN(q, dq, ddq) (learned nominal model, Block L1).

Nothing here is trained: the residual generators are given (analytic or learned nominal model); the detector is a
conformal p-value against nominal calibration cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..phase.registration import register_cycles
from ..sim.telemetry import JOINTS, LEGS
from .monitors import conformal_pvalues, eprocess_alarm, tracking_scores  # noqa: F401  (re-exported for callers)

RES_COLS = [f"res_{l}_{j}" for l in LEGS for j in JOINTS]


def residual_frame(df: pd.DataFrame, r_joint: np.ndarray) -> pd.DataFrame:
    """Attach the 12 joint residuals (T, 12) to the telemetry frame as res_<leg>_<joint> columns."""
    out = df.copy()
    for i, c in enumerate(RES_COLS):
        out[c] = r_joint[:, i]
    return out


def registered_residuals(df: pd.DataFrame, r_joint: np.ndarray, N: int = 64, drop_first: int = 10):
    """Phase-registered residual cycles (K, 12, N) + meta (t_start etc.)."""
    dfr = residual_frame(df, r_joint)
    Zr, meta = register_cycles(dfr, RES_COLS, N=N, drop_first=drop_first)
    return Zr, meta


def residual_scores(Zr: np.ndarray, per_leg: bool = False, phase_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-cycle magnitude scores from registered residuals (K, 12, N): sqrt(mean over joints & phase of r^2)
    (optionally restricted to a phase mask), per leg (K, 4) or summed over legs (K,)."""
    K = Zr.shape[0]; s = np.zeros((K, 4))
    for li in range(4):
        blk = Zr[:, 3 * li:3 * li + 3, :]
        if phase_mask is not None:
            blk = blk[:, :, phase_mask]
        s[:, li] = np.sqrt((blk ** 2).mean(axis=(1, 2)))
    return s if per_leg else s.sum(axis=1)


def rplus_pvalues(scores_cal: np.ndarray, scores_mon: np.ndarray) -> np.ndarray:
    """Conformal one-sided p per monitored cycle against the calibration scores."""
    return conformal_pvalues(scores_cal, scores_mon)
