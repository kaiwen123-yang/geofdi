"""Mahalanobis gate on per-cycle summary features of the nominal data element (nominal only).

Features per cycle: mean and std over phase of every Z channel (2 x d). A Ledoit-Wolf shrinkage covariance is fitted
on nominal training cycles; the score is the squared Mahalanobis distance of a cycle's feature vector.
"""
from __future__ import annotations

import numpy as np


def cycle_features(Z: np.ndarray) -> np.ndarray:
    """(K, d, N) -> (K, 2d)."""
    return np.concatenate([Z.mean(axis=2), Z.std(axis=2)], axis=1)


class MahalanobisGate:
    def __init__(self):
        self.mu = None; self.P = None

    def fit(self, F: np.ndarray):
        from sklearn.covariance import LedoitWolf
        self.mu = F.mean(0); lw = LedoitWolf().fit(F - self.mu); self.P = lw.precision_
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        d = F - self.mu
        return np.einsum("ki,ij,kj->k", d, self.P, d)
