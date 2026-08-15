"""Mirror-pair statistics and the Hemerik–Goeman random-subset permutation test for H0.

Data: phase-registered cycles Z (K, d, N); the gait element sigma_* acts by rho(sigma_*) (groups.c2).

Statistics (both computed on per-channel standardized data; the standardization uses the pooled
{Z_k, rho Z_k} std per channel, which is invariant under cycle flips, so exactness is preserved):
  paired_energy   T = || (1/K) sum_k D_k ||_2^2 with D_k = Z_k - rho(sigma_*) Z_k   (L2 energy of the
                  mean paired difference — a sign-flip statistic; flipping cycle k maps D_k -> -D_k)
  energy_distance two-sample energy distance between A = {Z_k} and B = {rho Z_k}; flipping cycle k
                  swaps its two elements between A and B.

Test (Hemerik & Goeman, 2018, "Exact testing with random permutations", random-subset construction):
  the transformation group is the product G^K of per-cycle flips (block version: flips are constant on
  blocks of `block_len` consecutive cycles, i.e. the group G^{ceil(K/block_len)}); we draw M-1 elements
  uniformly at random from the group and ADD THE IDENTITY, giving a set of M elements which is
  exchangeable under H0 (the identity is not special because the M-1 draws are uniform); the p-value
  p = (1 + #{ T(g Z) >= T(Z) : g among the M-1 random draws }) / M
  satisfies P_H0(p <= alpha) <= alpha for every alpha (equality at alpha in {1/M, 2/M, ...}). Both the
  "+1" (the identity element counted in the numerator) and the "/M" (identity counted in the set) are
  required for exactness — dropping either yields an anti-conservative test; tests/test_permutation.py
  checks the uniformity numerically.
"""
from __future__ import annotations

import numpy as np

STATISTICS = ("paired_energy", "energy_distance")


def pooled_scale(Z: np.ndarray, Zs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-channel std over the flip-invariant pooled multiset {Z_k} u {rho Z_k}. Shape (d, 1)."""
    pooled = np.concatenate([Z, Zs], axis=0)                 # (2K, d, N)
    s = pooled.transpose(1, 0, 2).reshape(pooled.shape[1], -1).std(axis=1)
    return np.maximum(s, eps)[:, None]


class MirrorStatistics:
    """Precomputes what is needed to evaluate the statistics for many flip patterns cheaply."""

    def __init__(self, Z: np.ndarray, rep, standardize: bool = True):
        Z = np.asarray(Z, dtype=float)
        self.K = Z.shape[0]
        Zs = rep.apply("s", Z)
        scale = pooled_scale(Z, Zs) if standardize else np.ones((Z.shape[1], 1))
        self.X = (Z / scale).reshape(self.K, -1)             # (K, dN)
        self.Y = (Zs / scale).reshape(self.K, -1)
        D = self.X - self.Y
        self.G = D @ D.T                                    # Gram of paired differences
        self._Dm = None

    # -- paired energy ---------------------------------------------------------------------------
    def paired_energy(self, flips: np.ndarray) -> np.ndarray:
        """flips: (M, K) in {+1,-1}; returns (M,) values of ||mean_k flips_k D_k||^2."""
        f = np.asarray(flips, dtype=float)
        return np.einsum("mk,kl,ml->m", f, self.G, f) / (self.K ** 2)

    # -- energy distance --------------------------------------------------------------------------
    def _dist_matrix(self):
        if self._Dm is None:
            P = np.concatenate([self.X, self.Y], axis=0)      # (2K, dN)
            sq = (P * P).sum(axis=1)
            D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0.0)
            self._Dm = np.sqrt(D2)
        return self._Dm

    def energy_distance(self, flips: np.ndarray) -> np.ndarray:
        Dm = self._dist_matrix(); K = self.K
        f = np.asarray(flips)
        out = np.empty(f.shape[0])
        ar = np.arange(K)
        for m in range(f.shape[0]):
            a = np.where(f[m] > 0, ar, K + ar); b = np.where(f[m] > 0, K + ar, ar)
            Daa = Dm[np.ix_(a, a)]; Dbb = Dm[np.ix_(b, b)]; Dab = Dm[np.ix_(a, b)]
            out[m] = 2 * Dab.mean() - Daa.sum() / (K * (K - 1)) - Dbb.sum() / (K * (K - 1))
        return out

    def statistic(self, name: str, flips: np.ndarray) -> np.ndarray:
        if name == "paired_energy":
            return self.paired_energy(flips)
        if name == "energy_distance":
            return self.energy_distance(flips)
        raise ValueError(name)


def random_flips(K: int, M: int, rng: np.random.Generator, block_len: int = 1) -> np.ndarray:
    """(M, K) matrix of +-1 flips, constant on blocks of block_len consecutive cycles."""
    nb = int(np.ceil(K / block_len))
    fb = rng.integers(0, 2, size=(M, nb)) * 2 - 1
    return np.repeat(fb, block_len, axis=1)[:, :K]


def hg_permutation_test(Z: np.ndarray, rep, statistic: str = "paired_energy", M: int = 512,
                        rng: np.random.Generator | None = None, block_len: int = 1,
                        standardize: bool = True, return_null: bool = False):
    """Hemerik–Goeman random-subset test; returns p (and the observed statistic, null draws)."""
    rng = np.random.default_rng() if rng is None else rng
    ms = MirrorStatistics(Z, rep, standardize=standardize)
    K = ms.K
    obs = ms.statistic(statistic, np.ones((1, K)))[0]
    flips = random_flips(K, M - 1, rng, block_len)
    null = ms.statistic(statistic, flips)
    p = (1.0 + np.sum(null >= obs)) / M
    if return_null:
        return p, obs, null
    return p, obs


def hg_permutation_tests(Z: np.ndarray, rep, statistics=STATISTICS, M: int = 512,
                         rng: np.random.Generator | None = None, block_len: int = 1) -> dict:
    """Both statistics from ONE shared set of random flips (same group elements)."""
    rng = np.random.default_rng() if rng is None else rng
    ms = MirrorStatistics(Z, rep)
    flips = random_flips(ms.K, M - 1, rng, block_len)
    out = {}
    for s in statistics:
        obs = ms.statistic(s, np.ones((1, ms.K)))[0]
        null = ms.statistic(s, flips)
        out[s] = {"p": (1.0 + np.sum(null >= obs)) / M, "obs": float(obs)}
    return out
