"""H0': asymmetry-CHANGE detection when the healthy loop is stably but measurably asymmetric.

Asymmetry functional nu(P) >= 0 through a fixed statistic class: the two-sample energy distance between
{Z_k} and {rho(sigma_*) Z_k} on standardized channels (nu = 0 iff the law is mirror-symmetric, in the
sense of the energy-distance metric on this class). Calibration window -> nu_0 and its (block-)bootstrap
sampling variability; monitoring window -> permutation test of nu_mon = nu_cal (H0': the asymmetry level
is unchanged) against nu_mon > nu_cal: cycles (or blocks) are re-assigned at random between the two
windows (exchangeable under a stationary asymmetric law), p = (1 + #{Delta* >= Delta}) / M with
Delta = nu_mon - nu_cal.
"""
from __future__ import annotations

import numpy as np

from .permutation import pooled_scale


def nu(Z: np.ndarray, rep, scale: np.ndarray | None = None) -> float:
    Z = np.asarray(Z, dtype=float)
    Zs = rep.apply("s", Z)
    sc = pooled_scale(Z, Zs) if scale is None else scale
    K = Z.shape[0]
    X = (Z / sc).reshape(K, -1); Y = (Zs / sc).reshape(K, -1)
    P = np.concatenate([X, Y]); sq = (P * P).sum(1)
    Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0.0))
    Daa = Dm[:K, :K]; Dbb = Dm[K:, K:]; Dab = Dm[:K, K:]
    return float(2 * Dab.mean() - Daa.sum() / (K * (K - 1)) - Dbb.sum() / (K * (K - 1)))


def calibrate(Z_cal: np.ndarray, rep, n_boot: int = 200, block_len: int = 1,
              rng: np.random.Generator | None = None) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    Z_cal = np.asarray(Z_cal, dtype=float)
    scale = pooled_scale(Z_cal, rep.apply("s", Z_cal))
    nu0 = nu(Z_cal, rep, scale)
    K = Z_cal.shape[0]; nb = int(np.ceil(K / block_len))
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, K - block_len + 1, size=nb) if K > block_len else np.zeros(nb, int)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:K]
        boots.append(nu(Z_cal[idx], rep, scale))
    boots = np.array(boots)
    return {"nu0": nu0, "nu0_boot_std": float(boots.std()), "nu0_boot_q05": float(np.quantile(boots, 0.05)),
            "nu0_boot_q95": float(np.quantile(boots, 0.95)), "scale": scale, "K_cal": K, "block_len": block_len}


def h0prime_test(Z_cal: np.ndarray, Z_mon: np.ndarray, rep, M: int = 512, block_len: int = 1,
                 rng: np.random.Generator | None = None, scale: np.ndarray | None = None) -> dict:
    """Permutation test of nu(mon) = nu(cal) vs nu(mon) > nu(cal); returns p, nu_cal, nu_mon, delta."""
    rng = np.random.default_rng() if rng is None else rng
    Zc = np.asarray(Z_cal, dtype=float); Zm = np.asarray(Z_mon, dtype=float)
    Kc, Km = Zc.shape[0], Zm.shape[0]
    Zall = np.concatenate([Zc, Zm]); Zs = rep.apply("s", Zall)
    sc = pooled_scale(Zall, Zs) if scale is None else scale
    K = Kc + Km
    X = (Zall / sc).reshape(K, -1); Y = (Zs / sc).reshape(K, -1)
    P = np.concatenate([X, Y]); sq = (P * P).sum(1)
    Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0.0))   # (2K, 2K); Y_k index K+k

    def nu_subset(idx):
        n = len(idx); a = np.asarray(idx); b = a + K
        return 2 * Dm[np.ix_(a, b)].mean() - Dm[np.ix_(a, a)].sum() / (n * (n - 1)) - Dm[np.ix_(b, b)].sum() / (n * (n - 1))
    obs = nu_subset(np.arange(Kc, K)) - nu_subset(np.arange(Kc))
    # block permutation of cycle labels between the windows
    nb = int(np.ceil(K / block_len)); blocks = [np.arange(i * block_len, min((i + 1) * block_len, K)) for i in range(nb)]
    n_cal_blocks = round(Kc / block_len)
    null = np.empty(M - 1)
    for m in range(M - 1):
        perm = rng.permutation(nb)
        cal_idx = np.concatenate([blocks[i] for i in perm[:n_cal_blocks]])
        mon_idx = np.concatenate([blocks[i] for i in perm[n_cal_blocks:]])
        null[m] = nu_subset(mon_idx) - nu_subset(cal_idx)
    p = (1.0 + np.sum(null >= obs)) / M
    return {"p": float(p), "delta": float(obs), "nu_cal": float(nu_subset(np.arange(Kc))), "nu_mon": float(nu_subset(np.arange(Kc, K)))}
