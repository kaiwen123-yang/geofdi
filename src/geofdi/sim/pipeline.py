"""Rollout -> phase registration -> cycles, with a multiprocessing map over replicates (deterministic per seed).

    Z, meta, man = simulate_cycles(sim_cfg_dict, n_cycles=60, N=64)
    results = pmap(fn, list_of_args, n_workers)
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..phase.registration import register_cycles
from .env import SimConfig, rollout
from .telemetry import z_channel_names


def simulate_cycles(sim_cfg: dict, n_cycles: int, N: int = 64, drop_first: int = 2, duration_pad_cycles: int = 2):
    """Run one rollout long enough for `n_cycles` complete cycles after `drop_first` warm-up cycles."""
    cfg = SimConfig(**sim_cfg)
    period = float(cfg.controller.get("period_s", 0.5))
    if not cfg.duration_s or cfg.duration_s <= 0:
        cfg.duration_s = (n_cycles + drop_first + duration_pad_cycles) * period
    df, man = rollout(cfg)
    Z, meta = register_cycles(df, z_channel_names(man), N=N, drop_first=drop_first)
    if Z.shape[0] > n_cycles:
        Z = Z[:n_cycles]; meta["t_start"] = meta["t_start"][:n_cycles]; meta["n_cycles"] = n_cycles
    return Z, meta, man, df


def _call(args):
    fn, a = args
    return fn(*a)


def pmap(fn, arglist, n_workers: int | None = None, chunksize: int = 1):
    """Order-preserving parallel map (fork); n_workers=None -> min(len, cpu_count-2)."""
    n = len(arglist)
    if n == 0:
        return []
    if n_workers is None:
        n_workers = max(1, min(n, (os.cpu_count() or 2) - 2))
    if n_workers <= 1:
        return [fn(*a) for a in arglist]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        return pool.map(_call, [(fn, a) for a in arglist], chunksize=chunksize)


def binom_ci(k: int, n: int, level: float = 0.95):
    """Clopper–Pearson interval for a proportion."""
    from scipy import stats
    if n == 0:
        return (np.nan, np.nan)
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def nominal_band(alpha: float, n: int, level: float = 0.95):
    """Binomial(n, alpha) acceptance band for the empirical size (proportion scale)."""
    from scipy import stats
    a = 1 - level
    return float(stats.binom.ppf(a / 2, n, alpha) / n), float(stats.binom.ppf(1 - a / 2, n, alpha) / n)
