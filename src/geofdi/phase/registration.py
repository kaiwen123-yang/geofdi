"""Phase registration: cut telemetry into gait cycles and resample each onto an N-point phase grid.

    Z, meta = register_cycles(df, channels, N=64)     # Z: (K, d, N), meta: cycle start times etc.

S1 uses the controller's ground-truth phase column `theta` (turns in [0,1)); an estimator (contact events /
phase oscillator) can be plugged in through `theta_col` or by writing its estimate into the frame — the
interface stays the same. Cycle boundaries are the wrap-arounds of theta; each complete cycle is
interpolated onto the grid theta_i = i/N (linear in theta). Incomplete first/last cycles are dropped and
`drop_first` warm-up cycles can be discarded. `write_cycles` stores a wide parquet (rows = cycle x phase
index) plus a manifest snapshot next to it.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def cycle_boundaries(theta: np.ndarray) -> np.ndarray:
    """Indices where a new cycle starts (theta wraps from ~1 to ~0), including 0 if theta[0] is small."""
    wraps = np.where(np.diff(theta) < -0.5)[0] + 1
    return wraps


def register_cycles(df: pd.DataFrame, channels: list[str], N: int = 64, theta_col: str = "theta",
                    drop_first: int = 2, drop_last: int = 0):
    theta = df[theta_col].to_numpy()
    X = df[list(channels)].to_numpy()
    t = df["t"].to_numpy() if "t" in df else np.arange(len(df), dtype=float)
    starts = cycle_boundaries(theta)
    grid = np.arange(N) / N
    Zs, t0s, idx = [], [], []
    for a, b in itertools.pairwise(starts):
        th = theta[a:b]
        if len(th) < 4 or th[0] > 0.1 or th[-1] < 0.9:
            continue          # incomplete cycle
        # make theta strictly increasing (it is, up to the wrap) and interpolate each channel
        Zk = np.empty((X.shape[1], N))
        for c in range(X.shape[1]):
            Zk[c] = np.interp(grid, th, X[a:b, c], left=X[a, c], right=X[b - 1, c])
        Zs.append(Zk); t0s.append(t[a]); idx.append(a)
    K = len(Zs)
    lo, hi = drop_first, K - drop_last
    Z = np.stack(Zs[lo:hi]) if hi > lo else np.empty((0, X.shape[1], N))
    meta = {"N": N, "channels": list(channels), "n_cycles": int(Z.shape[0]), "dropped_first": drop_first,
            "dropped_last": drop_last, "t_start": [float(v) for v in t0s[lo:hi]], "row_start": [int(v) for v in idx[lo:hi]]}
    return Z, meta


def write_cycles(out_dir: Path, Z: np.ndarray, meta: dict, manifest: dict) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    K, d, N = Z.shape
    cyc = np.repeat(np.arange(K), N); ph = np.tile(np.arange(N), K)
    data = {"cycle": cyc, "phase_idx": ph, "t_start": np.repeat(np.asarray(meta["t_start"]), N)}
    flat = Z.transpose(0, 2, 1).reshape(K * N, d)
    for j, name in enumerate(meta["channels"]):
        data[name] = flat[:, j]
    pd.DataFrame(data).to_parquet(out_dir / "cycles.parquet", index=False)
    (out_dir / "cycles_manifest.yaml").write_text(yaml.safe_dump({"registration": {k: v for k, v in meta.items() if k not in ("t_start", "row_start")},
                                                                    "manifest": manifest}, sort_keys=False))


def read_cycles(run_dir: Path):
    run_dir = Path(run_dir)
    df = pd.read_parquet(run_dir / "cycles.parquet")
    meta = yaml.safe_load((run_dir / "cycles_manifest.yaml").read_text())
    chans = meta["registration"]["channels"]; N = meta["registration"]["N"]
    K = df["cycle"].nunique()
    Z = df[chans].to_numpy().reshape(K, N, len(chans)).transpose(0, 2, 1)
    return Z, meta
