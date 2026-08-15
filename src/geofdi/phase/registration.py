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
    # unwrapped phase over the whole record (cycle index + theta): each cycle is interpolated on the GLOBAL (phi, X)
    # record, so that the grid point at the cycle start uses the neighbouring samples across the boundary instead of a
    # clamped edge value. With a controller-clock phase whose sample rows hit theta = 0 exactly this is identical to the
    # per-cycle interpolation; with an estimated phase (fractional offset) the clamped edge produced a systematic
    # one-grid-point asymmetry between the two half-cycles that the flip test detected (Sprint 7 W3 finding).
    cyc = np.zeros(len(theta), dtype=int)
    for a in starts:
        cyc[a:] += 1
    phi = theta + cyc
    # enforce strict monotonicity for np.interp (repeated theta values within a cycle are nudged)
    phi_m = np.maximum.accumulate(phi + np.arange(len(phi)) * 1e-12)
    Zs, t0s, idx = [], [], []
    for a, b in itertools.pairwise(starts):
        th = theta[a:b]
        if len(th) < 4 or th[0] > 0.1 or th[-1] < 0.9:
            continue          # incomplete cycle
        k = cyc[a]; g = k + grid
        Zk = np.empty((X.shape[1], N))
        for c in range(X.shape[1]):
            Zk[c] = np.interp(g, phi_m, X[:, c])
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


# ----------------------------------------------------------------------------- rolling mode (Sprint 7 Block W2)
def register_blocks(df: pd.DataFrame, channels: list[str], L_s: float = 1.0, N: int = 64, t_col: str = "t",
                    mask: np.ndarray | None = None, t_start: float | None = None, drop_first: int = 0, max_blocks: int | None = None):
    """Rolling-mode data elements (Σ = G, no phase): cut the telemetry into consecutive FIXED-DURATION blocks of L_s
    seconds and resample each block onto an N-point grid of normalized block time u = (t - t0)/L in [0, 1); only rows
    with `mask` True (e.g. straight-command segments after warm-up) are used — a block must be entirely inside one masked
    run. Returns Z (K, d, N) and meta; the pairing shift is 0 (C2Rep with delta_theta = 0 applies the pure reflection).
    """
    t = df[t_col].to_numpy(); X = df[list(channels)].to_numpy()
    m = np.ones(len(t), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if t_start is not None:
        m &= t >= t_start
    # contiguous masked runs
    idx = np.where(m)[0]
    if len(idx) == 0:
        return np.empty((0, X.shape[1], N)), {"N": N, "L_s": L_s, "channels": list(channels), "n_cycles": 0, "t_start": [], "row_start": []}
    breaks = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, breaks + 1)
    grid = np.arange(N) / N
    Zs, t0s, rows = [], [], []
    for r in runs:
        ta, tb = t[r[0]], t[r[-1]]
        nb = int(np.floor((tb - ta) / L_s))
        for b in range(nb):
            t0 = ta + b * L_s; t1 = t0 + L_s
            sel = r[(t[r] >= t0) & (t[r] < t1)]
            if len(sel) < 4:
                continue
            u = (t[sel] - t0) / L_s
            Zk = np.empty((X.shape[1], N))
            for c in range(X.shape[1]):
                Zk[c] = np.interp(grid, u, X[sel, c], left=X[sel[0], c], right=X[sel[-1], c])
            Zs.append(Zk); t0s.append(float(t0)); rows.append(int(sel[0]))
    Zs = Zs[drop_first:]; t0s = t0s[drop_first:]; rows = rows[drop_first:]
    if max_blocks is not None:
        Zs = Zs[:max_blocks]; t0s = t0s[:max_blocks]; rows = rows[:max_blocks]
    Z = np.stack(Zs) if Zs else np.empty((0, X.shape[1], N))
    meta = {"N": N, "L_s": L_s, "channels": list(channels), "n_cycles": int(Z.shape[0]), "dropped_first": drop_first,
            "t_start": t0s, "row_start": rows, "mode": "rolling"}
    return Z, meta


def straight_mask(df: pd.DataFrame, v_col: str = "v_cmd", warmup_s: float = 0.0, tol: float = 1e-6, t_col: str = "t") -> np.ndarray:
    """Rows where the commanded speed is at its (nonzero) plateau — i.e. steady straight driving — after warm-up."""
    v = df[v_col].to_numpy(); t = df[t_col].to_numpy()
    plateau = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 0.0
    return (np.abs(np.abs(v) - plateau) <= tol) & (t >= warmup_s) & (plateau > 0)
