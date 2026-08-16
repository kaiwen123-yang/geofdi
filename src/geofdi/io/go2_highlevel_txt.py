"""Unitree Go2 high-level telemetry captured as a terminal transcript (Sprint 9 Block Q1).

The QUADRIC-GINS Go2 sessions were recorded by running `ros2 topic echo /sportmodestate` inside `script(1)`, so each
file is a TERMINAL CAPTURE (ANSI escapes, CR line endings, a shell preamble) whose payload is a stream of YAML-ish
`unitree_go/msg/SportModeState` records separated by `---`. This module turns such a transcript into a DataFrame without
a YAML parser (a real YAML round-trip on ~90 000 records x ~66 lines per file is far too slow).

Message fields (all present in this corpus; see docs/protocol/go2_quadric_audit.md):
    stamp.sec/.nanosec, error_code, imu_state.{quaternion[4] (w,x,y,z), gyroscope[3], accelerometer[3], rpy[3],
    temperature}, mode, progress, gait_type, foot_raise_height, position[3], body_height, velocity[3], yaw_speed,
    range_obstacle[4], foot_force[4], foot_position_body[12], foot_speed_body[12].
**There is NO motorState** — the high-level SportModeState API carries no per-joint q/dq/tau (that is LowState). The
mirror element therefore uses the foot/IMU channels; see io/go2_quadric.py.

Robustness (the "tolerate variants" requirement): the parser is schema-agnostic. Any `key:` / `  key:` line becomes a
column path, scalars are stored as-is and `- x` items are appended to the open list; **unknown paths are kept** and
exported as `col_<path>` so a field the vendor adds or renames is never silently dropped. Lines that match nothing
(shell prompts, ANSI noise, typed commands) are counted in `report["skipped_lines"]` and ignored.

    df, report = parse_sportmodestate(path)             # DataFrame, one row per record
    df, report = parse_sportmodestate(path, max_records=1000)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|[\x00-\x08\x0b\x0c\x0e-\x1f]")
KEY = re.compile(r"^(\s*)([A-Za-z_][A-Za-z_0-9]*):\s*(.*)$")
ITEM = re.compile(r"^\s*-\s*(.*)$")

# canonical SportModeState layout: path -> (n_elements, output column stem)
VECTORS = {
    "imu_state.quaternion": (4, "quat"), "imu_state.gyroscope": (3, "gyro"), "imu_state.accelerometer": (3, "acc"),
    "imu_state.rpy": (3, "rpy"), "position": (3, "pos"), "velocity": (3, "vel"), "range_obstacle": (4, "range_obstacle"),
    "foot_force": (4, "foot_force"), "foot_position_body": (12, "foot_pos"), "foot_speed_body": (12, "foot_vel"),
}
SCALARS = {"stamp.sec": "stamp_sec", "stamp.nanosec": "stamp_nsec", "error_code": "error_code",
           "imu_state.temperature": "imu_temp", "mode": "mode", "progress": "progress", "gait_type": "gait_type",
           "foot_raise_height": "foot_raise_height", "body_height": "body_height", "yaw_speed": "yaw_speed"}


def _f(s: str):
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_sportmodestate(path: str | Path, max_records: int | None = None, chunk_lines: int = 1 << 20):
    """Parse a `ros2 topic echo /sportmodestate` terminal transcript. Returns (DataFrame, report)."""
    path = Path(path)
    rows: list[dict] = []
    cur: dict = {}
    list_path: str | None = None
    list_buf: list[float] = []
    top_key = ""
    skipped = 0
    unknown_paths: dict[str, int] = {}
    n_partial = 0

    def close_list():
        nonlocal list_path, list_buf
        if list_path is not None and list_buf:
            cur[list_path] = list(list_buf)
        list_path, list_buf = None, []

    def flush():
        nonlocal cur, n_partial
        close_list()
        if "stamp.sec" in cur:
            rows.append(cur)
        elif cur:
            n_partial += 1
        cur = {}

    with open(path, "rb") as fh:
        for raw in fh:
            line = ANSI.sub(b"", raw).decode("utf-8", "replace").rstrip("\r\n")
            if not line.strip():
                continue
            if line.strip() == "---":
                flush()
                if max_records is not None and len(rows) >= max_records:
                    break
                continue
            m = ITEM.match(line)
            if m and list_path is not None:
                list_buf.append(_f(m.group(1)))
                continue
            m = KEY.match(line)
            if m is None:
                skipped += 1
                continue
            indent, key, val = len(m.group(1)), m.group(2), m.group(3).strip()
            close_list()
            if indent == 0:
                top_key = key
                p = key
            else:
                p = f"{top_key}.{key}"
            if val == "":
                list_path = p                      # opens a list (or a mapping; harmless either way)
                list_buf = []
            else:
                cur[p] = _f(val)
                if p not in SCALARS:
                    unknown_paths[p] = unknown_paths.get(p, 0) + 1
    flush()

    # ------------------------------------------------------------------ to DataFrame
    out: dict[str, np.ndarray] = {}
    n = len(rows)
    for p, name in SCALARS.items():
        out[name] = np.array([r.get(p, np.nan) for r in rows], dtype=float)
    for p, (k, stem) in VECTORS.items():
        arr = np.full((n, k), np.nan)
        for i, r in enumerate(rows):
            v = r.get(p)
            if v is not None:
                arr[i, :min(k, len(v))] = v[:k]
        for j in range(k):
            out[f"{stem}{j}"] = arr[:, j]
        if any((r.get(p) is not None and len(r[p]) != k) for r in rows[:100]):
            unknown_paths[f"{p}(length!={k})"] = 1
    known = set(VECTORS) | set(SCALARS)
    extra = sorted({p for r in rows for p in r} - known)
    for p in extra:                                 # vendor additions: kept, never dropped
        col = "col_" + p.replace(".", "_")
        vals = [r.get(p, np.nan) for r in rows]
        out[col] = np.array([v if np.isscalar(v) else (v[0] if v else np.nan) for v in vals], dtype=float)
    df = pd.DataFrame(out)
    if n:
        t_abs = df["stamp_sec"].to_numpy() + df["stamp_nsec"].to_numpy() * 1e-9
        df.insert(0, "t_abs", t_abs); df.insert(0, "t", t_abs - t_abs[0])
    dt = np.diff(df["t_abs"].to_numpy()) if n > 2 else np.array([np.nan])
    report = {"file": str(path), "n_records": int(n), "n_partial_records": int(n_partial), "skipped_lines": int(skipped),
              "duration_s": float(df["t"].iloc[-1]) if n else 0.0,
              "rate_hz_median": float(1.0 / np.median(dt)) if n > 2 else float("nan"),
              "dt_ms_median": float(np.median(dt) * 1e3) if n > 2 else float("nan"),
              "dt_ms_max": float(np.nanmax(dt) * 1e3) if n > 2 else float("nan"),
              "n_time_backjumps": int(np.sum(dt <= 0)) if n > 2 else 0,
              "n_gaps_gt_100ms": int(np.sum(dt > 0.1)) if n > 2 else 0,
              "extra_paths": extra, "unknown_scalar_paths": sorted(unknown_paths),
              "t0_utc": pd.Timestamp(df["t_abs"].iloc[0], unit="s", tz="UTC").isoformat() if n else None,
              "has_motorstate": any("motor" in p.lower() for p in extra)}
    return df, report


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+"); ap.add_argument("--max-records", type=int, default=None); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    reps = []
    for p in a.paths:
        df, rep = parse_sportmodestate(p, a.max_records)
        reps.append(rep)
        print(f"{Path(p).name}: {rep['n_records']} rec, {rep['duration_s']:.1f} s, {rep['rate_hz_median']:.1f} Hz, "
              f"t0 {rep['t0_utc']}, gaps>100ms {rep['n_gaps_gt_100ms']}, backjumps {rep['n_time_backjumps']}, "
              f"skipped {rep['skipped_lines']}, extra {rep['extra_paths']}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(reps, indent=1, default=str))


if __name__ == "__main__":
    main()
