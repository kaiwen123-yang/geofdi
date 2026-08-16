#!/usr/bin/env python3
"""bench_pipeline.py — per-stage timing of the GeoFDI pipeline on one ingested session (Sprint 10 Block N).

Answers the only question an onboard deployment asks: does a gait cycle's worth of work fit inside a gait cycle, on one
core? Every stage of the detector is timed separately on real recorded data, normalised to milliseconds **per gait
cycle** (per rolling block for a wheeled robot), and compared with the cycle period and with the telemetry rate.

Stages timed
  load       parse/deserialise the session into the GeoFDI telemetry frame (amortised: done once per session, not per cycle)
  segment    straight-segment mask + gait-phase estimate
  element    phase registration / block construction into the data element
  permute    the group-randomisation flip test, M permutations, on one window
  eprocess   e-value + running product
  h0prime    the recalibrated-null window test
  inekf      (optional) contact-aided invariant filter: propagate + update, per telemetry sample

Single-core discipline: run it under `taskset -c 0`, and the script pins every BLAS backend to one thread **before**
numpy is imported, so the numbers are one-core numbers rather than accidentally-parallel ones.

    taskset -c 0 python scripts/bench_pipeline.py <session_dir> --robot go2|m1 [--inekf] [--repeats 5]

Output: $GEOFDI_DATA_ROOT/results/bench/<hostname>/bench_table.csv (+ .md), appended across robots/hosts so a laptop and
an onboard computer can be compared row by row in the same file.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                                   # must precede the numpy import to bind the BLAS thread pool

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _md_table(d: pd.DataFrame) -> str:
    """Markdown table without pulling in `tabulate` — this script must run on a bare onboard machine."""
    cols = list(d.columns)
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


class Timer:
    """Repeated timing of a callable; reports the median to suppress scheduler noise."""

    def __init__(self, repeats: int = 5):
        self.repeats = repeats; self.rows = []

    def run(self, name: str, fn, n_units: int, unit: str = "cycle", note: str = ""):
        ts = []
        out = None
        for _ in range(max(1, self.repeats)):
            t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        self.rows.append({"stage": name, "median_s": med, "min_s": float(np.min(ts)), "n_units": int(n_units),
                          "unit": unit, "ms_per_unit": 1e3 * med / max(n_units, 1), "repeats": self.repeats, "note": note})
        print(f"  {name:9s} {1e3*med:9.1f} ms total | {1e3*med/max(n_units,1):8.3f} ms/{unit} (n={n_units}) {note}", flush=True)
        return out


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def bench_go2(session_dir: Path, T: Timer, alpha: float, M: int, window: int, want_inekf: bool):
    from geofdi.detect.evalue import eprocess
    from geofdi.detect.h0prime import h0prime_test
    from geofdi.detect.permutation import hg_permutation_test
    from geofdi.groups.c2 import C2Rep
    from geofdi.io.go2_quadric import LEGS, load_go2_quadric_session, straight_mask_go2
    from geofdi.phase.estimator import estimate_phase, gait_signal_from_columns
    from geofdi.phase.registration import register_cycles

    load_go2_quadric_session(session_dir)                          # warm the parquet cache: time the steady state
    T.run("load", lambda: load_go2_quadric_session(session_dir), 1, "session", "(amortised once per session)")
    df, man, rep = load_go2_quadric_session(session_dir)
    fs = float(rep["rate_hz"])
    # the per-cycle unit count must be known BEFORE timing, so the printed and stored units agree
    mask0, _ = straight_mask_go2(df)
    idx0 = np.where(mask0)[0]
    run0 = max(np.split(idx0, np.where(np.diff(idx0) > 1)[0] + 1), key=len)
    sub0 = df.iloc[run0[0]:run0[-1] + 1]
    _, pinfo0 = estimate_phase(sub0.reset_index(drop=True), contact_cols=[f"c_{l}" for l in LEGS],
                               signal=gait_signal_from_columns(sub0.reset_index(drop=True)))
    n_cyc0 = max(1, int((sub0["t"].iloc[-1] - sub0["t"].iloc[0]) / float(pinfo0["period_s"])))
    mask, minfo = T.run("segment", lambda: straight_mask_go2(df), n_cyc0, "cycle",
                        "(whole session; amortised over the cycles it yields)")
    idx = np.where(mask)[0]
    run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    sub = df.iloc[run[0]:run[-1] + 1].reset_index(drop=True)
    chans = [c["name"] for c in man["channels"] if c["in_Z"]]
    theta, pinfo = estimate_phase(sub, contact_cols=[f"c_{l}" for l in LEGS], signal=gait_signal_from_columns(sub))
    period = float(pinfo["period_s"])
    d2 = sub.copy(); d2["theta_hat"] = theta
    n_cycles = max(1, int((sub["t"].iloc[-1] - sub["t"].iloc[0]) / period))
    T.run("phase", lambda: estimate_phase(sub, contact_cols=[f"c_{l}" for l in LEGS],
                                          signal=gait_signal_from_columns(sub)), n_cycles)
    Z = T.run("element", lambda: register_cycles(d2, chans, N=64, theta_col="theta_hat", drop_first=3, drop_last=3),
              n_cycles)[0]
    rep2 = C2Rep(man); Zw = Z[:window]
    T.run("permute", lambda: hg_permutation_test(Zw, rep2, statistic="paired_energy", M=M, rng=np.random.default_rng(0)),
          window, "cycle", f"(M={M}, {window}-cycle window)")
    pw = np.random.default_rng(0).uniform(size=max(2, Z.shape[0] // window))
    T.run("eprocess", lambda: eprocess(pw, alpha), len(pw) * window)
    Kcal = max(window, Z.shape[0] // 3)
    T.run("h0prime", lambda: h0prime_test(Z[:Kcal], Z[Kcal:Kcal + window], rep2, M=M, rng=np.random.default_rng(0)),
          window, "cycle", f"(M={M})")
    n_samp = None
    if want_inekf:
        from geofdi.estimate.pi_gating import run_gated_filter
        from geofdi.inekf.kinematics import Go2Kinematics
        kin = Go2Kinematics(); short = sub.iloc[:min(len(sub), 3000)].reset_index(drop=True); n_samp = len(short)
        T.run("inekf", lambda: run_gated_filter(short, kin, mode="none", use_provided_feet=True, sigma_gyro=0.02,
                                                sigma_accel=0.2, sigma_contact=0.02, sigma_kin_floor=0.02),
              n_samp, "sample", "(propagate+update)")
    return {"period_s": period, "rate_hz": fs, "n_cycles": n_cycles, "K": int(Z.shape[0]), "d": int(Z.shape[1]),
            "n_inekf_samples": n_samp}


def bench_m1(session_dir: Path, T: Timer, alpha: float, M: int, window: int, want_inekf: bool):
    from geofdi.detect.evalue import eprocess
    from geofdi.detect.h0prime import h0prime_test
    from geofdi.detect.permutation import hg_permutation_test
    from geofdi.groups.c2 import C2Rep
    from geofdi.io.m1_sdk import load_m1_session
    from geofdi.phase.registration import register_blocks, straight_mask_kinematic

    load_m1_session(session_dir)                                   # warm the parquet cache so 'load' times the steady state
    T.run("load", lambda: load_m1_session(session_dir), 1, "session", "(amortised once per session)")
    df, man, rep = load_m1_session(session_dir)
    L = 1.0
    _, minfo0 = straight_mask_kinematic(df, warmup_s=6.0)
    n_blocks = max(1, int(minfo0["masked_duration_s"] / L))
    mask, minfo = T.run("segment", lambda: straight_mask_kinematic(df, warmup_s=6.0), n_blocks, "block",
                        "(whole session; amortised over the blocks it yields)")
    zn = [c["name"] for c in man["channels"] if c["in_Z"]]
    Z = T.run("element", lambda: register_blocks(df, zn, L_s=L, N=64, mask=mask), n_blocks, "block")[0]
    keep = [i for i, n in enumerate(zn) if np.isfinite(Z[:, i, :]).all()]
    Z = Z[:, keep, :]
    man2 = dict(man); man2["channels"] = [c for c in man["channels"] if (c["name"] in [zn[i] for i in keep]) or not c["in_Z"]]
    rep2 = C2Rep(man2); Zw = Z[:window]
    T.run("permute", lambda: hg_permutation_test(Zw, rep2, statistic="paired_energy", M=M, rng=np.random.default_rng(0)),
          window, "block", f"(M={M}, {window}-block window)")
    pw = np.random.default_rng(0).uniform(size=max(2, Z.shape[0] // window))
    T.run("eprocess", lambda: eprocess(pw, alpha), len(pw) * window, "block")
    Kcal = max(window, Z.shape[0] // 3)
    T.run("h0prime", lambda: h0prime_test(Z[:Kcal], Z[Kcal:Kcal + window], rep2, M=M, rng=np.random.default_rng(0)),
          window, "block", f"(M={M})")
    return {"period_s": L, "rate_hz": float(rep["rate_hz_estimate"]), "n_cycles": n_blocks, "K": int(Z.shape[0]),
            "d": int(Z.shape[1]), "n_inekf_samples": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--robot", choices=["go2", "m1"], required=True)
    ap.add_argument("--inekf", action="store_true"); ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--window", type=int, default=10); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    host = socket.gethostname()
    aff = ""
    try:
        aff = f"{sorted(os.sched_getaffinity(0))}"
    except Exception:
        pass
    print(f"[bench] host={host} cpu='{_cpu_model()}' affinity={aff} BLAS threads=1 repeats={a.repeats}", flush=True)
    print(f"[bench] session={a.session_dir} robot={a.robot}", flush=True)
    T = Timer(a.repeats)
    meta = (bench_go2 if a.robot == "go2" else bench_m1)(Path(a.session_dir), T, a.alpha, a.M, a.window, a.inekf)
    df = pd.DataFrame(T.rows)
    df.insert(0, "robot", a.robot); df.insert(0, "session", Path(a.session_dir).name); df.insert(0, "host", host)
    df["cpu"] = _cpu_model(); df["single_core"] = len(os.sched_getaffinity(0)) == 1 if hasattr(os, "sched_getaffinity") else None
    df["cycle_period_s"] = meta["period_s"]; df["telemetry_hz"] = meta["rate_hz"]; df["tag"] = a.tag
    # per-cycle budget: everything except the one-off load
    per_cycle = df[(df.unit.isin(["cycle", "block"])) & (df.stage != "load")]["ms_per_unit"].sum()
    budget_ms = 1e3 * meta["period_s"]
    inekf_row = df[df.unit == "sample"]
    inekf_ms = float(inekf_row["ms_per_unit"].iloc[0]) if len(inekf_row) else np.nan
    summary = {"host": host, "robot": a.robot, "session": Path(a.session_dir).name, "cpu": _cpu_model(),
               "detector_ms_per_cycle": round(per_cycle, 3), "cycle_period_ms": round(budget_ms, 1),
               "realtime_factor": round(budget_ms / per_cycle, 1) if per_cycle > 0 else np.nan,
               "inekf_ms_per_sample": round(inekf_ms, 4) if np.isfinite(inekf_ms) else None,
               "inekf_sustainable_hz": round(1e3 / inekf_ms, 1) if np.isfinite(inekf_ms) and inekf_ms > 0 else None,
               "telemetry_hz": round(meta["rate_hz"], 1), "K": meta["K"], "d": meta["d"], "M": a.M, "window": a.window}
    out = Path(os.environ["GEOFDI_DATA_ROOT"]) / "results" / "bench" / host
    out.mkdir(parents=True, exist_ok=True)
    p = out / "bench_table.csv"
    df.to_csv(p, mode="a", header=not p.exists(), index=False)
    ps = out / "bench_summary.csv"
    pd.DataFrame([summary]).to_csv(ps, mode="a", header=not ps.exists(), index=False)
    (out / f"bench_{a.robot}_{Path(a.session_dir).name}.md").write_text(
        f"# GeoFDI pipeline benchmark — {host} / {a.robot} / {Path(a.session_dir).name}\n\n"
        f"CPU: `{_cpu_model()}`; affinity {aff}; BLAS threads 1; median of {a.repeats} repeats; M={a.M}, window={a.window}.\n\n"
        + _md_table(df[["stage", "median_s", "n_units", "unit", "ms_per_unit", "note"]])
        + "\n\n```json\n" + json.dumps(summary, indent=1) + "\n```\n")
    print()
    print(f"[bench] detector {summary['detector_ms_per_cycle']} ms per {'cycle' if a.robot=='go2' else 'block'} "
          f"vs a {summary['cycle_period_ms']} ms budget -> real-time factor {summary['realtime_factor']}x")
    if summary["inekf_sustainable_hz"]:
        print(f"[bench] InEKF {summary['inekf_ms_per_sample']} ms/sample -> sustainable "
              f"{summary['inekf_sustainable_hz']} Hz vs {summary['telemetry_hz']} Hz telemetry")
    print(f"[bench] -> {p}")


if __name__ == "__main__":
    main()
