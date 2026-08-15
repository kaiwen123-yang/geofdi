#!/usr/bin/env python3
"""Synthetic rehearsal sessions for the two hardware pipelines (Sprint 7 Block W3/W4):
  Go2: go2_urdf_sym trot, 3 speeds x 2 directions x 30 s, exported in the Unitree LowState CSV layout (io/go2_lowstate)
  M1 : m1_wheeled_sym rolling, 3 speeds x 2 directions x 30 s, exported in the GENISOM SDK CSV layout (io/m1_sdk)
and ingested with scripts/ingest_session.sh into $GEOFDI_DATA_ROOT/data/raw/sim/{go2_rehearsal,m1_rehearsal}/<session>.
"Back" runs are independent seeds with direction: back in meta (the flat sim world has no direction; the paired-direction
bookkeeping of assumption A3 is what is rehearsed).

    python scripts/make_rehearsal_sessions.py [--robot go2|m1|both] [--no-ingest]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
DATE = "20260816"


def go2_sessions(ingest: bool):
    from geofdi.io.go2_lowstate import write_go2_session
    from geofdi.sim.env import SimConfig, rollout
    made = []
    for si, sp in enumerate((0.3, 0.5, 0.8)):
        for di, direction in enumerate(("out", "back")):
            seed = 81000 + 10 * si + di
            df, man = rollout(SimConfig(model="go2_urdf_sym", speed=sp, duration_s=30.0, seed=seed))
            name = f"{DATE}_trot_{sp}_flat_{direction}_rep01"
            meta = {"date": "2026-08-16", "robot": "go2", "source": "sim (go2_urdf_sym, geofdi.sim.env)", "gait": "trot", "speed": sp, "terrain": "flat",
                    "direction": direction, "rate_hz": 200, "seed": seed, "motor_temp_start": "", "motor_temp_end": "", "operator": "sim", "notes": "rehearsal session for run_pipeline.sh (Sprint 7 W3)",
                    "topics": ["lowstate.csv (LowState layout)", "lowcmd.csv (tau command)"]}
            with tempfile.TemporaryDirectory() as tmp:
                write_go2_session(tmp, df, meta)
                if ingest:
                    subprocess.run(["bash", str(REPO / "scripts" / "ingest_session.sh"), tmp, "sim/go2_rehearsal", name], check=True)
                made.append(name); print("go2 session", name, len(df), "rows", flush=True)
    return made


def m1_sessions(ingest: bool):
    from geofdi.io.m1_sdk import write_m1_session
    from geofdi.sim.env_m1 import SimConfigM1, rollout_m1
    made = []
    for si, sp in enumerate((0.5, 1.0, 2.0)):
        for di, direction in enumerate(("out", "back")):
            seed = 82000 + 10 * si + di
            df, man = rollout_m1(SimConfigM1(model="m1_wheeled_sym", speed=sp, duration_s=36.0, seed=seed, controller={"ramp_s": 3.0}))
            name = f"{DATE}_rolling_{sp}_flat_{direction}_rep01"
            meta = {"date": "2026-08-16", "robot": "m1_wheeled", "source": "sim (m1_wheeled_sym, geofdi.sim.env_m1)", "gait": "rolling", "speed": sp, "terrain": "flat",
                    "direction": direction, "rate_hz": 200, "seed": seed, "efforts_semantics": "torque", "motor_temp_start": "", "motor_temp_end": "", "operator": "sim",
                    "notes": "rehearsal session for run_pipeline.sh (Sprint 7 W3); SDK-name layout with a shuffled column order", "topics": ["joint_states.csv", "imu.csv", "cmd.csv"]}
            with tempfile.TemporaryDirectory() as tmp:
                write_m1_session(tmp, df, meta)
                if ingest:
                    subprocess.run(["bash", str(REPO / "scripts" / "ingest_session.sh"), tmp, "sim/m1_rehearsal", name], check=True)
                made.append(name); print("m1 session", name, len(df), "rows", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--robot", choices=["go2", "m1", "both"], default="both"); ap.add_argument("--no-ingest", action="store_true")
    a = ap.parse_args()
    if a.robot in ("go2", "both"):
        go2_sessions(not a.no_ingest)
    if a.robot in ("m1", "both"):
        m1_sessions(not a.no_ingest)


if __name__ == "__main__":
    main()
