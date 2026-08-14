#!/usr/bin/env python3
"""e03 — head-to-head vs. baselines on the public Liu et al. A1 fault dataset.

Stub: parses the experiment config, prints the execution plan, and exits.
Real implementation lands with workstream N3 (public-data loaders + metrics).
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import yaml

EXP_NAME = "e03_liu_a1_headtohead"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) or {}
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / EXP_NAME / args.run_id

    print(f"[{EXP_NAME}] plan (stub — nothing executed):")
    print(f"  config : {args.config}")
    for key, val in cfg.items():
        print(f"    {key}: {val}")
    print(f"  output : {out_dir}")
    print("  status : STUB — implementation pending (N3)")


if __name__ == "__main__":
    main()
