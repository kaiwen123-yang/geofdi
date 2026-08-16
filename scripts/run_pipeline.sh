#!/usr/bin/env bash
# run_pipeline.sh — the Day-0 first command: one recorded session -> loader -> data elements -> R- H0 / H0' -> report.
#
#   scripts/run_pipeline.sh <session_dir> --robot m1|go2 --mode rolling|trot [--residual off|analytic|delan_equiv] [more args]
#
# Report reading (rolling mode): the PRIMARY test on hardware is H0' (asymmetry-CHANGE): naive H0 (the whole-session flip
# test) IS EXPECTED to reject on a healthy real robot, because the healthy loop is stably asymmetric (real epsilon_dyn) and
# fixed-duration blocks are serially correlated at short L (Sprint 7 Block W; confirmed on the M1 hardware 2026-08-10, all
# three sessions rejected naive H0 while the sequential H0' monitor stayed silent, docs/protocol/m1_h_data_audit.md §13).
# Read: H0' differenced/per-window p and the H0' e-process (should not alarm within a healthy session); nu_0 is the robot's
# stable asymmetry level; the block-correlation lag-1 tells you whether to raise L (1 s default -> 2 s).
#
# <session_dir> is a raw session directory (data/raw/<robot>/... or data/raw/sim/*_rehearsal/<session>) with the
# per-robot file layout of io/m1_mapping.yaml (M1: joint_states.csv, imu.csv, cmd.csv, meta.yaml) or io/go2_mapping.yaml
# (Go2: lowstate.csv, lowcmd.csv, meta.yaml). Output: $GEOFDI_DATA_ROOT/results/pipeline/<robot>_<session>_<stamp>/report.md
# (+ report.json, rminus_qq_eprocess.png). Nothing here needs a controller clock or fault data.
set -euo pipefail
: "${GEOFDI_DATA_ROOT:?GEOFDI_DATA_ROOT is not set — run scripts/setup_paths.sh and export the printed line}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/geofdi}"
[[ $# -ge 3 ]] || { echo "usage: $(basename "$0") <session_dir> --robot m1|go2 --mode rolling|trot [--residual off|analytic|delan_equiv]" >&2; exit 2; }
SESSION=$1; shift
[[ -d "$SESSION" ]] || { echo "ERROR: session dir not found: $SESSION" >&2; exit 1; }
PYTHONPATH= "$VENV/bin/python" -m geofdi.pipeline.run_session "$SESSION" "$@"
