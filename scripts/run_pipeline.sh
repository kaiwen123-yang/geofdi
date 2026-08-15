#!/usr/bin/env bash
# run_pipeline.sh — the Day-0 first command: one recorded session -> loader -> data elements -> R- H0 / H0' -> report.
#
#   scripts/run_pipeline.sh <session_dir> --robot m1|go2 --mode rolling|trot [--residual off|analytic|delan_equiv] [more args]
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
