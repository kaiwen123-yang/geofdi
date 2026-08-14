#!/usr/bin/env bash
# setup_paths.sh — resolve this machine's data root and (re)create the
# repo-local symlinks  data/  and  results/  pointing into it.
#
# Machine resolution order:
#   1. $GEOFDI_MACHINE (must name a file env/machines/<name>.yaml)
#   2. /proc/version contains "microsoft"  ->  wsl
#   3. hostname matches a machine yaml name
#
# The data root itself is declared ONLY in env/machines/*.yaml. Never hard-code
# mount paths in code; use $GEOFDI_DATA_ROOT or the data/ and results/ symlinks.
# Idempotent: ln -sfn re-points existing links. This script does NOT edit your
# shell rc; it prints the export line to add yourself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACHINES_DIR="$REPO_ROOT/env/machines"

detect_machine() {
    if [[ -n "${GEOFDI_MACHINE:-}" ]]; then
        echo "$GEOFDI_MACHINE"
        return 0
    fi
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl"
        return 0
    fi
    local hn
    hn="$(hostname)"
    if [[ -f "$MACHINES_DIR/$hn.yaml" ]]; then
        echo "$hn"
        return 0
    fi
    echo "ERROR: cannot identify machine; set GEOFDI_MACHINE to one of:" >&2
    ls "$MACHINES_DIR" | sed 's/\.yaml$//' >&2
    return 1
}

MACHINE="$(detect_machine)"
YAML="$MACHINES_DIR/$MACHINE.yaml"
[[ -f "$YAML" ]] || { echo "ERROR: no machine profile $YAML" >&2; exit 1; }

DATA_ROOT="$(sed -n 's/^data_root:[[:space:]]*//p' "$YAML" | head -n1)"
[[ -n "$DATA_ROOT" ]] || { echo "ERROR: data_root not set in $YAML" >&2; exit 1; }

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "WARNING: data root $DATA_ROOT does not exist (yet); creating symlinks anyway." >&2
fi

ln -sfn "$DATA_ROOT/data"    "$REPO_ROOT/data"
ln -sfn "$DATA_ROOT/results" "$REPO_ROOT/results"

echo "machine          : $MACHINE"
echo "data root        : $DATA_ROOT"
echo "symlink data/    -> $(readlink "$REPO_ROOT/data")"
echo "symlink results/ -> $(readlink "$REPO_ROOT/results")"
echo
echo "Add this line to your shell rc (~/.bashrc) if not present:"
echo "  export GEOFDI_DATA_ROOT=\"$DATA_ROOT\""
