#!/usr/bin/env bash
# ingest_session.sh — copy one recording session into the immutable raw area.
#
# Usage:
#   ingest_session.sh <src_dir> <raw_subpath> <session_name>
# Example:
#   ingest_session.sh ~/rec01 m1/nominal 20260901_trot_0.6_flat_north_rep01
#
# Effects:
#   1. rsync <src_dir>/ into $GEOFDI_DATA_ROOT/data/raw/<raw_subpath>/<session_name>/
#   2. write checksums.sha256 over all payload files (excludes meta.yaml so the
#      payload fingerprint stays stable while metadata is edited)
#   3. create a meta.yaml template (only if absent — fill it in afterwards)
#   4. best-effort chmod -R a-w. NOTE: on NTFS/drvfs mounts (the WSL setup)
#      chmod may be a no-op; until the migration to native Ubuntu/ext4, raw
#      immutability is convention + checksum verification, not enforcement.
#   5. append a row to docs/data_catalog.md (skipped if already present)
#
# Idempotent: re-running refreshes the payload and checksums; meta.yaml and
# existing catalog rows are left alone.
set -euo pipefail

usage() { echo "usage: $(basename "$0") <src_dir> <raw_subpath> <session_name>" >&2; exit 2; }
[[ $# -eq 3 ]] || usage

SRC=$1; SUBPATH=$2; SESSION=$3
: "${GEOFDI_DATA_ROOT:?GEOFDI_DATA_ROOT is not set — run scripts/setup_paths.sh and export the printed line}"
[[ -d "$SRC" ]] || { echo "ERROR: source dir not found: $SRC" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$GEOFDI_DATA_ROOT/data/raw/$SUBPATH/$SESSION"
mkdir -p "$DEST"

# allow refreshing a previous (possibly write-protected) ingest
chmod -R u+w "$DEST" 2>/dev/null || true

rsync -a --info=stats1 "$SRC"/ "$DEST"/

(
    cd "$DEST"
    find . -type f ! -name checksums.sha256 ! -name meta.yaml -print0 \
        | sort -z | xargs -0 -r sha256sum > checksums.sha256
)

META="$DEST/meta.yaml"
if [[ ! -f "$META" ]]; then
    cat > "$META" <<'EOF'
# Session metadata — fill in every field.
date: ""              # YYYY-MM-DD
robot: ""             # e.g. m1
gait: ""              # trot | walk | stand | ...
speed: ""             # commanded speed [m/s]
terrain: ""           # flat | grass | gravel | ramp | ...
direction: ""         # out | back  (paired-direction protocol, assumption A3)
motor_temp_start: ""  # [deg C]
motor_temp_end: ""    # [deg C]
operator: ""
notes: ""
topics: []            # recorded topics / signal groups
EOF
fi

chmod -R a-w "$DEST" 2>/dev/null \
    || echo "WARN: chmod a-w had no effect (expected on NTFS/drvfs)" >&2

CATALOG="$REPO_ROOT/docs/data_catalog.md"
SHA8="$( (cd "$DEST" && sha256sum checksums.sha256) | cut -c1-8 )"
ROW="| $(date +%F) | $SESSION | $SUBPATH | TODO | TODO | $SHA8 | |"
if grep -qF "| $SESSION | $SUBPATH |" "$CATALOG" 2>/dev/null; then
    echo "catalog: row for $SESSION already present — not duplicating"
else
    echo "$ROW" >> "$CATALOG"
    echo "catalog: appended row for $SESSION (fill gait/terrain/notes by hand)"
fi

echo "done: $DEST"
