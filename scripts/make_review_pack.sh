#!/usr/bin/env bash
# make_review_pack.sh — assemble a review pack for the human reviewer.
#
# Usage:
#   make_review_pack.sh <NNN> <topic-slug> [extra files/dirs ...]
# Example:
#   make_review_pack.sh 001 gate1-controller-symmetry results/e01_h0_qq/run1/figures
#
# Assembles, in a temp dir:
#   MANIFEST.md   template with Purpose / Expected / Actual / Git commit /
#                 Data hashes / Open questions (commit pre-filled). If one of
#                 the extra files is itself named MANIFEST.md it replaces the
#                 template.
#   figures/ tables/ logs/ code/
# Extra args are bucketed by extension: png/pdf/svg/jpg -> figures,
# csv/tsv/parquet -> tables, log/txt/out -> logs, everything else -> code.
# A directory named figures|tables|logs|code is merged into that bucket;
# other directories are copied into code/ recursively.
#
# Output: $GEOFDI_DATA_ROOT/review/outbox/rp<NNN>_<YYYYMMDD>_<slug>.zip
# Warns if the zip exceeds 20 MB and lists the ten largest staged files.
# Idempotent: an existing zip of the same name is overwritten.
set -euo pipefail

usage() { echo "usage: $(basename "$0") <NNN> <topic-slug> [files...]" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
NNN=$1; SLUG=$2; shift 2
[[ "$NNN" =~ ^[0-9]{3}$ ]] || { echo "ERROR: pack number must be three digits (got '$NNN')" >&2; exit 1; }
: "${GEOFDI_DATA_ROOT:?GEOFDI_DATA_ROOT is not set — run scripts/setup_paths.sh and export the printed line}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
OUTDIR="$GEOFDI_DATA_ROOT/review/outbox"
OUT="$OUTDIR/rp${NNN}_${STAMP}_${SLUG}.zip"
mkdir -p "$OUTDIR"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"/{figures,tables,logs,code}

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo 'no-commit')"
cat > "$STAGE/MANIFEST.md" <<EOF
# Review pack rp${NNN} — ${SLUG} (${STAMP})

## Purpose
TODO

## Expected
TODO

## Actual
TODO

## Git commit
${GIT_COMMIT}

## Data hashes
TODO (checksums.sha256 fingerprints of every session used)

## Open questions
TODO
EOF

bucket_for() {
    case "${1##*.}" in
        png|pdf|svg|jpg|jpeg) echo figures ;;
        csv|tsv|parquet)      echo tables ;;
        log|txt|out)          echo logs ;;
        *)                    echo code ;;
    esac
}

for item in "$@"; do
    if [[ ! -e "$item" ]]; then
        echo "WARN: skipping missing item $item" >&2
        continue
    fi
    base="$(basename "$item")"
    if [[ "$base" == MANIFEST.md && -f "$item" ]]; then
        cp "$item" "$STAGE/MANIFEST.md"
    elif [[ -d "$item" ]]; then
        case "$base" in
            figures|tables|logs|code) cp -r "$item"/. "$STAGE/$base"/ ;;
            *)                        cp -r "$item" "$STAGE/code/$base" ;;
        esac
    else
        cp "$item" "$STAGE/$(bucket_for "$base")/"
    fi
done

rm -f "$OUT"
( cd "$STAGE" && zip -qr "$OUT" . )

SIZE=$(stat -c%s "$OUT")
echo "wrote $OUT ($(numfmt --to=iec "$SIZE" 2>/dev/null || echo "$SIZE bytes"))"
LIMIT=$((20 * 1024 * 1024))
if (( SIZE > LIMIT )); then
    echo "WARN: pack exceeds 20 MB — ten largest staged files:" >&2
    ( cd "$STAGE" && find . -type f -printf '%s\t%p\n' | sort -rn | head -10 \
        | awk -F'\t' '{printf "  %8.1f MB  %s\n", $1/1048576, $2}' ) >&2
fi
