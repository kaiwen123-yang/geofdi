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
#                 the extra files is named MANIFEST.md (or MANIFEST*.md, e.g.
#                 MANIFEST_rp025.md) it replaces the template.
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

MANIFEST_GIVEN=0
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
    # Any file whose name contains MANIFEST (any case, any prefix/suffix: MANIFEST.md, MANIFEST_rp025.md,
    # rp031_manifest.md) replaces the top-level template. History: rp020-024 passed MANIFEST_rpNNN.md and the strict
    # "MANIFEST.md" match sent it to code/, leaving the TODO stub at top level (Sprint 7 bug, Sprint 8 partial fix).
    # The silent-failure mode is closed for good by the post-build assertion below, not only by this matcher.
    shopt -s nocasematch
    if [[ -f "$item" && "$base" == *manifest*.md ]]; then
        shopt -u nocasematch
        cp "$item" "$STAGE/MANIFEST.md"; MANIFEST_GIVEN=1
    elif [[ -d "$item" ]]; then
        shopt -u nocasematch
        case "$base" in
            figures|tables|logs|code) cp -r "$item"/. "$STAGE/$base"/ ;;
            *)                        cp -r "$item" "$STAGE/code/$base" ;;
        esac
    else
        cp "$item" "$STAGE/$(bucket_for "$base")/"
    fi
done

# --- self-check (Sprint 9 B9): the packing bug must never fail silently again. If a MANIFEST was supplied, the staged
# top-level MANIFEST.md must NOT still be the stub template; otherwise the pack would ship with an empty Purpose.
if grep -qx "TODO" "$STAGE/MANIFEST.md" 2>/dev/null; then
    if [[ "$MANIFEST_GIVEN" == "1" ]]; then
        echo "ERROR: a MANIFEST file was passed but the top-level MANIFEST.md is still the TODO template." >&2
        echo "       (staged files: $(cd "$STAGE" && ls))" >&2
        exit 1
    fi
    echo "WARN: no MANIFEST supplied — the pack ships with the TODO template. Pass one as an extra argument." >&2
fi

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
