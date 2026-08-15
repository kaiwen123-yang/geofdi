#!/usr/bin/env bash
# Fetch the visual mesh assets of the Unitree Go2 from MuJoCo Menagerie (only needed for rendering;
# GeoFDI's headless simulation uses go2_sym.xml, which has no meshes). Idempotent; ~30 MB.
set -euo pipefail
COMMIT=da76818e269b82289eba39808e2fb91d679d6994
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/geofdi/sim/assets/unitree_go2/assets"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/m"
( cd "$TMP/m" && git sparse-checkout set unitree_go2 >/dev/null && git checkout --quiet "$COMMIT" 2>/dev/null || true )
mkdir -p "$DEST" && cp -r "$TMP/m/unitree_go2/assets/." "$DEST"/ && echo "meshes in $DEST (menagerie $(git -C "$TMP/m" rev-parse --short HEAD))"
