#!/usr/bin/env bash
# One-shot, idempotent environment setup for WSL2 Ubuntu (22.04).
#
# Installs system packages (build + TeX toolchain), creates the project venv at
# ~/venvs/geofdi, installs the package editable with dev extras, and wires the
# data/ and results/ symlinks via scripts/setup_paths.sh.
#
# Data lives on the Windows data volume — on this machine /mnt/g/geofdi-data,
# declared in env/machines/wsl.yaml. That yaml, this comment block, and the
# README are the only places the mount path may be spelled out; everything else
# goes through $GEOFDI_DATA_ROOT or the repo symlinks.
#
# Network note: this laptop reaches the internet through the Windows host's
# Clash Verge proxy. If apt/pip time out, export the proxy first:
#   export https_proxy=http://$(ip route show default | awk '{print $3}'):7897
#   export http_proxy=$https_proxy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/venvs/geofdi"

sudo apt-get update
sudo apt-get install -y \
    build-essential git python3-venv tree zip \
    latexmk texlive-latex-extra texlive-science

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$REPO_ROOT[dev]"

bash "$REPO_ROOT/scripts/setup_paths.sh"
