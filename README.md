# geofdi

**Geo**metric **F**ault **D**etection & **I**solation for manipulators and legged robots:
a healthy, morphologically symmetric robot is treated as a *distributional invariant* —
faults reveal themselves as symmetry breaking in phase-registered gait data, with no fault
model required. Theory notes live in `theory/` (Part 0: hypotheses and assumptions);
target venue for the paper is IEEE T-RO.

## 三条铁律 / The three iron rules

1. **Code on ext4, data on the data volume.** The repo lives at `~/research/geofdi`
   (never on the Windows mount). Bulk data lives under this machine's data root — on the
   WSL laptop that is `/mnt/g/geofdi-data`, declared **only** in `env/machines/wsl.yaml`.
   Code and configs reach data exclusively through `$GEOFDI_DATA_ROOT` or the repo
   symlinks `data/` and `results/`; a literal mount path anywhere outside
   `env/machines/*.yaml`, this README, or setup-script comments is a bug.
2. **git tracks text only.** `data/` and `results/` are gitignored symlinks; binary
   extensions are flagged in `.gitattributes`; anything a reviewer needs travels as a
   review pack (see below), never as a committed blob.
3. **Raw data is immutable.** Ingest sessions with `scripts/ingest_session.sh`
   (rsync → `checksums.sha256` → `meta.yaml` → catalog row) and never edit them in
   place. On NTFS/drvfs immutability is convention + checksums; after the migration to
   native Ubuntu/ext4 it becomes chmod-enforced.

## Quick start

```sh
git clone git@github.com:kaiwen123-yang/geofdi.git ~/research/geofdi
cd ~/research/geofdi
make setup     # venv at ~/venvs/geofdi + editable install with dev extras
make links     # create data/, results/ symlinks for this machine (prints the export line)
make test      # pytest
make theory    # build theory/build/main.pdf (needs latexmk + texlive)
```

First time on a fresh WSL machine: `bash env/setup_wsl.sh` (apt packages incl. the TeX
toolchain, then venv + symlinks; needs sudo). Add the `export GEOFDI_DATA_ROOT=...` line
that `make links` prints to your shell rc.

## Layout

| path | what lives there |
|---|---|
| `src/geofdi/` | library code (src layout): `groups` `residuals` `phase` `detect` `inekf` `isolation` `dynamics` `io` `sim` `viz` |
| `experiments/eNN_<slug>/` | one `config.yaml` + `run.py` per experiment; outputs go to `results/<exp>/<run_id>/` |
| `theory/` | LaTeX theory notes; Part 0 formalizes H0 and assumptions A1–A4 |
| `docs/` | `decisions.md` (decision log), `data_catalog.md` (session catalog), `protocol/` (audit protocols) |
| `scripts/` | `setup_paths.sh`, `ingest_session.sh`, `make_review_pack.sh`, `m1_bag_tools/` |
| `env/` | `setup_wsl.sh` + per-machine data-root declarations in `machines/*.yaml` |
| `data/`, `results/` | symlinks into `$GEOFDI_DATA_ROOT` (created by `make links`, never committed) |

## Review-pack workflow

Results are reviewed asynchronously as zip packs, not by poking around the repo:

```sh
make review-pack ARGS="001 gate1-controller-symmetry path/to/fig.png path/to/notes.md"
```

assembles `MANIFEST.md` (Purpose / Expected / Actual / Git commit / Data hashes / Open
questions) plus `figures/ tables/ logs/ code/`, and writes
`$GEOFDI_DATA_ROOT/review/outbox/rp001_<YYYYMMDD>_gate1-controller-symmetry.zip`
(warns above 20 MB and lists the largest files). Reviewer feedback returns via
`review/feedback/`; settled packs move to `review/archive/`.
