# CI-less verification record — init + theory Part 0 (2026-08-15)

Manual verification performed on the WSL machine before assembling review pack rp000.
(No CI exists yet; this file is the audit trail for the first three commits.)

## Checks

| check | result |
|---|---|
| repo at `~/research/geofdi` (ext4), 4 conventional commits | PASS |
| `data/`, `results/` are symlinks into the machine data root | PASS (`setup_paths.sh` output verified) |
| data root tree incl. `review/{outbox,archive,feedback}` provisioned | PASS |
| no literal mount paths outside `env/machines/wsl.yaml`, `README.md`, `env/setup_wsl.sh` comments | PASS (`git grep` clean) |
| `make test` | PASS (1 passed) — note: `PYTHONPATH` cleared in the test target because the host ROS Humble install leaks broken pytest plugins into any venv |
| `make theory` | PASS — `theory/build/main.pdf`, 11 pages, zero errors, zero undefined citations, bibtex clean |
| experiment stubs parse configs and print plans | PASS (e01 smoke run) |
| GitHub SSH auth (`git@github.com`) | PASS ("Hi kaiwen123-yang!") |

## Environment notes

- System TeX Live could not be apt-installed (sudo requires a password in this
  session); the build above used a user-space TinyTeX at `~/.TinyTeX` instead.
  `env/setup_wsl.sh` still installs the system TeX toolchain
  (`latexmk texlive-latex-extra texlive-science`) for the normal path — run it
  once with sudo when convenient.
- `tree` is not installed; the review-pack tree listing was generated with
  `find` (depth 3).

## Citation status (rule: zero fabrication)

All 11 entries in `theory/references.bib` are finalized. The 8 supplied
`[verified]` entries were used as-is. The 3 gait-symmetry entries
(Golubitsky–Stewart–Buono–Collins Nature 1999; Collins–Stewart JNS 1993;
Golubitsky–Stewart *The Symmetry Perspective* 2002) were verified against
publisher pages (nature.com/articles/44416; Springer BF02429870; Springer
978-3-0348-8167-8 + LMS Bulletin review) on 2026-08-15. No `\todo{verify
citation}` markers remain.

## Open TODOs carried forward

1. **M1 joint count/ordering and per-joint reflection signs** — unresolved
   whether M1 is a 12-DoF point-foot quadruped or carries wheel DoFs; visible
   `\todo` markers in `theory/sections/00_notation.tex` and
   `01_assumptions.tex` (Table `tab:joint-signs`). Back-fill after the Day-0
   telemetry audit.
2. **Audit thresholds** τ_dyn, τ_ctrl, τ_env, α_audit — to be frozen in
   `docs/protocol/` before data collection (see theory §audit checklist).
3. **System TeX Live** — run `env/setup_wsl.sh` with sudo, or keep TinyTeX.
4. **Machine yamls** for `y9000p-kaiwen` and `vultr` hold placeholder
   `data_root` values until those machines exist.
5. Gate 1 statistic implementation (e01 / workstream N1-3) — stubs only.
