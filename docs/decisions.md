# Decision log

One entry per hard-to-reverse decision. Format: `D<nnn> — <title> — <date> — <status>`.

## D001 — Naming conventions — 2026-08-15 — accepted

- Repo and Python package: `geofdi` (src layout).
- Experiments: `e<NN>_<slug>` (e.g. `e01_h0_qq`); every run writes to
  `results/<exp>/<run_id>/`, `run_id = YYYYMMDD-HHMMSS` unless given explicitly.
- Raw sessions: `<YYYYMMDD>_<gait>_<speed>_<terrain>_<direction>_repNN`
  (e.g. `20260901_trot_0.6_flat_north_rep01`).
- Review packs: `rp<NNN>_<YYYYMMDD>_<slug>.zip` in `review/outbox/`.
- Theory notes: `theory/sections/NN_<slug>.tex`, one numbered part per milestone.

## D002 — Data layout & machine portability — 2026-08-15 — accepted

- The git repo lives on the Linux filesystem (`~/research/geofdi`); bulk data lives on a
  per-machine data volume addressed ONLY through `$GEOFDI_DATA_ROOT`.
- Per-machine roots are declared in `env/machines/<machine>.yaml`;
  `scripts/setup_paths.sh` resolves the machine (env var → /proc/version → hostname) and
  (re)creates the repo-local symlinks `data/` and `results/`.
- No literal mount paths in code/config; the only allowed occurrences are the machine
  yamls, the README, and comments inside the setup scripts.
- git tracks text only; `data/` and `results/` are gitignored symlinks.
- Raw data is immutable by convention + `checksums.sha256`; filesystem-enforced
  immutability arrives with the migration to native Ubuntu/ext4.

## D003 — Formalization of H0 — 2026-08-15 — accepted

- H0 is *distributional* Σ-invariance of phase-registered per-cycle functional data
  elements: Law(ρ(σ) Z_k) = Law(Z_k) for every σ in the gait's spatio-temporal symmetry
  group Σ ⊂ G × S¹ — not pointwise trajectory symmetry, and not a parametric residual
  model.
- Rationale: (i) tolerates stochastic controllers, terrain, and sensor noise;
  (ii) makes permutation tests and group-invariance e-tests exactly valid without any
  dynamics model of the fault; (iii) faults appear as distribution asymmetries, matching
  the isolation story via isotypic components.
- Cross-cycle exchangeability is deliberately kept OUT of H0: it is a separate
  maintained assumption required by the test machinery (workstream N1-1).
- ε-approximate variants of A1–A4 are measured in total-variation distance; type-I
  inflation is additive (ε-robustness lemma, theory Part 0 §5).
- If Gate 1 shows the vendor controller is measurably asymmetric, the deployed null
  becomes H0′: asymmetry-*change* detection against a calibration baseline.
