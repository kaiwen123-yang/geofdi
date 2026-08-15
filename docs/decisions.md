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
- Implementation note (2026-08-15): data root provisioned under `$GEOFDI_DATA_ROOT` with
  `data/raw/{m1/{legacy-aug,audit,nominal,nuisance,pilot-fault},public/{liu-a1-fault,street-a1,legkilo-go1},sim}`,
  `data/processed`, `results`, `models`, `lit`, `scratch`, `review/{outbox,archive,feedback}`;
  repo symlinks `data/`, `results/` created and verified by `scripts/setup_paths.sh`.

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

## D004 — Simulation world for S0/S1 (Go2, headless MuJoCo) — 2026-08-15 — accepted

- Model: Menagerie Unitree Go2 (commit `da76818e`) vendored mesh-free (`go2_sym.xml`), with two
  symmetrization edits (FL calf collision cylinder; base inertia ixy/iyz projected out) so that the
  dynamics are mirror-equivariant to floating point (t01: mirror-sim agrees to 1e-10). Actuators are
  torque motors; the joint convention is uniform-axis (S = diag(-1,+1,+1)).
- Controller: phase-driven PD trot, Σ-equivariant by construction (one left-leg template; RF/RH from
  LF/LH by S and a half-period shift). Reference is C¹ (sin² lift) and the phase is an integer clock:
  a discontinuous reference velocity sampled exactly at the stance/swing transition (float ambiguity in
  θ = t/T) produced a systematic mirror asymmetry at two grid points — a lesson worth remembering.
- Gait parameters: kp = 80, kd = 2, lift (KFE 0.45, HFE 0.20), period 0.5 s, speed 0 (trot in
  place). At kp = 60 with the same lift the Σ-symmetric orbit is *unstable* and the trot settles into
  a chiral limit cycle (spontaneous symmetry breaking: A1–A4 hold, H0 is false for the realized law).
  H0 therefore also needs the symmetric orbit to be the unique attractor (theory: ergodicity caveat).
- Data element Z: q, dq, tau_cmd, tau_meas (12 each), IMU a/w (body frame), contacts — 58 channels.
  The temperature surrogate stays in the telemetry but is **not** in Z (`in_Z: false`): it is a slow
  monotone nuisance whose within-cycle drift makes Z(θ) − Z(θ+½) systematically nonzero.
- Registration: controller truth phase, N = 64 grid, first 10 cycles discarded as warm-up.
- Test: Hemerik–Goeman random-subset flips over cycles (identity included, p = (1+#)/M, M = 512),
  statistics = paired-difference L2 energy and mirror energy distance, per-channel standardization
  by the flip-invariant pooled std.
