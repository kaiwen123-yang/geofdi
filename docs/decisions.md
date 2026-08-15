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

## D005 — Simulation worlds after Block G (go2_description URDF) — 2026-08-15 — accepted

- Three Go2 worlds exist (`geofdi.sim.env.MODELS`): `go2_menagerie_sym` (S0/S1–S3 baseline, kept for reproduction),
  `go2_urdf` (Unitree go2_description URDF converted by `geofdi.sim.urdf2mjcf`, every number from the URDF incl. the
  base products of inertia ixy = 1.2166e-4, iyz = −3.12e-5 and the FL-calf collision chirality) and `go2_urdf_sym`
  (same, mirror-symmetrized: base I ← (I+EIE)/2, left-leg collision primitives = mirror of the right ones).
- **Default world for new experiments: `go2_urdf_sym`** (true URDF inertias, mirror-exact: t01 7e-11). `go2_urdf` is
  the A1′ audit-rehearsal world (t01 fails at 4.9e-3 = ε_dyn; e01a size unchanged under S1 noise, e01c H₀′ size 0.028
  in band). The menagerie world stays the reproduction baseline for S1–S3.
- The URDF world uses the URDF/xacro joint dynamics (damping 0.01, frictionloss 0.2; armature 0.01 as in menagerie).
  Consequences measured in Block G: the symmetric trot orbit is stable but attracts slowly (half-period mirror residual
  1e-3 rad at 5 s, 6e-5 at 10 s) → **20 warm-up cycles** in the URDF worlds (10 in the menagerie world); an
  under-damped body-pitch / hind-knee mode with a period of ≈ 2 gait cycles makes consecutive mirror differences
  negatively correlated (lag-1 −0.03 overall, −0.11/−0.15 in hind-knee q/dq/τ and pitch rate) → the per-cycle flip
  test is *conservative* (size 0.020/0.025 at α = 0.05, KS p 0.000; also for the unfolded statistic and block-4
  flips). With joint damping 2 (`SimConfig.joint_damping`, diagnostic) the same world gives uniform p (sizes
  0.055/0.060). This is an exchangeability (A-exch) effect, not a symmetry effect; the flip test stays valid
  (conservative). Experiments that need nominal size in the URDF world either use `joint_damping: 2.0` explicitly or
  accept the conservative size — state which.
- The generic converter (`urdf2mjcf.py`) is the path for the M1 URDF/MJCF (Block M).

## D006 — e05a re-characterized: every magnitude channel inflates under symmetric drift; only R⁻ is silent — 2026-08-16 — accepted

- e05a (rp008) showed rplus_resid (τ_cmd and τ_meas) at 0.14–0.16 per-cycle FAR under the S2 symmetric torque/friction
  drift (0.09 with K_cal = 200), rplus_track 0.19–0.23; e07 (rp009) showed Mahalanobis / AE alarming under the same drift;
  only the R⁻ mirror channel keeps its FAR. The L1 gate item "rplus_resid back in band under symmetric drift" is
  therefore not a bug to fix but a property: magnitude channels calibrated on a short window are not nuisance-invariant
  by construction, the invariance channel is. Deployment consequence: R⁻ is the FAR-guaranteed channel; every magnitude
  channel needs recalibration per operating condition (or the H₀′-style two-sample comparison) and its FAR is reported
  per nuisance (Block P re-runs the nuisance rows under three noise levels).

## D007 — Hardware M1 is the wheeled-legged `zgws` platform; the point-foot candidate is retired — 2026-08-16 — accepted

- The only M1 hardware available (August legacy bags, vendor SDK) is the wheeled-legged 16-joint robot (fl/fr/bl/br ×
  hip_roll, hip_pitch, knee_pitch, wheel). The point-foot STEP-derived candidate model (`assets/m1`, rp010) is retired:
  no further effort on the STEP, no point-foot M1 experiments. The M1 world of record is `m1_wheeled(_sym).xml` built
  from the MATRiX `zgws` MJCF (Sprint 7 Block W1); meshes stay in `~/research/third_party` (not vendored).

## D008 — Milestone numbering: sim-milestone-2/3 are folded into 4; no retroactive tags — 2026-08-16 — accepted

- Sprints 2 and 4 ended without their milestone tags (gate misses documented in rp005/rp006 and rp008). Their delivered
  content is covered by `sim-milestone-4` (Sprint 6). Tags 2 and 3 are not created retroactively; the next milestone is
  `sim-milestone-5` at the end of Sprint 7. Theory tags continue per part (`theory-part1-v1`, `theory-part2-v1.1`).

## D009 — Dual hardware programme: Go2 point-foot (trot) + M1 wheeled-legged (rolling / stepping) — 2026-08-16 — accepted

- Go2: trot with Σ = {(e,0), (g_s, ½)} ⊂ G × S¹ — the spatio-temporal null of Part 0; data via CycloneDDS LowState
  (Unitree joint order FR, FL, RR, RL × hip, thigh, calf); phase from the kinematic estimator (`phase/estimator.py`).
- M1 wheeled: rolling mode with Σ = G (pure sagittal reflection, no phase; fixed-duration blocks as data elements,
  `mode: rolling`) and, if the stepping gait is stable, a G × S¹ trot-like mode. Wheel angles are unbounded and excluded
  from the data element (wheel rates and efforts stay). Legacy August bags (wheeled driving) become the first nominal
  corpus for the rolling mode once the SDK channel semantics are fixed on Day 0.
- Both robots run through one pipeline (`scripts/run_pipeline.sh --robot m1|go2 --mode rolling|trot`), each with its own
  mapping yaml (`io/m1_mapping.yaml`, `io/go2_mapping.yaml`, both `unverified: true` until Day 0).
