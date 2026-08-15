# Sprint 7 — final report (2026-08-16)

Dual-hardware readiness + external benchmark + low-SNR + three-channel isolation + N2 estimator + theory Part 1 full
text + bonus (Panda arm, figure factory). Every block shipped with a review pack (rp015–rp024) and per-item commits;
`docs/sprints/sprint7_progress.md` is the gate checklist. Iron rules held throughout: data via `$GEOFDI_DATA_ROOT`
only (zero `/mnt/g` literals in code), deterministic seeds, figures ≤1600 px, packs <20 MB, unmet results reported
honestly without tuning, R⁻ detection statistics carry zero trainable parameters.

## Completion

| Block | What | Result | Pack |
|---|---|---|---|
| 0 | Liu/Sprint-6 backfills, decisions D006–D009 | done | rp015 |
| W | wheeled M1 world + Go2 rehearsal → `hw-ready` | done; tag `hw-ready` | rp016 |
| E | e03 external benchmark + sequential redesign (E1 ≤2-cycle delay) | done | rp018 |
| T | theory Part 1 full text + Part 2 N3-3 | done; tags `theory-part1-v1`, `theory-part2-v1.1` | rp017 |
| P | low-SNR full grid (e08) | done | rp019 |
| I | three-channel isolation (e09) + friction inversion fix | done | rp020 |
| S | sequential unification (e11) | done | rp021 |
| N2 | bias-augmented + rolling InEKF + signatures (e10) | done | rp022 |
| A (extra) | Panda-arm residual + DK (e14) | done | rp024 |
| F (extra) | paper figure/table factory | done | rp023 |

Tags this sprint: `hw-ready`, `theory-part1-v1`, `theory-part2-v1.1`, `sim-milestone-5` (final).

## N1 / N2 / N3 anchor lists (theory claim → simulation evidence)

**N1 — distributional Σ-invariance test (the R⁻ channel).**
- N1-1 exactness (flip test size = α): e01 `s1-20260815-1422` QQ/KS in-band; e04b nuisance × channel FAR in the
  binomial band (T1); e03 external `e03-20260816` R⁻ FAR 0.049 ≈ α on the Liu A1 Gazebo data.
- N1-2 isotypic power split (antisymmetric faults visible, Σ-invariant faults blind): e04c raw + e13c residual isotypic
  power (T2); the blind cell is the bilateral-equal fault (R⁻ in-band, R⁺ detects).
- N1-2 nonlinear response + wrapped/unfolded boundary: e04d noise sweep (F3); rem:wrap boundary held.
- N1-5 sequential layer (≤2-cycle delay at valid ARL₀): e03 E1 decimated e-process delay 2.0 cycles / FA 0.08;
  e11 ARL₀–delay trade-off (F4b): conformal-CUSUM ARL₀ 233 / delay 1.5, e-process parameter-free ARL₀ 383 / delay 2.
- Low-SNR degradation: e08 grid (F6b) — analytic-residual R⁻ is the most noise-robust; R⁻ stays nuisance-silent.
- Rolling mode (Σ = G): e01-W plain-H₀ size is anti-conservative at some (speed, L) from slow lateral/yaw modes
  (Assumption-E limitation); deployment uses the exact H₀′ differenced test.

**N2 — invariant estimator / residual channel (the InEKF).**
- N2 NIS as a CFAR channel + fault-signature geometry: e02 `s3-20260815-1734`.
- N2 bias augmentation (equivariance + reconstruction): e10 `e10-20260816` — the augmented state mirrors to 1e-9
  (tests/test_inekf_bias_equivariance.py); a stepped encoder bias's innovation matches the analytic Jacobian direction
  (cos 0.73) and is reconstructed (b̂ 0.085 for a 0.05 injection); a pitch gyro-bias step is partially recovered
  (0.010 — IMU bias weakly observable in flat trot, yaw unobservable); slip mirror-covariance cos 0.993.
- N2 rolling contact (wheeled InEKF): `inekf_rolling.RollingRIEKF` — the moving contact is group-affine (A[d_i,R]=0,
  verified), and the rolling model is required to track (base-pos RMSE 0.71 m vs fixed-foot 13.4 m on m1_wheeled_sym).

**N3 — isolability (the residual / three-channel + DK certificate).**
- N3 DK isolability certificate ↔ nearest-subspace confusion: e06 `e06-20260815-2003` (analytic class acc 1.00 on the
  floating base; welded leg = arm 0.91) with the calibration-centring lemma.
- N3-1/N3-2 residual inherits H₀; N3-3 equivariant-model error is Π⁺-only: e13c isotypic isolation (T2, T9).
- Three-channel isolation (R⁻ state + joint residual rows + base momentum rows): e09 `e09-20260816b` — analytic-row
  accuracy 0.943, contact-wrench sensitivity 0.94 → 0.78/0.62 at ∓10/20 %; the LH-KFE friction left/right inversion
  fixed (|row mean shift|, LH 0/16 → 16/16).
- N3 symmetry-free generalisation to a manipulator: e14 `e14-20260816` — Panda 7-DoF arm, residual + DK isolation
  **class acc 1.00, joint acc 1.00, DK agreement 0.95, β² 0.47, floor RMS 0.18**, matching the floating-base Go2 (1.00)
  and beating the welded 3-DoF leg (0.91, DK agreement 0.81) — the arm's joints have better-separated torque
  signatures. Confirms the model-residual (R⁺) + N3 certificate is symmetry-free (no C₂ needed).

## Confirmed limitations (honest, un-tuned)

1. **Rolling-mode plain H₀ is anti-conservative** at several (speed, L) from positive block dependence (slow
   lateral/yaw modes, lag-1 up to +0.47); neither k_yaw=1 nor 50 fixes all. Documented as an Assumption-E limitation;
   the exact H₀′ differenced test is the deployment path (e01-W).
2. **Mirror double-blindness on external data is not clean** (e03 straight-segment R⁻ 0.50, n=8): a small-n + stable
   gait-asymmetry effect, reported with the mechanism rather than tuned away.
3. **IMU gyro-bias reconstruction is partial** (e10): ~50 % of a pitch bias recovered in short flat trot; yaw is
   InEKF-unobservable. Reported as an observability result, not a clean reconstruction. The encoder-bias reconstruction
   over-estimates ~1.7× (direction is the strong claim).
4. **e11 R⁻ fires on the front-pair bilateral onset-transient** (legs half a cycle apart when the step lands), so the
   clean "R⁻ blind to symmetric faults" story is only exact for a phase-synchronised bilateral onset; the timeline
   figure is the honest view.
5. **e08 GRU cannot extrapolate below training magnitudes** (det 0.00); only R⁻ is nuisance-silent.
6. **e09 symmetric-drift vs nominal** at low drift magnitude (10/30 → nominal) — the smallest-drift seeds are genuinely
   near-nominal.
7. **Rolling NIS is conservative** (over-modelled covariance); tracking RMSE, not per-bin FAR, is the discriminator in
   the wheeled NIS smoke.

## Hardware-only questions (cannot be settled in sim)

- Do the M1 legacy wheeled bags (`raw/m1/legacy-aug`) confirm the rolling-InEKF and the rolling-mode H₀′ test on real
  tire compliance / slip? (The memo names them the natural first rolling test set.)
- Real symmetric actuator drift / aging: is the e09 symmetric-drift operating point (τ_s, magnitude) representative,
  and does Gate 1's inability to resolve loaded-loop kp asymmetry matter on hardware?
- Real IMU/encoder bias magnitudes and excitation: does gyro-bias observability improve with richer real motion?
- The unverified M1/Go2 name/index mappings (`io/*_mapping.yaml`, `unverified: true`) — Day-0 verification.
- Contact-force sensing accuracy: e09's payload/drift boundary moves under ±10/20 % wrench error; what is the real
  sensor error?

## First commands on each robot (Day 0)

```
# wheeled M1, rolling mode
scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/m1/nominal/<session> --robot m1 --mode rolling

# Go2, trot
scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/go2/nominal/<session> --robot go2 --mode trot
```

Both robots have: name/index-based loaders + mappings (unverified until Day 0), a kinematic phase estimator
(<5 % cycle error in sim), one pipeline (`run_session.py`), and Day-0 protocols (`docs/protocol/{m1_day0_wheeled,
go2_day0}.md`).

## Split-option status

Unchanged from the Sprint-6 decision (default: one T-RO long paper; RA-L manipulator paper is a ready fallback), with
**one strengthening**: Block A (e14) extends the RA-L "leg = arm" story from the 3-DoF welded Go2 leg to a real 7-DoF
Franka Panda arm through the identical residual + DK code. The RA-L half's claim (iii) — DK isolability certificate vs
nearest-subspace confusion on a fixed-base manipulator — now has a genuine manipulator anchor, not only the welded leg.
The RA-L half still needs the e13a/e13b power + size-vs-δ_f runs on the weld world (the one remaining `e06 TODO`);
everything else in the split plan holds.
