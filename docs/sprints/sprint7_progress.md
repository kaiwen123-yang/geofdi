# Sprint 7 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. Each item: `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint7_spec.md`.

## Anti-loss
- [x] 2509404 spec + progress file committed (`docs: sprint7 spec and progress file`)

## Block 0 — backfills (`chore: liu/sprint6 backfills, decisions D006–D009`)
- [x] 23a69d9 0.1 Liu PDF in `$GEOFDI_DATA_ROOT/lit/liu2025_grufd_ftc.pdf`; provenance = Gazebo simulation in `liu_a1_audit.md`
- [x] 23a69d9 0.2 fault model τ_real = η τ_cmd = `actuator_gain` recorded
- [x] 23a69d9 0.3 official joint order (0–2 LF, 3–5 LH, 6–8 RF, 9–11 RH) corrected in the audit doc
- [x] 23a69d9 0.4 CSV η fields scanned for diagonal double faults; e03 pre-registration class 4 decided
- [x] 23a69d9 0.5 GRU spec (Table I) → `baselines/gru.py` `mode: regression_eta`; `baseline_protocol.md` updated
- [x] 23a69d9 0.6 latency benchmark (~1 s) and the episode-length consequence recorded
- [x] 23a69d9 0.7 real-robot arguments (drift/aging/threshold limitation) → `theory_intake.md`
- [x] 23a69d9 0.8 theory_intake: N3-3 candidate, contamination saturation, centring trap
- [x] 23a69d9 0.9 decisions D006–D009
- [x] 23a69d9 0.10 protocol_params additions (URDF damping conservativeness, torque source, floors, centring trap, wheel-angle exclusion)
- [x] rp015 Block 0 review pack

## Block W — dual hardware readiness → tag `hw-ready`
- [x] 97f4097 W1 wheeled M1 world (`m1_wheeled.xml` / `m1_wheeled_sym.xml`, ctrlrange/forcerange recorded, damping 0.05, IMU site)
- [x] 97f4097 W1 manifest `sim/manifests/m1_wheeled.yaml` (16 joints; WHEEL q excluded; c2 reuse)
- [x] 97f4097 W1 `io/m1_mapping.yaml` + `io/m1_sdk.py` (names reorder, NaN for missing, efforts_semantics)
- [x] 97f4097 W1 t01: sym world 1e-10; original world ε_dyn candidate
- [x] 97f4097 W1 rolling controller `sim/controller_wheeled.py` 0.5/1.0/2.0 m/s × 60 s smoke
- [x] 97f4097 W1 stepping mode tried (kept as `m1_stepping` or skipped with record)
- [x] W-done W2 `phase/registration.py mode: rolling`; H0′ two-sample construction
- [x] W-done W2 e01-W (R=200): three speeds QQ + size table; L ∈ {0.5,1,2} s → minimal exchangeable L; original-world column; ε_ctrl → H0′ size recovery + δ doubling alarm
- [x] W-done W2 nuisance/fault snapshot (R=30) R⁻ timelines
- [x] W-done W2 e13d: equivariant DeLaN on M1 rolling nominal data; residual R⁻ size + wheel motor κ=0.8 power; nuisance readings
- [x] W-done W3 `io/go2_mapping.yaml` + `io/go2_lowstate.py`; synthetic Go2 session ingested to `raw/sim/go2_rehearsal/`
- [x] W-done W3 `phase/estimator.py` (< 5 % cycle error on Go2 sim)
- [x] W-done W4 `scripts/run_pipeline.sh` — M1 synthetic rolling and Go2 synthetic trot sessions run with zero manual steps
- [x] W-done W4 Gate 1 estimator rehearsal (< 30 % error)
- [x] W-done W4 `docs/protocol/m1_day0_wheeled.md` + `docs/protocol/go2_day0.md`; protocol_params L boundary + phase-estimator error
- [x] rp016 Block W review pack; tag `hw-ready`

## Block E — e03 external benchmark + sequential redesign
- [x] E1-done `detect/sequential.py` (e-process / e-CUSUM / conformal-CUSUM; half-cycle elements; calibration ≥ 400 cycles); e04a κ=0.7: median delay ≤ 2 cycles, nominal ARL ≥ 1/α
- [x] E2-done E2 pre-registration committed before the run (incl. diagonal class if present)
- [x] e03-done E2 e03 run: R⁻ half-cycle e-process (raw), R⁺ tracking + Mahalanobis, GRU regression (leave-one-file-out; η 0.4↔0.6; single→double; 3 seeds); per-episode + summary tables + four-class figure
- [~] e03-done E gate: mirrored-bilateral R⁻ reduced (straight 0.50, n=8) not clean ≈ α — reported
- [x] rp018 Block E review pack

## Block T — theory Part 1 full text + Part 2 addenda (tags `theory-part1-v1`, `theory-part2-v1.1`)  [DONE @ e58a5af]
- [x] `02_n1_theorems.tex` replaces the stand-in (labels kept): N1-1 exactness, N1-2 isotypic incl. nonlinear response, N1-5 sequential, A5 chiral prop, H0' exact differenced test
- [x] Part 2: Corollary N3-3, necessity remark, centring trap in the lemma
- [x] bib check; `make theory` zero error / 0 overfull / 0 undefined
- [x] rp017 review pack; tags `theory-part1-v1`, `theory-part2-v1.1` pushed

## Block P — low-SNR full grid (`experiments/e08_low_snr/`)
- [x] e08-done inertia_add {10,20,50} g; noise ×{1,2,4}; full detector set incl. AE + GRU regression (5 seeds); nuisance under three noise levels
- [x] e08-done curves not saturated; merged minimal-detectable table; GRU spread; R⁻ nuisance silence
- [x] rp019 Block P review pack

## Block I — three-channel isolation + anomaly diagnosis (`isolation/three_channel.py`, `experiments/e09_three_channel/`)  [DONE @ d969773; run e09-20260816b]
- [x] LH-KFE friction left/right inversion: root cause (whole-leg residual ENERGY score decreases on a friction-damped leg → max picks wrong leg) + fix (|mean shift| of the pair's joint row). diagnosis LH 0/16 → 16/16; gain 16/16 both
- [x] `docs/protocol/e09_preregistration.md` committed before the run (@ 0cbe7a2)
- [x] confusion (analytic rows acc 0.943 / equivariant rows 0.883), confusion figure, contact-force ±10/20 % sensitivity (0.94 → 0.78/0.62 at ∓10/20 %); single-leg calls robust throughout
- [x] rp020 Block I review pack

## Block S — sequential unification (`experiments/e11_sequential/`)
- [ ] ARL₀ {1/α, 5/α} for conformal-CUSUM / e-CUSUM / e-process (incl. half-cycle); ARL–delay trade-off curve
- [ ] two-channel complementarity figure
- [ ] Block S review pack

## Block N2 — bias augmentation + signatures + rolling contact (`inekf/`, `experiments/e10_n2_signatures/`)  [code @ d969773; e10-20260816 running]
- [x] two augmented InEKF variants: `rinekf_bias.RIEKFBias` (IMU gyro/accel bias + optional 12-dim encoder-bias RW) and `inekf_rolling.RollingRIEKF` (moving contact); equivariance unit tests (tests/test_inekf_bias_equivariance.py, 4 pass, mirror identity to 1e-9)
- [~] signature reconstruction (e10 signatures): encoder-bias innovation matches J[:,j]b (cos≈0.89) + reconstructed; pitch gyro-bias partially recovered (weak observability, yaw unobservable); slip mirror-covariance cos≈1.0 — full 20-seed run in flight
- [x] `docs/decisions/n2_rolling_contact_memo.md` (corrected: A[d_i,R]=0, group-affine) + `inekf_rolling` + M1 wheel-contact FK + NIS smoke; smoke: rolling-InEKF base-pos RMSE 0.35 m vs fixed-foot ~7 m
- [ ] Block N2 review pack (after e10-20260816)

## Block A (extra) — Panda arm (`sim/assets/panda/`, `experiments/e14_arm/`)
- [ ] Panda world; residual + DK table; side-by-side with the welded leg
- [ ] Block A review pack

## Block F (extra) — figure factory (`scripts/make_paper_figures.py`)
- [ ] one-shot regeneration of the 10 figures + 10 tables; figure_plan statuses updated
- [ ] Block F review pack

## Wrap-up
- [ ] tag `sim-milestone-5`; final report (anchor lists, confirmed limitations, hardware-only questions, two first commands, split option status)
