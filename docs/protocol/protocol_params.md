# Protocol parameters (collected from S1/S2 simulation evidence)

Single place for the numerical choices of the H0 mirror-test machinery, with the experiment that justifies each.
Values are for the S0 Go2 world (kp 80 / kd 2 / yaw damper 0.2, S1 baseline noise) unless stated; hardware
values will be re-derived on M1 audit data (Gate 1) but the *structure* of the advice carries over.

| parameter | value | justification |
|---|---|---|
| phase grid N | 64 | S1 e01a: size in band; finer grids do not change p-values (registration truth phase) |
| warm-up cycles dropped | 10 | S1: transient of the trot start-up |
| permutations M | 512 | Hemerik–Goeman random subset incl. identity, p = (1+#)/M; e01a size in band |
| per-channel standardization | flip-invariant pooled std of the *calibration* cycles | e04e: pooled-over-post standardization inflates low-variance HAA channels (wrong isolation ranking) |
| R⁻ window (single-window FAR tables) | 20 cycles | 2^19 sign patterns: p-values effectively continuous; e04b nominal FAR 0.052–0.054 |
| R⁻ window (e-process / e-CUSUM) | 5 cycles | only 16 distinct sign patterns → p ≥ ~1/16 → e = p^{-1/2}/2 ≤ ~2 per window; e-process alarm has a hard floor of 5 windows = 25 cycles and a single 5-cycle window can never reject at α = 0.05 (e04a/e04c). Use windows ≥ 9 cycles (256 patterns) whenever a per-window rejection rate is to be compared with a binomial band |
| e-value | e = p^{-1/2}/2 (calibrator) | S1/S2; e-process = running product, alarm at 1/α (Ville) |
| e-CUSUM threshold h | calibrated by block bootstrap on nominal p-values (horizon 20 windows, FAR 0.05): h = 1.81 (paired energy), 2.12 (energy distance) | e04a `e04a_ecusum_thresholds.csv`; the calibrated e-CUSUM detects κ ≤ 0.9 within 15 cycles (3 windows) with det20 = 1.0, whereas the uncalibrated e-process needs 25 cycles |
| block length for merged out-and-back slope data | 2 (one +5° and one −5° cycle per block) | e04b: single-cycle flips of the interleaved ± sequence are over-conservative (FAR 0.000, alternating asymmetry cancels); paired blocks give 0.069/0.075 (band [0.037, 0.064]) |
| wrapped (within-cycle) test — noise boundary | valid (size 0.060–0.075, band [0.02, 0.08]) up to actuator noise 0.10 N·m; out of band (0.095) at 0.20 N·m | e04d, R = 200, K = 60 cycles |
| unfolded (true half-period shift) test | in band at all tested noise levels (0.040–0.065); costs 3 cycles per element (K/3 elements) | e04d; use it when the noise regime is dynamics-dominated (S1 Finding 4) |
| R⁺ (magnitude) channel | per-cycle tracking-error score, conformal p against the calibration cycles | e04b: in band under speed / symmetric payload (0.045–0.050), **inflated under symmetric drift (0.19)** — the magnitude channel must be recalibrated per operating condition or use a drift-robust score; it is *not* nuisance-invariant by construction (only R⁻ is) |
| isolation | (pair, joint) ranking by nominal-scaled R⁻ projection energy, swing-phase conditioning, left/right resolved by the R⁺ per-leg deviation (pooled pair std) | e04e: target LF-KFE recovered 50/50 |
| onset cycle / calibration length | 60 calibration cycles (after warm-up), 100 monitored cycles | e04a–e |
| replicates | R = 50 (power), 100 (nuisance FAR), 200 (size boundary) | binomial band widths ±0.02–0.04 at α = 0.05 |

Seeds: e04a 11000+, e04b 12000+, e04c 13000+, e04d 14000+, e04e 15000+ (replicate index added); the sim itself is
deterministic given the seed (MuJoCo 3.11, integer phase clock).

## Residual channel (Sprint 6, e13; theory Part 2)

| parameter | value | justification |
|---|---|---|
| residual data element | 12 joint rows of r = τ_cmd + Jᵀf_c − f̂ (analytic momentum observer at 10 Hz, or DeLaN), phase-registered on the same N = 64 grid; ρ_R = torque signs/partners (`geofdi.residuals.mirror_pairs`) | Part 2 Def. def:residual-element; the same Hemerik–Goeman flip test acts on it (Prop N3-1) |
| nominal model for R⁻ on residuals | **equivariant** (analytic observer under A1, or DeLaN with mirror weight sharing, δ_f = 0) | e13b: a plain per-leg DeLaN residual has size 1.00 at K = 60 already for δ_f^{(0.95)} = 0.67 N·m; equivariant / analytic residual sizes 0.015–0.06 (band [0.02, 0.08], URDF world conservative) |
| H₀′ on residuals (non-equivariant model) | difference each monitored cycle with an independent calibration cycle and flip-test the paired-energy statistic; **never** subtract an estimated mean profile and re-run the exact test | e13b: naive centring size 1.00 for every model; differenced test 0.00–0.045 for every model incl. δ_f^{(0.95)} = 14.8 N·m (Part 2 Lemma centring (iii)/(iv)) |
| e-CUSUM threshold per R⁻ variant | calibrated separately on the variant's own pooled nominal windows (raw 1.07–1.33, analytic residual 1.47–1.56, equivariant residual 1.37–1.53 in e13a/e13c) | the nominal p-value law differs per element; a plain-DeLaN residual calibrates to h ≈ 14 (its nominal windows already reject) — a diagnostic of contamination |
| low-SNR minimal detectable severity (det100 ≥ 0.9, R = 50) | residual R⁻: bias 0.10 N·m, gain 1−κ 0.02, friction ×1.5; raw R⁻: bias 0.5, gain 0.05/0.02, friction undetected on the grid; Mahalanobis (magnitude): 0.05 / 0.01 / ×1.5 | e13a `e13a_min_detectable.csv` |
| R⁺ on residuals | conformal magnitude on Π⁺r (pure trivial component) or on the full residual energy; the full-energy score is more powerful for single-leg faults (half of a one-joint footprint is antisymmetric) | e13a: `Rplus_res_an_full` min-detectable gain 0.02 / bias 0.10 vs `Rplus_res_eq_sym` 0.10 / 0.5 |
| three-channel isolation | pair–joint from the R⁻ projection energy on the residual element (swing conditioning, calibration scale) → left/right from the per-leg residual score deviation → payload if the base-row f_z shift ≥ 3σ | e13c: raw+tracking 0.57, analytic rows 0.86 (LH-KFE friction attributed to RH-KFE by the analytic rows), equivariant rows 1.00 (7 classes × R = 20) |

## Sprint 7 Block 0 additions (2026-08-16)

| parameter | value | justification |
|---|---|---|
| URDF-world damping 0.01 → conservative flip test | size 0.02–0.025 at α = 0.05 (KS 0), 2-cycle under-damped pitch/knee mode; `joint_damping: 2.0` restores uniform p | D005 / Block G; state which is used in every experiment (default: accept the conservative size, report the band) |
| torque source for residuals | commanded torque τ_cmd (+ Jᵀf_c); τ_meas is physically consistent and blind to actuator gain/bias | Sprint 4 L1 Finding 1; Part 2 Prop N3-2 (a) |
| floors of the sequential layer | e-process on per-cycle conformal p: ≥ 3 cycles (three e ≈ 61/… products to reach 1/α); R⁻ 5-cycle windows: alarm floor 25 cycles for the plain e-process, ~10 for the calibrated e-CUSUM; a per-window rejection rate is not usable below 9-cycle windows | e04a/e05b/e13c; Block E redesigns the layer (half-cycle elements, ≥ 400 calibration cycles) |
| centring trap | never subtract an estimated calibration mean profile before the exact flip test (size ≈ 1); H₀′ = differencing monitored vs calibration cycles, or the two-sample construction | e13b; Part 2 Lemma centring (iii)/(iv) |
| wheel-angle exclusion (M1 wheeled) | wheel joint angles are unbounded (rolling) and are excluded from the data element (`in_Z: false`); wheel rates and efforts stay, transformed as pitch-axis pseudovector rates (sign +) | D009; manifest `sim/manifests/m1_wheeled.yaml` |

## Sprint 7 Block W (hardware readiness) additions (2026-08-16)

| parameter | value | justification |
|---|---|---|
| rolling-mode data element | fixed-duration blocks, L = 1.0 s default (N = 64), cut on the command plateau after a 6 s warm-up (3 s speed ramp); Σ = G (shift 0) | e01-W stage a: size vs L (see W pack for the minimal exchangeable L) |
| M1 sim world contact model | `cone="elliptic"` in the M1 scenes; wheel `solref 0.05` | the default pyramidal friction cone is anisotropic in the world frame: a wheel rolling at a heading angle picks up a heading-dependent lateral force → persistent lean → the flip test rejects nominal runs at heading ≠ 0 (W2 finding); tire compliance removes wheel chatter at ≥ 1 m/s |
| kinematic phase estimator | `phase/estimator.py`: diagonal knee signal s = q_LF + q_RH − q_RF − q_LH, band-pass ±50 % around f0, **piecewise-linear phase clock** (10-cycle local fits of the unwrapped Hilbert phase), template origin φ0 = −1.4977 (calibrated on go2_urdf_sym); error vs the controller clock: 0.01–3.4 % of a period (constant offset, speed-dependent), jitter < 0.1 %; drop the last 5 cycles (Hilbert edge transient) | W3; the raw Hilbert phase warps within the cycle in a non-equivariant way and inflates the H₀′ differenced test (p 0.002 → 0.2–0.9 with the linear clock) |
| registration with an estimated phase | `register_cycles` interpolates on the global unwrapped phase (no clamped cycle edges) | a fractional phase offset with clamped edges gave a systematic one-grid-point half-cycle asymmetry (p 0.002 on nominal data); identical results for the controller clock |
| Gate-1 estimator | mirrored-command gap per torque channel = |mean_k mean_θ (τ_leg(θ) − s τ_partner(θ+½))| with bootstrap CI (`detect/gate1.py`) | rehearsal: wheel-rate 1.02 recovered within 1 %, Go2 HFE kp 1.05 within 26 % (median), M1 HIP kp 1.02 not resolvable (the loaded loop absorbs a gain change: command gap 0.011 vs counterfactual 0.12 N·m) — Gate 1 sees the closed-loop command asymmetry, not the controller's parameters |
| Day-0 first commands | `scripts/run_pipeline.sh <session> --robot m1 --mode rolling` / `--robot go2 --mode trot` | W4; both zero-touch on the synthetic rehearsal sessions |

## Sprint 9 Block B2 additions (2026-08-16) — robustness sweeps (`experiments/e19_robustness`, run `e19-20260816`)

Three standard reviewer probes, all on the nominal `go2_urdf_sym` world under H₀, R = 60 per point, α = 0.05.

| probe | setting | H₀ / H₀′ size | reading |
|---|---|---|---|
| **phase-clock error** | ±2, ±5, ±10 % of the period | 0.000 at every nonzero error (0.033 / 0.050 at 0 %) | **The level survives; the cost is power, not FAR.** A mis-scaled phase clock decorrelates the mirror pairing, so the flip test becomes *conservative* rather than anti-conservative. Practical rule: a phase estimate good to ±10 % cannot create false alarms; quantifying the power it costs needs a separate sweep (owed). |
| **H₀′ calibration size** | K_cal ∈ {60, 200, 400}, K_mon = 60 | 0.083 / 0.033 / 0.000 | In band at all three; the size falls (more conservative) as the calibration set grows. **K_cal = 60 is already sufficient** for the level; larger sets buy stability of ν₀, not validity. |
| **block length vs nuisance correlation time** | B/τ ∈ {0.5, 1, 2} under `drift_lateral` (mirror-symmetric in law, τ = 4 cycles) | **0.350 / 0.217 / 0.100** | The only probe that breaks the level. **Protocol number: the flip block must be at least 2× the nuisance correlation time** (B/τ = 2 is the first point inside the band; B/τ = 1 is still 4× α). This supersedes the looser Sprint-1 statement "block 8 gives 0.075, block 16 is in band" by expressing the requirement in units of τ. |

Consequence for deployment: of the three things a practitioner can get wrong, only the block length threatens the false-
alarm guarantee. Estimate the nuisance correlation time (lag autocorrelation of the per-element anti-symmetric energy)
and set the block to ≥ 2τ; then a phase estimate within ±10 % and K_cal ≥ 60 are enough.
