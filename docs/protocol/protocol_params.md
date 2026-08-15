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
