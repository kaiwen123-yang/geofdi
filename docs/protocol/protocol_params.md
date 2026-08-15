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
