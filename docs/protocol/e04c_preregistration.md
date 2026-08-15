# Pre-registration — e04c isotypic prediction (Sprint 2)

Committed before the experiment is run (see git log for the timestamp of this file vs. the results run id).

## Setting
Simulated Go2 (S0 world, closed-loop trot with the equivariant yaw damper, S1 baseline noise), fault onset at
cycle 60 (after 10 warm-up cycles), 100 monitored cycles after onset, R = 50 replicates per group.
Faults are actuator gain reductions on the calf (KFE) joints:

| group | fault | Σ-orbit of the fault pattern |
|---|---|---|
| G1 single-leg | LF-KFE κ = 0.7 | breaks the mirror: pattern ≠ its mirror image |
| G2 bilateral equal | LF-KFE κ = 0.7 and RF-KFE κ = 0.7 | mirror-invariant pattern (fixed by σ_*) |
| G3 bilateral unequal | LF-KFE κ = 0.7 and RF-KFE κ = 0.5 | mirror image has κ swapped: 0.5/0.7 |

Two detection channels, both calibrated on the 60 pre-onset nominal cycles only:
- **R⁻ (mirror channel)**: Hemerik–Goeman mirror test (paired-difference energy, window 5 cycles, M = 512) → e-values
  → e-process alarm at 1/α (α = 0.05). Sees only the antisymmetric (sign-representation) content of the signature.
- **R⁺ (magnitude channel)**: per-cycle tracking-error score Σ_legs ‖q − q_ref‖ (phase-binned L2; mirror-invariant),
  conformal p against the calibration cycles → e-process alarm at 1/α. Sees the trivial-representation content.

## Predictions (theory Part 0, Example *bilateral* and the isotypic preview)
1. **G1** (single leg): R⁻ detects (power ≫ α within 100 cycles) **and** R⁺ detects.
2. **G2** (bilateral, equal η): the faulty loop is still Σ-equivariant, so H₀ still holds — the R⁻ channel is
   *blind*: its post-onset per-window rejection rate stays inside the binomial band of α = 0.05, and its e-process
   alarm fraction stays ≤ α; **R⁺ detects**.
3. **G3** (bilateral, unequal η): the antisymmetric content is the *difference* 0.7 − 0.5 = 0.2 in gain, i.e.
   smaller than G1's 0.3 → R⁻ power strictly between G2 (≈ α) and G1; R⁺ detects (its content is the *sum*).

Quantitative reading of "detects": e-process alarm fraction ≥ 0.9 within 100 cycles at α = 0.05.
Projection-energy prediction: the standardized mean post-onset deviation Δ decomposes into Δ⁻ = (Δ − ρΔ)/2 and
Δ⁺ = (Δ + ρΔ)/2; the antisymmetric energy share ‖Δ⁻‖²/(‖Δ⁻‖²+‖Δ⁺‖²) is largest for G1, ≈ 0 for G2, intermediate for G3.

## What would falsify
- G2 R⁻ alarm fraction clearly above α → either the R⁻ statistic is not purely antisymmetric or the bilateral
  fault is not Σ-invariant in the sim (e.g. load redistribution that is itself asymmetric) — report as such.
- G3 R⁻ power ≥ G1's → the "difference" heuristic is wrong (nonlinear coupling) — report as such.
No parameter will be tuned after seeing the results; deviations are reported as findings.
