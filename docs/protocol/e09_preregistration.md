# Pre-registration — e09: three-channel isolation (Sprint 7 Block I)

Committed before the run. Go2 `go2_urdf_sym`, 20 warm-up + 60 calibration + 100 monitored cycles, onset cycle 60,
R = 30 per class. Reading vector (`geofdi.isolation.three_channel.readout`, nothing trained):
1. **R⁻ state**: alarm (FAR-calibrated e-CUSUM on the equivariant-DeLaN residual element) and, if alarmed, the ranked
   (pair, joint) by the R⁻ projection energy (swing-conditioned, calibration scale).
2. **Joint residual rows**: per-(leg, joint) standardized mean shift of the equivariant-DeLaN residual; leg energy
   share (0.25 even … 1 single leg); left/right within the R⁻ pair by the **|mean shift| of the pair's joint row**
   (Block-I fix — see the diagnosis below).
3. **Floating-base rows**: standardized mean shift of the 6 base momentum-residual rows (analytic observer); f_z →
   payload, m_x → lateral offset.

## Diagnosis committed before the run (the e13c analytic-row LH-KFE friction inversion)
Root cause (verified, `results/e09_three_channel/diagnosis.md` after the run reproduces it): the e13c `analytic_rows`
left/right rule used the per-leg residual **energy** score `residual_scores(per_leg=True)` and took `max(signed
deviation)`. A friction increase **damps** the faulty leg's motion, so its whole-leg residual energy **decreases**
(deviation goes negative) — `max` then selects a non-faulty leg. Across 16 seeds the friction fault on LH-KFE gave a
per-leg score deviation with LH the most negative (−0.32) → RH picked. The fix (this module) resolves left/right by the
**|mean shift| of the specific joint's residual row**, which is monotone in the fault for gain (torque loss raises the
row), bias (constant offset) and friction (Coulomb term): LH is picked 16/16. A gain fault raises the energy so `max`
happened to work for it; the row-shift rule works for all three types.

## Hypothesis classes (R = 30)
| class | injection | expected label |
|---|---|---|
| single_gain | LF-KFE κ = 0.8 | single_leg:LF-KFE |
| single_bias | LF-HFE b = 0.5 N·m | single_leg:LF-HFE |
| single_friction | LH-KFE ×2 | single_leg:LH-KFE (the diagnosis target) |
| bilateral_mirror | LF-KFE & RF-KFE κ = 0.8 | bilateral_mirror (R⁻ silent) |
| payload_lateral | 0.5 / 1 kg, offset 0.05 m | payload_lateral (base fz & mx) |
| payload_symmetric | 1 kg centred | payload_symmetric (base fz only) |
| calf_inertia | LF calf +100 g | single_leg:LF-KFE via the N3 inertia class (reported: inertia ≈ a mass/Coriolis change on the calf) |
| drift_symmetric | OU σ 0.10 | symmetric_drift_or_bilateral (R⁻ silent, spread rows) |
| nominal | — | (no alarm) |

## Decision rules (fixed a priori; `three_channel.decide`)
- base f_z & m_x ≥ 3σ → payload_lateral; base f_z ≥ 3σ, m_x < 3σ → payload_symmetric.
- R⁻ silent + joint rows quiet, leg share < 0.4 → symmetric_drift_or_bilateral; share ≥ 0.4 → bilateral_mirror.
- R⁻ alarmed + leg share ≥ 0.5 → single_leg:(resolved leg)-(resolved joint); share < 0.5 → pair:(pair)-(joint)
  (left/right low-confidence).

## Products
7-class (+ nominal, drift) confusion for **two** R⁻/row sources — analytic-observer rows and equivariant-DeLaN rows —
side by side with e13c's original (broken) analytic-row result; a per-class three-row readout figure; a **contact-force
sensitivity table** (the momentum residual's contact wrench scaled by 1 ± {0.10, 0.20} to emulate an estimated wrench).

## What would falsify
Single-friction still mis-resolved after the fix → the row-shift rule is insufficient (report the residual signature);
payload not separated by the base rows → the contact wrench is not clean enough; drift labelled as a single-leg fault →
the leg-share threshold is wrong. No tuning after seeing the results.
