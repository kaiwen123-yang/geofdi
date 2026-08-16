# Split option: RA-L manipulator paper + T-RO main paper

Default plan: one T-RO long paper (outline.md). If the decision is to split, this is the minimal content set of each
half and what changes.

## RA-L short paper — "Equivariant residuals for fault isolation on a fixed-base arm (leg = 3-DoF manipulator)"

Minimal content (all of it exists in simulation today):
- Setting: welded-trunk Go2 leg = 3-DoF fixed-base manipulator with a mirror twin (LF/RF): the group is C₂ acting on
  the pair; the "gait" is the periodic reference; H₀ = mirror invariance of the residual data element.
- Nominal models: analytic inverse dynamics (Pinocchio) and DeLaN, plain vs **equivariant by mirror weight sharing**
  (Part 2 Prop equiv-delan; Block Q weld models `weld_plain_v1` / `weld_equiv_v1`: δ_f 1.60 vs 0 N·m, val RMSE 0.625
  vs 0.573).
- Claims: (i) residual inherits H₀ (N3-1) → exact flip test on Π⁻r for the pair; (ii) non-equivariant learner
  contaminates H₀ (ε̄_model) — **run on the weld world (Sprint 8 L3, e13 weld addendum, R=50, `results/e13_residual_symmetry/
  e13weld-20260816`): H₀ flip-test size = raw 0.04 / analytic residual 0.02 / equivariant DeLaN 0.00 (δ_f=0) — all in the
  band [0, 0.12] — vs plain DeLaN 1.00 (δ_f q95 = 1.60); the H₀′ differenced test restores the plain model to 0.04 while
  naive mean-centring stays at 1.00 (calibration-centring Lemma: centring does NOT fix it)**; (iii) DK isolability
  certificate vs nearest-subspace confusion (e06 weld: analytic accuracy 0.91 / DeLaN-weld 0.80; rp008) with the
  calibration-centring lemma; (iv) power gain of the residual channel on the weld world — **run (e13a weld, R⁻ H₀′ power
  vs 1−κ on LF-KFE actuator gain): analytic residual 0.14 / 0.60 / 1.00 / 1.00 at 1−κ = 0.01 / 0.02 / 0.05 / 0.10, above
  the raw signal (0.14 / 0.24 / 0.94 / 1.00); the equivariant DeLaN residual is valid but slightly less powerful
  (0.08 / 0.08 / 0.78 / 1.00)**.
- Figures: 4 (architecture-lite; δ_f ladder; size vs δ_f; DK certificate vs confusion). Tables: 2 (ladder; confusion).
- What it does NOT need: floating base, InEKF, gait phase estimation, nuisance study (payload/slope), baselines table.
- Additional runs required: **DONE (Sprint 8 L3)** — `experiments/e13_residual_symmetry/weld_addendum.py` reuses the e13
  worker/residual machinery with `weld_base=True` and the frozen weld DeLaN models (`models/delan_weld/weld_equiv_v1`,
  `weld_plain_v1`); figure `e13weld_power_and_contamination.png`, tables `e13a_weld_power.csv` / `e13b_weld_size.csv`.
  The welded legs still swing under the trot controller (contact term 0), so no separate leg-only reference generator was
  needed. Remaining figure work is layout only; the numbers are in hand.

## T-RO main paper adjustments if the RA-L half is split off
- Keep Part 0/1 theory, the raw-signal R⁻ story, nuisance table, isotypic prediction (raw + residual on the floating
  base), the low-SNR residual comparison (e13a floating base), baselines (e07), estimation channel (e02), hardware.
- Move to the RA-L half: DK certificate details (e06), the equivariant-DeLaN construction proof (cite the RA-L paper;
  keep a one-paragraph statement of N3-1 and the ε̄_model corollary in T-RO), the weld/leg-as-arm bridge.
- Risk: the two papers share Prop N3-1 / Cor contamination; the RA-L version states them for the fixed-base pair, the
  T-RO version for the floating-base gait element with the phase shift — the statements differ (Σ = C₂ vs Σ ⊂ G × S¹),
  so self-plagiarism is avoidable if the T-RO version cites the RA-L one for the fixed-base case.
- Timing: RA-L half can be submitted from simulation alone (fixed-base arms are commonly evaluated in sim + one arm);
  the T-RO half waits for hardware (Go2 trot; M1 rolling).
