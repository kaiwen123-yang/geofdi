# Pre-registration — e03: external public simulation benchmark (Liu et al. RA-L 2025, Unitree A1, Gazebo)

Committed before the run (Sprint 7 Block E2; see git log for this file vs the e03 run id). Data:
`raw/public/liu-a1-fault/grufd-ftc_84ca180` (10 CSV files, 100 Hz, no torques / contacts / timestamps; provenance =
Gazebo simulation, `docs/protocol/liu_a1_audit.md` §0). Loader `geofdi.io.liu_a1`. **Only derived statistics are
published (licence: none in the repo — no raw rows leave the data volume).**

## Setting
- Data element: cycles registered by the kinematic phase estimator (`phase/estimator.py`, thigh (HFE) diagonal signal,
  piecewise-linear phase clock) — no controller clock; channels q, q_des, dq, dq_des (48), body pitch/roll angles,
  body rates (53 in Z; yaw angle and command excluded); mirror pairs (LF,RF), (LH,RH) with the uniform-axis signs
  (HAA −1, HFE/KFE +1) — official joint order 0–2 LF, 3–5 LH, 6–8 RF, 9–11 RH.
- Sequential layer: half-cycle mirror score (Block E1) → conformal p against the pooled nominal half-cycles of the SAME
  command condition (vx, vy, wz) across files ("same speed, cross-file") → e-process (e = ½p^{-1/2}, alarm at 1/α = 20).
  H₀′ (per file): ν₀ from the ≈30 s in-place prefix — reported, not tested (the dataset's healthy gait carries a stable
  hip offset ≈ 0.01 rad, audit §7).
- Fault episodes: 250 (125 single, 125 double), 1 s or 2 s each (≈ 1.7–3.3 cycles), calf joints only, η ∈ {0.4, 0.6};
  healthy gaps of 4 s between episodes. Detection is scored INSIDE the episode window (+ 0.5 s grace); false alarms are
  scored on the healthy gaps (excluding the first 0.5 s after an episode end).

## Four fault classes (from the η fields, audit §0)
| class | joints | mirror-symmetric? | R⁻ prediction |
|---|---|---|---|
| single | one calf (LF/LH/RF/RH) | no | detects (η 0.4 more often than 0.6) |
| mirror double | (LF,RF) or (LH,RH), same η | **yes** (fixed by σ_*) | **blind: detection ≈ nominal false-alarm rate** (Theorem N1-2 on external data) |
| same-side double | (LF,LH) or (RF,RH) | no | detects |
| diagonal double | (LF,RH) or (LH,RF) — the paper's Case 3 | no (mirror maps it to the other diagonal) | detects |
Detectors: **R⁻** half-cycle e-process (raw signals — no residual: no torque in the data, stated); **R⁺** tracking-error
(‖q − q_des‖ per half-cycle, mirror-invariant) conformal e-process; **Mahalanobis** on cycle features (conformal
e-process); **GRU regressor** (Table I spec: 57 → 256 → 12, MSE, 100 epochs, batch 32, lr 1e-4, 1 layer to verify,
50 Hz windows) with the paper's deployment rule (low-pass η̂, joint faulty iff η̂_j < 0.7): trained leave-one-file-out
(10 folds), 3 seeds; generalisation splits: train η = 0.4 → test 0.6 and vice versa; train Single files → test Double
files.

## Predictions
1. Single: R⁻ episode detection rate ≥ 0.5 at η = 0.4 with median delay ≤ 1 s; R⁺ and Mahalanobis detect ≥ 0.5.
2. Mirror double: R⁻ detection rate within its nominal false-alarm rate (binomial band); R⁺ / Mahalanobis detect.
3. Same-side and diagonal doubles: R⁻ detects at least as often as single (two broken joints).
4. GRU LOFO: detection ≥ 0.8 with delay ≈ 0.5–1 s (paper's own regime); unseen-η and single→double splits degrade
   (reported as the generalisation table); GRU is not blind to the mirror class (it is trained on it).
5. Localisation: R⁻ pair–joint ranking names the faulty pair (calf) in ≥ 0.8 of detected single episodes; left/right by
   the R⁺ per-leg deviation.
## What would falsify
Mirror-double episodes detected by R⁻ well above the false-alarm rate → the sim's bilateral fault is not Σ-invariant
(load redistribution) or the estimator/registration leaks asymmetry — report as such. Diagonal not detected while
single is → the mirror pairing is wrong (order/sign) — report. No parameter is tuned after seeing the results.
