# Baseline comparison protocol (e07)

## What is compared
| detector | training data | per-cycle score |
|---|---|---|
| R⁻ (mirror channel, ours) | **none** (nominal calibration cycles only) | Hemerik–Goeman mirror-test p per 5-cycle window → FAR-calibrated e-CUSUM (h from the pooled nominal windows) |
| rplus_resid (ours) | none (analytic nominal model; nominal calibration cycles) | phase-registered momentum-residual magnitude |
| rplus_track (ours, S2 reference) | none | tracking-error magnitude (needs q_ref: sim-only) |
| GRU classifier (e07 placeholder) | **fault rollouts** of the *seen* magnitudes (gain 0.8, bias 1.0 N·m, friction ×2, +100 g) on LF-HFE/KFE + nominal; a second GRU trained *without* inertia faults | P(fault) of 50-step windows, averaged per cycle |
| GRUFD regressor (Liu et al. RA-L 2025, Table I; `mode: regression_eta`, Sprint 7) | fault + nominal sequences (Liu CSVs for e03; our sim for e08) | η̂ ∈ R¹² by MSE regression (input 57 = body angles/rates + q, q_des, dq, dq_des + cmd; hidden 256; 1 layer, to verify; 100 epochs, batch 32, lr 1e-4); deployment rule = low-pass η̂ then joint faulty iff η̂_j < 0.7 (Algorithm 1); our per-cycle score for the unified protocol = 1 − min_j η̂_j (also reported with the paper's own threshold rule) |
| window autoencoder | nominal rollouts only | reconstruction error of 50-step windows, averaged per cycle |
| Mahalanobis gate | nominal rollouts only (Ledoit–Wolf covariance of per-cycle channel means/stds) | squared Mahalanobis distance of the cycle's feature vector |

The e07 GRU architecture (2 × 64 GRU + FC, window 50 steps = 0.25 s, stride 10) was a placeholder; the paper's GRUFD
spec (Table I, read 2026-08-16) is now implemented as `GRURegressor` and is the baseline for e03/e08. The *protocol*
(train on some fault magnitudes, test on unseen ones and on an unseen fault type) is what the table is about.

## Unified alarm rule (identical for every detector)
1. Each run has K_cal = 60 nominal calibration cycles (after 20 warm-up cycles) followed by 100 monitored cycles;
   faults switch on at monitored cycle 1.
2. The detector's per-cycle score s_k on the monitored cycles is turned into a one-sided conformal p-value against
   the run's own calibration scores: p_k = (1 + #{s_cal ≥ s_k}) / (n_cal + 1).
3. Alarm rule: e-process E_t = ∏_{k≤t} e(p_k), e(p) = p^{-1/2}/2, alarm at E_t ≥ 1/α, α = 0.05 (Ville: false-alarm
   probability over the whole monitoring horizon ≤ α under exchangeability of nominal cycles).
   R⁻ uses the same idea on window p-values with the FAR-calibrated e-CUSUM of S2 (its 5-cycle windows are discrete,
   see protocol_params.md).
4. Reported: detection rate within 100 (and 20) cycles, median / q90 delay in cycles; nuisance rows (symmetric
   torque/friction drift, symmetric payload) report the alarm fraction = false-alarm rate at fixed FAR.
5. A naive rule (fixed threshold at the 95 % quantile of the calibration scores, first exceedance) is reported next to
   it with its actual per-cycle exceedance rate, to show why a FAR-controlled aggregation is needed (a 5 % per-cycle
   rate gives ≈ 99 % "detections" in 100 cycles without a fault).

Because calibration set, p-value map and alarm rule are shared, differences between rows of the table come from the
scores alone. Ours never see fault data; the GRU does — its advantage on seen faults is expected and is *not* the
claim; the claim is: comparable at fixed FAR, no fault data needed, FAR guarantee, no nuisance alarms on the mirror
channel.
