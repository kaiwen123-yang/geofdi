# Baseline comparison protocol (e07)

## What is compared
| detector | training data | per-cycle score |
|---|---|---|
| R⁻ (mirror channel, ours) | **none** (nominal calibration cycles only) | Hemerik–Goeman mirror-test p per 5-cycle window → FAR-calibrated e-CUSUM (h from the pooled nominal windows) |
| rplus_resid (ours) | none (analytic nominal model; nominal calibration cycles) | phase-registered momentum-residual magnitude |
| rplus_track (ours, S2 reference) | none | tracking-error magnitude (needs q_ref: sim-only) |
| GRU classifier (Liu-style baseline) | **fault rollouts** of the *seen* magnitudes (gain 0.8, bias 1.0 N·m, friction ×2, +100 g) on LF-HFE/KFE + nominal; a second GRU trained *without* inertia faults | P(fault) of 50-step windows, averaged per cycle |
| window autoencoder | nominal rollouts only | reconstruction error of 50-step windows, averaged per cycle |
| Mahalanobis gate | nominal rollouts only (Ledoit–Wolf covariance of per-cycle channel means/stds) | squared Mahalanobis distance of the cycle's feature vector |

GRU architecture (2 × 64 GRU + FC, window 50 steps = 0.25 s, stride 10) is a placeholder to be back-filled from Liu et
al. (RA-L 2025) once the paper is available; the *protocol* (train on some fault magnitudes, test on unseen ones and on
an unseen fault type) is what the table is about.

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
