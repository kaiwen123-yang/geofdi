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

---

## Classical model-based baseline row (Sprint 9 B1) — `experiments/e21_classical_baseline`, run `e21-20260816`

The method reviewers name first: the De Luca–Mattone generalised-momentum observer with a **χ² threshold** on the
whitened joint residual (Haddadin, De Luca & Albu-Schäffer, *Robot Collisions: A Survey*, T-RO 2017). Implemented in
`geofdi/baselines/momentum_chi2.py` and run on the **same rollouts** as R⁻ (`go2_urdf_sym`, R = 20 per cell,
K_cal = 60, K_post = 100, α = 0.05, debounce 5), so the rows below merge directly into the e07 table.

Two variants are reported, because the classical recipe's guarantee and this protocol's guarantee are not the same object:
* **`classical_fixed`** — the textbook threshold $\chi^2_{k,1-\alpha}$.
* **`classical_far_matched`** — the same statistic with the threshold re-calibrated **on the run's own nominal data so
  that the probability of any alarm within the monitoring horizon is α** (block-bootstrapped exactly like the e-CUSUM
  threshold h), then evaluated on a **held-out** second half of the nominal stretch.

| detector | mean detection over the fault grid | **nominal alarm rate** (target ≤ 0.05) | median delay [cycles] | symmetric drift nuisance | symmetric payload nuisance |
|---|---:|---:|---:|---:|---:|
| `classical_fixed` | 0.77 | **0.15** | 13.5 | **0.95** | 0.00 |
| `classical_far_matched` | 0.96 | **0.50** | 4.6 | **0.90** | **0.65** |
| `Rminus` (e-CUSUM) | 0.71 | **0.00** | 25.0 | **0.00** | **0.00** |

Per-cell detection: both classical variants and R⁻ reach 1.00 on every actuator-gain and actuator-bias cell; on the
friction cells the classical detector is far stronger (0.85–0.90 far-matched, 0.20 fixed) than R⁻ (0.00), which is the
honest weak spot of the mirror channel on this fault family at these window settings.

**Reading — three findings, only one of which flatters us.**
1. *The classical detector is more powerful on raw detection* (0.77 / 0.96 vs 0.71), and much faster (4.6–13.5 vs 25
   cycles). Where a dynamics model and its residual are available and the fault is a friction change, it should be used.
2. *But it does not hold the false-alarm rate it claims.* At its own nominal threshold the measured nominal alarm rate is
   **0.15, three times α**; the χ² law assumes an i.i.d. Gaussian residual and a gait residual is periodic and serially
   correlated. Worse, re-calibrating the threshold on nominal data does **not** repair it: on a **held-out** stretch of
   the *same nominal recording* the far-matched rate is **0.50**, because the momentum residual is not stationary across
   a recording. R⁻ measures 0.00 in every cell — its level is a permutation guarantee, not a distributional assumption.
   The detection numbers above are therefore *not* at equal FAR and cannot be read as a like-for-like power comparison.
3. *Symmetric nuisances break it and cannot be re-calibrated away.* A symmetric torque/friction drift alarms the
   classical detector on 0.90–0.95 of runs (and a symmetric payload on 0.65 of far-matched runs) because it genuinely
   changes the residual; R⁻ stays at 0.00 by construction, since the nuisance is Σ-invariant. This is the property no
   threshold tuning can supply, and it is the reason the two channels are complementary rather than competing.

The claim this table supports is the one stated at the top of this protocol, unchanged: **comparable detection at a
guarantee the baseline does not provide — no fault data, an exact level, and silence under symmetric nuisances** — plus
the newly quantified caveat that the classical baseline wins on friction faults and on delay.
