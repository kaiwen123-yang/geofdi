# Pre-registration — GeoFDI on the QUADRIC-GINS Go2 corpus (Sprint 9 Block R)

Committed BEFORE any Block-R run (this file's git timestamp is the pre-registration timestamp). Corpus: the 11 ingested
Unitree Go2 sessions of `raw/go2/2026-01-05/{xb1..xb4,nmb1..nmb4}` and `raw/go2/2026-03-06/{by1,by2,by3}` (audit:
`docs/protocol/go2_quadric_audit.md`; conditions: `docs/protocol/go2_session_meta_form.md`). Machine in service > 1 year,
constant ≈ 5 kg payload, ≈ 1 m/s commanded trot, clear weather, three sites, two dates two months apart.

## Fixed settings (nothing below is tuned after seeing a result)
α = 0.05; N = 64 phase points; 10-cycle windows; M = 512 permutations; seed 0; element = the 34 in-Z channels of
`io/go2_quadric.build_go2_quadric_manifest` (foot_pos 12 + foot_vel 12 + foot_force 4 + IMU 6 — **there is no joint
stream**, see the audit's determination list); cycles registered by `phase/estimator.estimate_phase` driven by the trot
contrast of the vertical foot positions, phase origin at LF stance onset; segments from
`io/go2_quadric.straight_mask_go2` (|mean ω_z| < 0.15 rad/s AND RTK |mean heading rate| < 8°/s where the fix is OK AND
mean speed > 0.3 m/s AND `mode == 3`); H₀′ calibration = the first third of the analysed cycles. R⁻ has zero trainable
parameters throughout.

## Baseline predictions
1. **Naive H₀ rejects on most sessions.** A machine in service > 1 year, carrying a constant ≈ 5 kg payload and an
   LH-mounted IMU board with cabling, has a stable natural left/right asymmetry: the whole-session flip test should give
   p < 0.05 on ≥ 8 of the 11 sessions. This is the *expected healthy* regime (real ε_dyn), not a fault.
2. **Within-session H₀′ is in band / silent.** The per-window H₀′ test (window vs the session's own calibration third)
   has a rejection rate inside the binomial band around α, and the sequential H₀′ e-process does **not** alarm
   (max E < 1/α = 20) on a healthy session. Falsified if the H₀′ e-process alarms on the majority of sessions.
3. **ν₀ drift across two months is bounded.** For same-site session pairs, |ν₀(Mar) − ν₀(Jan)| stays within the spread
   of the within-day session-to-session variation of ν₀ (i.e. the between-date variance does not exceed the between-run
   variance at the same site by more than a factor 2). Reported either way — this is the reproducibility claim.
4. **πᵢ gating does not hurt.** In R4, GeoFDI-πᵢ gating is no worse than no gating on the RTK-referenced trajectory
   error, and its nominal per-event false-rejection rate is ≈ α.

## Added predictions (operator metadata, 2026-08-16)
- **P-LH — the natural asymmetry has a left-hind localisation.** The foot-IMU board and its cable sit on the LH leg and
  are a *known* true asymmetry of this robot. Prediction: the per-channel / per-leg isolation ranking of the R⁻ statistic
  points at LH (or the hind pair) more often than chance across the 8 Jan sessions (which carry the board), and the 3 Mar
  sessions (**no foot IMU recorded**) show a weaker or differently-located ranking. A hit is logged as *known-true-cause
  verification*, not as a fault detection. Falsified if the ranking is uniform over legs on the Jan sessions.
- **P-A — the within-session surface switch at site A is a symmetric nuisance.** At site A (`xb*`) the robot crosses
  from semi-rough to smooth tile inside a session. A surface change is bilaterally symmetric, so it must **not** trigger
  the H₀′ monitor: prediction — no H₀′ e-process alarm attributable to the surface transition, while a symmetric
  magnitude readout (Π⁺ energy / foot-force level) does change across it. The smooth stretch is the natural-slip hunting
  ground for R5 and the estimator segment for R4. Falsified if H₀′ alarms exactly at the surface transition.
- **P-RTK — the Fixposition reference must be quality-gated before use.** R4 uses the RTK reference only where the
  fusion status says the fix is OK (`gnss1_status ∈ {5, 8}`); the GNSS-degraded site-B stretches are dropped or
  downgraded and **the dropped fraction is reported in the table**. Prediction: the site-B sessions (`nmb*`) lose a
  substantially larger fraction than the site-C sessions (`by*`). Falsified if the gate removes so much of every session
  that no RTK-referenced comparison is possible — in which case R4 is reported as skipped with the numbers.

## Per-experiment falsification
- **R1** (per-session H₀/H₀′): falsified if H₀′ within a session rejects at a rate far above α on the majority of the
  corpus while naive H₀ also rejects — that would mean the asymmetry is *not* stationary within a session and the H₀′
  construction fails on this robot.
- **R2** (cross-period): falsified if ν₀ at the same site changes by more than the stated factor between January and
  March — reported as a reproducibility limit, with the candidate causes (wear, payload placement, site sub-area).
- **R4**: falsified if gating *degrades* the RTK-referenced error, or if its nominal false-rejection rate exceeds 2α.
- **R5** (natural-anomaly hunt): every H₀′ rejection is diagnosed with the session-173247 procedure (ν-trajectory shape:
  slow drift vs boundary jump; cross-check against site/segment conditions) and logged per session, whatever the result.
- **R6**: the LH foot IMU is used **only** as an independent phase/touch-down check (single leg ⇒ it cannot be a mirror
  channel). Report the phase error between the foot-IMU impact events and the estimator's LH touch-down phase; falsified
  as a *validation* if that error exceeds 10 % of a gait period, which would indict the phase estimator on this corpus.

## Recorded deviations from the original sprint plan
R3 does not run a payload-variation nuisance row: the operator states the payload was **constant** across all 11 sessions,
so the corpus has no payload contrast. "Constant ≈ 5 kg payload" is recorded as a corpus-wide condition instead.
