# Sprint 9 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint9_spec.md`.

## Anti-loss
- [ ] spec + progress committed

## Block Q — corpus audit and ingest (commits q1/q2/q3)
- [ ] Q1 `docs/protocol/go2_quadric_audit.md`: high-level txt format + determination checklist (IMU / footForce /
  motorState joints / body pose / gait-mode / time base); parser `io/go2_highlevel_txt.py`
- [ ] Q1.2 foot-IMU csv: columns, rate, which leg (LH per user), alignment to the high-level stream
- [ ] Q1.3 Fixposition: enumerate session dirs, parse the trajectory product (pos/vel/att + GPS time), time-sync strategy
- [ ] Q1.4 per-session summary table (duration, rows, rates, gaps/anomalies)
- [ ] Q1.5 `docs/protocol/go2_session_meta_form.md` (11 rows; user metadata filled + [inferred] site attribution)
- [ ] Q2 11 sessions ingested to `raw/go2/2026-01-05/…` and `raw/go2/2026-03-06/…` with checksums; catalog rows
- [ ] Q3 `io/go2_quadric.py` unified loader + straight-segment split (yaw rate + RTK heading rate); unit tests
- [ ] rp031 Block Q review pack

## Block R — own-Go2 hardware experiments (pre-registration FIRST)
- [ ] `docs/protocol/go2_real_preregistration.md` committed BEFORE any run (incl. P-LH, P-A, P-RTK)
- [ ] R1 per-session H0/H0' (11 reports) + summary figure
- [ ] R2 cross-period reproducibility table (ν₀ per session; Jan-calibrated H0' applied to Mar and reverse)
- [ ] R3 constant ~5 kg payload recorded as a corpus-wide condition (no payload-variation row)
- [ ] R4 estimator value (contact-aided InEKF vs RTK; πᵢ gating) or honest skip note
- [ ] R5 natural-anomaly hunt across 11 sessions + per-session conclusion table
- [ ] R6 foot-IMU as independent phase-estimator check (LH only, validation source not mirror channel)
- [ ] rp032 Block R review pack

## Block B — background backlog (one commit each)
- [ ] B1 classical baseline row (De Luca momentum observer + χ² fixed threshold) on e08 grid + e07 nuisance
- [ ] B2 robustness sweeps `experiments/e19_robustness/` (phase error, K_cal, block-length mismatch)
- [ ] B3 N2 written up in `theory/sections/04_n2.tex`; `make theory` zero error; tag `theory-part3-v1`
- [ ] B4 N1-3 attempt (timeboxed): power lower bound or conjecture + empirical curve
- [ ] B5 173247 diagnosis (ν trajectory shape classification + condition cross-check)
- [ ] B6 Mini Cheetah flying-trot block-mode rerun (does H0' come back into band?)
- [ ] B7 joint sign tables back-filled into `00_notation` (M1 verified; Go2 per Q1)
- [ ] B8 P1 Remark rewritten per the falsification result + empirical anchor
- [ ] B9 MANIFEST packing bug third fix + VERIFIED by generating a test pack and asserting non-template
- [ ] rp033 Block B review pack

## Wrap-up
- [ ] tag `real-data-milestone-2`; final summary (Q1 checklist incl. **joint stream present?**, R1/R2 tables,
  B nine-item status, updated "only a robot day can fix these" list); `raw/m1/nominal-crossday/` placeholder created
