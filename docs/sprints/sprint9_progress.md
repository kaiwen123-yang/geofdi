# Sprint 9 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint9_spec.md`.

## Anti-loss
- [x] 282183a spec + progress committed

## Block Q — corpus audit and ingest (commits q1/q2/q3)
- [x] Q1 audit doc + determination checklist. **KEY: NO joint stream** (files are script(1) transcripts of `ros2 topic echo
  /sportmodestate`; the high-level API has no motorState). Compensated by foot_position_body/foot_speed_body = FK(q) →
  34-channel element. Parser `io/go2_highlevel_txt.py` (schema-agnostic, keeps unknown vendor fields)
- [x] Q1.2 foot IMU: 2 redundant IMUs on ONE board, 200 Hz, local-time stamps → epoch; LH per operator ⇒ validation source only; March has none
- [x] Q1.3 Fixposition parsed (geodetic/odometry/status); **common Unix clock ⇒ exact 1:1 session match, no motion-correlation needed**; minimal subset ingested (trade-off recorded)
- [x] Q1.4 per-session table (11 rows): 249.8 Hz, 0 back-jumps, gaps 0-6, `by3` has a 4.74 s dropout
- [x] Q1.5 meta form with operator conditions + site attribution [inferred] (nmb→B, xb→A, by→C) on two independent indicators
- [x] Q2 11/11 ingested (1.6 GB, per-file sha256), catalog + meta.yaml filled; source never written
- [x] Q3 `io/go2_quadric.py` loader + dual-criterion straight split (windowed MEAN, not RMS — gait wobble); 6 unit tests green
- [x] rp031 Block Q review pack

## Block R — own-Go2 hardware experiments (pre-registration FIRST)
- [x] 96c842e pre-registration committed 17:50 BEFORE any run (incl. P-LH, P-A, P-RTK)
- [x] R1: naive H0 rejects **11/11** (prediction 1 ✓); H0' alarms 8/11 (prediction 2 ✗ as stated → R5); ν₀ 7.2-21.9; figure
- [x] R2: **two-month ν₀ ratio 1.18 < within-site sd 3.1-6.4 ⇒ prediction 3 ✓**; cross-session H0' transfer poor (0.32/0.49)
- [x] R3 constant ≈5 kg payload recorded as a corpus-wide condition (no contrast exists)
- [x] R4 ran (foot_position_body replaces the missing joints): ATE 6.6 % of path, gating-invariant (honest null); **threshold rejects 40.6 % of nominal contacts vs GeoFDI-soft 1.9 %**
- [x] R5 **diagnosis: H0' alarms 8/11 pooled vs 2/11 within one run; 8/11 boundary-jump, 0 drift ⇒ calibrate ν₀ per run**; P-LH side stable 11/11 but the LH board is FALSIFIED as its cause (March has no board)
- [~] R6 phase error 21.2 % > the pre-registered 10 % ⇒ **check fails**; candidate causes recorded, does not affect R1-R5
- [x] rp032 Block R review pack

## Block B — background backlog (one commit each)
- [ ] B1 classical baseline row (De Luca momentum observer + χ² fixed threshold) on e08 grid + e07 nuisance
- [ ] B2 robustness sweeps `experiments/e19_robustness/` (phase error, K_cal, block-length mismatch)
- [ ] B3 N2 written up in `theory/sections/04_n2.tex`; `make theory` zero error; tag `theory-part3-v1`
- [ ] B4 N1-3 attempt (timeboxed): power lower bound or conjecture + empirical curve
- [ ] B5 173247 diagnosis (ν trajectory shape classification + condition cross-check)
- [ ] B6 Mini Cheetah flying-trot block-mode rerun (does H0' come back into band?)
- [ ] B7 joint sign tables back-filled into `00_notation` (M1 verified; Go2 per Q1)
- [ ] B8 P1 Remark rewritten per the falsification result + empirical anchor
- [x] B9 matcher broadened (any *manifest*.md, any case) + **post-build assertion that fails loudly**; verified by generating packs (positive + negative cases)
- [ ] rp033 Block B review pack

## Wrap-up
- [ ] tag `real-data-milestone-2`; final summary
- [x] `raw/m1/nominal-crossday/` placeholder created with ingest instructions
