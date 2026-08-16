# Sprint 8 progress (gate checklist)

Read this first in a new session; continue from the first unchecked item. Each item: `[ ]` open / `[x] <commit>` done /
`[~] <commit>` done with a documented miss. Update and push at the end of every Block. Spec: `sprint8_spec.md`.

## Anti-loss / pre-check
- [x] spec + progress file committed
- [~] `/mnt/h` NOT mounted in WSL and sudo needs a password (cannot mount from the agent). Workaround used
  (2026-08-16): H:\m1_data (4 rosbag2 sessions, 15.5 GB) staged read-only via Windows-side robocopy into
  `$GEOFDI_DATA_ROOT/scratch/h_stage/m1_data`, source SHA-256 taken with Get-FileHash on the H: originals
  (`scratch/h_stage/m1_data_source_sha256.txt`) so ingest checksums are verified against the source. H: never written.
  **User action still useful (next session):** `sudo mkdir -p /mnt/h && sudo mount -t drvfs H: /mnt/h` — then
  `sha256sum -c` against the source list can be repeated directly on `/mnt/h/m1_data`.

## Block D — M1 real data (headline; commits d1/d2/d3)
- [ ] D1 `docs/protocol/m1_h_data_audit.md`: 4 sessions × (structure/format/size/time; topic table with measured rate,
  duration, sample fields; decision list: joint count, `names` verbatim, q/dq/efforts + magnitudes, wheel encoders +
  wrap, IMU topic/rate/frame, cmd, contact, motor temp, localization pose, timestamp monotonicity + inter-topic skew)
- [ ] D1 `io/m1_mapping.yaml` checked against `names` (unverified→false or corrected + diff recorded)
- [ ] D1 per-session verdict (rolling H0' / rolling InEKF / DeLaN / archive-only)
- [ ] D2 qualified sessions ingested via `ingest_session.sh` into `raw/m1/nominal/` (or audit/), meta.yaml filled
- [ ] D2 straight-segment splitting (cmd if present; else |ω_z| + wheel-speed-difference rule, thresholds recorded)
- [ ] D3 `docs/protocol/m1_real_preregistration.md` committed BEFORE any run
- [ ] D3.1 `run_pipeline.sh <session> --robot m1 --mode rolling` per session → report.md; first real-robot R⁻ H0' figure
  (QQ, per-window p, e-process); block length L from protocol boundary, adjusted after measured block correlation
- [ ] D3.2 rolling InEKF vs fixed-foot RIEKF vs ESKF on real data (reference = localization topic if present, else
  relative metrics)
- [ ] D3.3 M1 real equivariant DeLaN (if ≥ 20 min nominal rolling) — residual R⁻ vs model-free R⁻ table
- [ ] D3.4 efforts semantics / real ε_dyn candidate / ν₀ magnitude → audit doc (theory intake)
- [ ] rp025 Block D review pack

## Block L — leftover fixes (`chore: leftover fixes`)
- [ ] L1 `make_review_pack.sh` in-block MANIFEST overrides top-level template; rp020–024 top-level MANIFEST back-filled
- [ ] L2 Day-0 docs + `run_pipeline.sh` report notes: rolling primary test = H0', naive H0 excursion expected
- [ ] L3 weld world e13a/e13b (power + δ_f contamination, R=50) → `docs/paper/split_option.md`
- [ ] L4 e03 audit doc note: mirrored-bilateral external test n=8 inconclusive
- [ ] L5 `docs/sprints/` registers N2 theorem write-up as next theory sprint item
- [ ] rp026 Block L review pack

## Block T2 — N1-2 two-layer restatement (`feat(theory): N1-2 two-layer restatement`; tag `theory-part1-v1.1`)
- [ ] (I) law-level blindness dichotomy (Σ-fixed fault + A5-under-fault ⟹ power ≤ α; law-level breaking detectable
  by consistent statistics) + amplitude-ceiling remark (→ P1)
- [ ] (II) mean-level power characterisation (K‖Π⁻μ‖², single-leg share ½)
- [ ] (III) statistic-consistency remark (paired-mean blind to zero-mean law differences; energy distance sees them → P2)
- [ ] falsification conditions + empirical-anchor placeholders; `make theory` zero error
- [ ] rp027 Block T2 review pack; tag `theory-part1-v1.1`

## Block P — three pre-registered prediction experiments (`experiments/e15_predictions/`)
- [ ] `docs/protocol/e15_preregistration.md` committed BEFORE the run
- [ ] P1 chirality ceiling: κ ∈ {0.7,0.5,0.4,0.3}, R=30 → power + chirality index vs κ figure + ceiling estimate
- [ ] P2 statistic split: left-leg encoder noise var ×4, R=50 → paired_energy ≈ α vs energy_distance > α + protocol note
- [ ] P3 slip regimes: Go2 unilateral patch / uniform low μ; M1 single-wheel / both-wheel μ×0.5, R=30 → regime × channel table
- [ ] results back-filled into T2 empirical anchors, theory recompiled
- [ ] rp028 Block P review pack

## Block G — slip statistics + πᵢ gating (`detect/stance_event.py`, `estimate/pi_gating.py`, `experiments/e16_pi_gating/`)
- [ ] G1 per-stance-event conformal p / e-values (legged) + per-wheel rolling-constraint residual (wheeled); FAR per event
- [ ] G2 πᵢ gating (hard e ≥ 1/α, soft covariance scaling) → InEKF update
- [ ] G3 e16: no gating / literature threshold (0.4 m/s + cov ×10) / GeoFDI hard / soft — RMSE, NEES, gating timeline
- [ ] G4 `docs/protocol/hw_slip_protocol.md`
- [ ] rp029 Block G review pack

## Block PUB — public data harvest (`experiments/e17_public_realdata/`)
- [ ] `docs/protocol/e17_preregistration.md` committed BEFORE the run
- [ ] PUB1 Mini Cheetah contact dataset ingested to `raw/public/minicheetah-contact/` (URL, licence, hashes)
- [ ] PUB2 Mini Cheetah: 8-terrain R⁻ H0' FAR table; real residual R⁻ (model choice recorded); air-sequence weld run; (opt) K₄
- [ ] PUB3 Street A1 / legkilo Go1: straight-segment mining → real H0' FAR figures; legkilo three-estimator gating table
- [ ] PUB4 `docs/data_catalog.md` + paper outline evidence stack (five layers)
- [ ] rp030 Block PUB review pack

## Wrap-up
- [ ] tag `real-data-milestone-1`; final summary (audit one-liners, first real-robot figure list, leftover status
  table, "only things still missing" list)
