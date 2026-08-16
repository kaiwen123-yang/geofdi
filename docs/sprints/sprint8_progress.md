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
- [x] d1 D1 `docs/protocol/m1_h_data_audit.md`: 4 sessions audited (rosbag2 v5; 200 Hz joints/IMU; session 4 truncated at the
  source: 76 of 151 s readable; all "walk" = wheeled rolling, no gait); tools `geofdi.io.m1_rosbag` (sqlite+typestore, tolerant of
  0-byte/partially corrupt db3), `scripts/m1_h_audit.py`; inventory/metadata tools made truncation-tolerant
- [x] d1 D1 `io/m1_mapping.yaml`: names match (fl1_hip_roll..br4_foot) → `unverified: false`; per_leg_sign RF/RH = [+1,−1,−1,−1]
  (the yaml's earlier candidate [−1,−1,−1,−1] was wrong for ABAD — recorded); IMU frame (x right, y back, z down), g units,
  R_body_from_sensor; odom conventions; loader accepts raw bags (auto-extract), vendor vs sim conventions via meta.yaml
- [x] d1 D1 per-session verdicts (§9): static → audit/rest-noise only; 172847/173028/173247 → rolling H0' + InEKF (173028 primary);
  DeLaN impossible (4.6 min total < 20 min)
- [x] d1 D2 all four ingested (`raw/m1/audit/m1_static_…`, `raw/m1/nominal/m1_walk_…`), payload hashes verified equal to the H:
  Get-FileHash list; meta.yaml + catalog rows filled (fingerprints 5ee91cfb / ab623e3e / 2566ec71 / 4fc3c926)
- [x] d1 D2 straight-segment fallback `straight_mask_kinematic` (RMS|ω_z|<0.15, RMS(L−R)<1 rad/s, mean|wheel|>2 rad/s, runs ≥2 s;
  thresholds from the distributions, audit §10): 5/12/5 runs, 30.6/83.2/38.4 s
- [x] 46ffc52 D3 pre-registration committed BEFORE any run (2026-08-16 14:05)
- [x] d3 D3.1 pipeline on 3 sessions (L=1 and 2 s); first real-robot R⁻ H0/H0' figure `e18_real_h0_h0prime.png`. Naive H0 rejects
  all 3 (predicted); H0' sequential e-process silent all 3; H0' differenced in band on 172847/173028, rejects on 173247 (real slow
  drift — prediction 2 partially falsified, reported, nothing tuned). ν0 13.7/2.8/9.8; lag1 0.65/0.52/0.79 → 0.23/0.23/0.55 at L=2
- [x] d3 D3.2 rolling InEKF vs fixed-foot RIEKF/ESKF + RollingESKF (`experiments/e18_m1_real`): path recovered 0.99/1.00/0.99 vs
  0.04/0.03/0.04, per-run arclen-err 0.6-1.4% vs 98% (real-robot e10). Vector metrics dominated by shared skid-steer yaw
  corruption (all filters) → Block G motivation. Reference = vendor odometry (not truth)
- [~] d3 D3.3 DeLaN NOT run (pre-declared: 4.6 min nominal rolling < 20 min minimum) — reported, not a miss
- [x] d3 D3.4 efforts observations / real ε_dyn candidates / ν0 magnitudes → audit doc §8/§11/§13
- [x] rp025 Block D review pack

## Block L — leftover fixes (`chore: leftover fixes`)
- [ ] L1 `make_review_pack.sh` in-block MANIFEST overrides top-level template; rp020–024 top-level MANIFEST back-filled
- [ ] L2 Day-0 docs + `run_pipeline.sh` report notes: rolling primary test = H0', naive H0 excursion expected
- [ ] L3 weld world e13a/e13b (power + δ_f contamination, R=50) → `docs/paper/split_option.md`
- [ ] L4 e03 audit doc note: mirrored-bilateral external test n=8 inconclusive
- [x] L5 `docs/sprints/theory_backlog.md` registers the N2 formalisation (sparse-correction consistency + signature=observable projection) as the next theory-sprint item
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
