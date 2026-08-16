# QUADRIC-GINS Go2 corpus — format audit and ingest record (Sprint 9 Block Q1)

Date: 2026-08-16 · Tooling: `io/go2_highlevel_txt.py` (transcript parser), `io/go2_quadric.py` (unified loader),
`scripts/ingest_session.sh` · Status: **audited, 11/11 sessions ingested, loader + tests green**.

This is the project's second own-hardware corpus (after the M1 wheeled sessions of Sprint 8) and the first own **legged**
one: 11 Unitree Go2 sessions recorded for the operator's QUADRIC-GINS project, re-used here with a Fixposition
Vision-RTK2 reference, a constant ≈ 5 kg payload, on a machine in service for over a year, across three sites and two
dates two months apart.

## 0. Provenance and the read-only rule

Source (read **only**, never written; the paths appear here and in the sprint spec, never in code):
`G:\QUADRIC-GINS\01_raw_data\original_mirror\2026.1.5数据\{高层数据,足部惯导数据,fixpositon数据}` and
`...\2026.3.6-测试数据\{高层数据,fixption数据}` (the vendor's spelling `fixpositon` / `fixption` is kept verbatim).
Source volumes measured before ingest: high-level 1.4 GB (Jan) + 417 MB (Mar), foot IMU 134 MB (Jan only),
Fixposition 895 MB (Jan) + 886 MB (Mar). Everything reaches the repo through `ingest_session.sh` only
(per-file sha256 + `meta.yaml`); nothing in this sprint writes to the source tree.

**Fixposition subsetting (recorded trade-off).** A Fixposition session directory holds ~20 CSVs, ~110 MB, most of it raw
sensor streams (`gnss1-raw`, `gnss2-raw`, `imu-data`, `userio-raw`, `corr-raw`). Only the fused products are needed for a
GeoFDI reference, so the ingest keeps **`user_io-out-poi_geodetic.csv`, `user_io-out-poi_odometry.csv`,
`user_io-out-poi_smooth_odometry.csv`, `user_io-out-odom_status.csv`, `tf.csv`, `tf_static.csv`** (~12–20 MB/session) and
drops the raw streams. Consequence: a future re-fusion of the raw GNSS/IMU is not possible from the repo copy and must go
back to the source mirror. Total ingested: **1.6 GB over 11 sessions**.

## 1. What the "high-level data" files actually are

They are **not** structured data files. Each `.txt` is a `script(1)` **terminal transcript** of a
`ros2 topic echo /sportmodestate` session: ANSI escapes, CR line endings, a shell preamble, then a stream of YAML-ish
`unitree_go/msg/SportModeState` records separated by `---`. `xb1.txt` alone is 6.79 M lines / 128 MB for 91 753 records.

`io/go2_highlevel_txt.py` parses this with a line state machine (a real YAML parse of ~90 000 × 66 lines per file is far
too slow), stripping ANSI/CR, and is deliberately **schema-agnostic**: every `key:` / `  key:` path becomes a column,
list items are appended in order, and **unknown paths are kept** and exported as `col_<path>` rather than dropped
(unit-tested). Lines matching nothing — prompts, typed commands, ANSI noise — are counted in `report["skipped_lines"]`
(3–7 per file) and ignored. Partial records at the head/tail (the transcript starts mid-message) are counted separately
and discarded.

### 1a. Determination checklist (the item that decides everything)

| item | verdict | detail |
|---|---|---|
| IMU: quaternion / gyro / accel | **present** | `imu_state.quaternion` (w,x,y,z — verified against `rpy`: yaw = 2·atan2(z,w) matches), `gyroscope` (rad/s), `accelerometer` (m/s², body FLU), `rpy`, plus `temperature`. Rate = the message rate, ≈ 250 Hz |
| `foot_force[4]` | **present** | integer per leg, Unitree order FR, FL, RR, RL; session means 18–22 (units are raw sensor counts, not newtons) |
| **per-joint `motorState` (q / dq / τ_est)** | **ABSENT** | `SportModeState` is the **high-level sport-mode API** and carries no `motorState` at all; the joint stream lives in `LowState`, which was not recorded. Not 20 motors, not 12 — **zero**. The parser's `has_motorstate` flag is False on all 11 files |
| body position / velocity estimate | **present** | `position[3]`, `velocity[3]`, `yaw_speed`, `body_height` — the on-board estimator's output (drifting odometry, used as a diagnostic only) |
| gait / mode fields | **present** | `mode` (3 = locomotion, 1 = balance-stand), `gait_type` (1 throughout), `progress`, `foot_raise_height` |
| **`foot_position_body[12]` / `foot_speed_body[12]`** | **present** | *the decisive compensation* — the robot's own forward kinematics of each leg, in the body frame |
| `range_obstacle[4]` | present | ultrasonic ranges, diagnostic |
| time base | **Unix epoch, UTC, common to all streams** | `stamp.sec/nanosec`; e.g. `xb1` starts 1767615915.805 = 2026-01-05T12:25:15Z. Monotone on every file (0 back-jumps) |

**Consequence for R⁻ — neither of the two pre-planned paths, but the better one.** The sprint plan foresaw either "full
power (joint channels)" or "degraded (foot-force mirror + IMU)". The corpus gives a third, stronger option: with no joint
stream at all, the element is built from **`foot_position_body` + `foot_speed_body` + `foot_force` + IMU = 34 in-Z
channels, 28 of them leg-resolved**. `foot_position_body` *is* FK(q) — the joint asymmetry expressed in Cartesian body
coordinates — so the per-leg kinematic asymmetry that the joint channels would carry is still in the element. Only the
torque channel (τ) is genuinely lost, which costs the residual/R⁺ channel, not R⁻. Both fallbacks remain implemented in
`io/go2_quadric.build_go2_quadric_manifest` (dropping `foot_pos`/`foot_vel` leaves exactly the "foot force + IMU"
degraded element).

Mirror representation (sagittal reflection E = diag(1,−1,1); leg swap LF↔RF, LH↔RH): `foot_pos`/`foot_vel` are polar
vectors → sign (+1,−1,+1); `foot_force` is a magnitude → +1; IMU accel polar, IMU gyro axial. Unit-tested: ρ² = I, ρ
orthogonal, correct per-channel signs, and a symmetric-by-construction element is an exact fixed point
(`tests/test_go2_quadric.py`, 6 tests).

## 2. Foot inertial unit (`足部惯导数据`, January only)

CSV, header `(imu_0)gyro_x,gyro_y,gyro_z,(imu_0)acc_x,acc_y,acc_z,(imu_1)…,mag_x..z,press,temp,sn,timestamp`.
**Two redundant IMUs on one board** (`imu_0`, `imu_1`, same serial `sn`), 200 Hz, 79 k–91 k rows per session.
Timestamps are **local time** (`2026-01-05 20:25:13.103000`, +08:00) — the only stream not already in UTC; the loader
converts to epoch seconds so all three streams share one clock. Gravity sits on **−x** (acc ≈ (−9.75, 0.52, −1.18),
|a| ≈ 9.83), i.e. the board's x-axis points up when the foot is planted: a foot-mounted, not body-mounted, orientation.

**Which leg: LH (left hind), per the operator.** The data are consistent with a single-leg mount (one board, one serial,
one gravity direction) and cannot by themselves identify *which* leg — the inference basis is the operator's statement,
recorded as such. Because it is a single leg, the board **cannot be a mirror channel** (it has no right-side partner);
it is used only as an independent phase / touch-down validation source (R6). Its mass and cable are, however, a *known*
left-hind asymmetry of this robot and thus a candidate true cause for any hind-localised R⁻ signal (pre-registration P-LH).
**March sessions have no foot-IMU recording at all** — which makes Jan-vs-Mar a natural with/without-board contrast.

## 3. Fixposition Vision-RTK2

Each session directory is a decomposed recording (~20 CSVs). The products used:

| file | content | rate |
|---|---|---|
| `user_io-out-poi_geodetic.csv` | WGS-84 lat/lon/alt + ENU velocity + yaw/pitch/roll + **variances** for each | 10 Hz (20 rows/s: paired entries) |
| `user_io-out-poi_odometry.csv` | ECEF pose + twist + full covariance | 10 Hz |
| `user_io-out-odom_status.csv` | fusion/init status, `gnss1_status`, `gnss2_status`, corr/cam/ws status | 5 Hz |
| `tf.csv`, `tf_static.csv` | POI ↔ base extrinsics | — |

**Time sync is trivial and exact: no motion-correlation alignment is needed.** `header.stamp` on the Fixposition CSVs is
Unix epoch from the same recording host as the SportModeState stamps; matching the eight January sessions to their eight
Fixposition directories by first-stamp gives a unique 1:1 assignment with |Δt| ≤ 54 s and no ambiguity (the sessions are
7–9 min apart). The March directories are already named `by1/by2/by3`. The loader resamples the reference onto the
telemetry clock by interpolation and reports the overlap (244–367 s per session).

**Fix-quality gate (P-RTK).** `gnss1_status ∈ {5, 8}` is treated as "fix OK"; the fraction is reported per session and
is strongly site-dependent (§5). Reported position variance ranges from 0.002 m (March) to 34 m (`xb4`), so the reference
is *not* uniformly usable and every RTK-referenced number must state its gated fraction.

## 4. Per-session summary

| session | day | start UTC | records | dur [s] | rate [Hz] | gaps>100ms | backjumps | max dt [ms] | straight runs | straight [s] | % | median v [m/s] | FP rows | FP [Hz] | fix_ok | pvar_x [m] | overlap [s] | foot IMU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmb1 | 2026-01-05 | 11:16:05 | 93267 | 408 | 249.9 | 0 | 0 | 20.0 | 21 | 234 | 63 | 1.07 | 3664 | 10.0 | 0.55 | 0.412 | 323 | yes |
| nmb2 | 2026-01-05 | 11:25:05 | 81811 | 363 | 249.8 | 0 | 0 | 20.0 | 22 | 272 | 83 | 1.03 | 3612 | 10.0 | 0.52 | 0.483 | 321 | yes |
| nmb3 | 2026-01-05 | 11:32:45 | 81951 | 365 | 249.8 | 0 | 0 | 22.0 | 15 | 259 | 79 | 1.05 | 3834 | 10.0 | 0.57 | 0.226 | 328 | yes |
| nmb4 | 2026-01-05 | 11:39:51 | 78610 | 348 | 249.8 | 0 | 0 | 22.0 | 11 | 264 | 84 | 1.06 | 3561 | 10.0 | 0.51 | 0.802 | 315 | yes |
| xb1 | 2026-01-05 | 12:25:15 | 91753 | 396 | 249.8 | 1 | 0 | 198.0 | 10 | 95 | 26 | 1.05 | 4057 | 10.0 | 0.82 | 7.559 | 367 | yes |
| xb2 | 2026-01-05 | 12:33:39 | 84436 | 359 | 249.9 | 1 | 0 | 168.0 | 21 | 258 | 76 | 1.11 | 3766 | 10.0 | 0.82 | 19.522 | 338 | yes |
| xb3 | 2026-01-05 | 12:40:55 | 80306 | 346 | 249.9 | 2 | 0 | 222.0 | 16 | 265 | 82 | 1.11 | 3516 | 10.0 | 0.85 | 27.547 | 321 | yes |
| xb4 | 2026-01-05 | 12:49:28 | 79738 | 361 | 249.7 | 0 | 0 | 28.1 | 15 | 254 | 80 | 1.08 | 3564 | 10.0 | 0.82 | 34.298 | 315 | yes |
| by1 | 2026-03-06 | 07:51:41 | 95860 | 471 | 249.0 | 6 | 0 | 346.0 | 10 | 131 | 34 | 1.11 | 4464 | 10.0 | 1.0 | 0.002 | 350 | **no** |
| by2 | 2026-03-06 | 08:00:44 | 63278 | 305 | 249.3 | 0 | 0 | 91.7 | 13 | 172 | 68 | 1.19 | 3020 | 10.0 | 1.0 | 0.003 | 244 | **no** |
| by3 | 2026-03-06 | 08:06:34 | 63222 | 298 | 249.6 | 4 | 0 | 4740.0 | 12 | 181 | 71 | 1.19 | 2968 | 10.0 | 1.0 | 0.003 | 248 | **no** |

Notes: **no time back-jumps anywhere**; the record rate is a very stable ≈ 249.8 Hz. Stream gaps > 100 ms are rare
(0–6 per session) except **`by3`, which has a single 4.74 s dropout** — the largest anomaly in the corpus; it is inside
the session, and the straight-run splitter naturally cuts there. `xb1` yields much less straight time (26 %) than its
site-mates (76–82 %), consistent with more manoeuvring in that first run of the site. Median speed on the straight
segments is **1.03–1.19 m/s across all 11 sessions**, independently confirming the operator's "≈ 1 m/s".

## 5. Site attribution [inferred] and its basis

The operator described three sites (A indoor–outdoor transition with a rough→smooth tile switch; B smoother ground with
restricted GNSS; C rough ground) but did not record which session was where. Two independent data channels attribute them:

**(i) Position — the three name groups are three distinct places.** Session-median WGS-84: `nmb*` 39.9783 N/116.3456 E,
`xb*` 39.9834 N/116.3394 E (≈ 700 m away), `by*` 39.9846 N/116.3424 E. No session mixes two groups, so the name prefix
*is* the site label.

**(ii) GNSS quality and ground texture.**

| group | fix_ok | median pos-var | IMU >20 Hz vibration (std) | foot-force std | attribution |
|---|---:|---:|---:|---:|---|
| `nmb*` | 0.51–0.57 | 0.23–0.80 m | **1.97 (lowest)** | 4.83 | **B** — worst GNSS *and* smoothest ground: matches "smoother ground, GNSS-restricted" on two independent indicators |
| `xb*` | 0.82–0.85 | **7.6–34 m (most erratic)** | 2.15 | **3.92 (lowest)** | **A** — fix mostly available yet wildly varying variance = repeatedly losing/regaining sky view (walking in and out of a building); lowest foot-force variability fits smooth tile |
| `by*` | 1.00 | **0.002–0.003 m (best)** | 2.11 | **5.38 (highest)** | **C** — open sky with the roughest contact: "rough ground" |

**Honest limitation:** a coarse first-third vs last-third contrast of the vibration texture does **not** resolve site A's
within-session rough→smooth switch (all sites give a ratio 1.00 ± 0.04). Detecting that switch needs finer segmentation;
until then the P-A prediction is tested only in its *monitor* form (H₀′ must not alarm at a symmetric surface change).
All site cells are marked `[inferred]` in `go2_session_meta_form.md` for the operator to correct; R2 additionally reports
the weaker same-day pairing as a fallback.

## 6. Straight-segment splitting (Q3)

`io/go2_quadric.straight_mask_go2` uses the **dual criterion** the sprint requires — body turn rate *and* RTK heading
rate — plus a motion and a locomotion-mode gate:
|mean ω_z| < 0.15 rad/s (IMU) **and** RTK |mean heading rate| < 8 °/s wherever the fix is OK **and** mean speed > 0.3 m/s
**and** `mode == 3`, with runs shorter than 4 s dropped.

**The averaging matters and is a genuine difference from the wheeled M1.** A trotting quadruped yaw-oscillates within
every step cycle: RMS |ω_z| is 0.23–0.31 rad/s on this corpus *even when walking dead straight*, so the RMS rule that
works for the wheeled M1 (`phase.registration.straight_mask_kinematic`) rejects everything here. The **windowed mean**
cancels the gait wobble and leaves the actual turn rate. Where the fix is not OK the RTK half is skipped and the IMU
criterion carries the segment; the skipped fraction is reported. Yield: 95–272 s of straight trot per session
(26–84 %), 10–22 runs each.

Gait phase comes from `phase/estimator.estimate_phase`, driven — since there is no joint stream — by the trot contrast of
the **vertical foot positions** (`gait_signal_from_columns`, added this sprint alongside the existing joint-based
`gait_signal`), with the phase origin fixed at LF stance onset from the foot-force contact flags. Measured gait period:
**0.44 s (2.26 Hz)** on the January sessions.

## 7. Ingest record

11 sessions, layout `raw/go2/<day>/<name>/{highlevel/,foot_imu/,fixposition/}` + `checksums.sha256` + `meta.yaml`
(provenance = "QUADRIC-GINS original_mirror 2026.1.5/2026.3.6"). Catalog rows in `docs/data_catalog.md`.
Parsed transcripts are cached as parquet under `data/processed/go2/<name>.parquet` (25–29 MB each; a cold parse is
~25–40 s per session, a cached load ~0.3 s).

---

# Block R results — GeoFDI on the corpus (pre-registration `go2_real_preregistration.md`, committed 17:50 before any run)

Experiment `experiments/e20_go2_quadric`, run `e20-20260816`. Settings exactly as pre-registered; nothing tuned after
seeing a result. Element = the 34 in-Z channels; cycles from the foot-height trot contrast; straight segments from the
dual criterion; H₀′ calibration = the first third.

## R1 — per-session H₀ / H₀′ (figure `e20_r1_go2_h0_h0prime.png`, table `e20_r1_sessions.csv`)

| result | value | vs pre-registration |
|---|---|---|
| naive H₀ (whole-element flip test) | **rejects on 11/11 sessions**, p ≈ 0.002 everywhere; per-window rejection 0.67–0.98 | **prediction 1 confirmed** (≥ 8/11 expected). A machine >1 yr in service with a 5 kg payload has a large, unmistakable natural left/right asymmetry |
| H₀′ per-window (pooled runs) | median rejection 0.45; sequential e-process alarms on **8/11** | **prediction 2 falsified as stated** — see R5 for the diagnosis |
| ν₀ | 7.2–21.9 (bootstrap sd 0.6–2.0) | recorded for R2 |
| anti-symmetric energy | carried by the **hind pair LH/RH in 6/11** sessions (median share 0.52 vs 0.44 front); dominant channel family **`foot_force`** in 11/11 | see P-LH below |

*Method correction made during this sprint (recorded because it changed a conclusion):* the first version of the
per-leg ranking standardised every channel by its own std before summing Π⁻ energy, which **breaks the identity that a
mirror pair's two partners carry equal anti-symmetric energy** and produced a spurious "front-left" ranking. R⁻ can rank
mirror *pairs*, never sides. The corrected (unnormalised) statistic reverses the answer to *hind-dominant*, and the side
is obtained separately from the signed mean (below).

## R5 — natural-anomaly hunt and the diagnosis of the H₀′ alarms (figure `e20_r5_diagnosis.png`)

**The diagnosis is clean: H₀′ alarms on 8/11 when the element pools several straight runs, but on only 2/11 when it is
built from a single run.** The ν-trajectory shape classifier agrees: **8/11 sessions are `boundary-jump`, 3 are
`flat/noisy`, none is `drift`** — i.e. the asymmetry level is stationary *inside* a run and steps *between* runs.

Interpretation: each straight run is a different traverse (different heading, different piece of ground, different part
of the site); the healthy asymmetry level ν depends on those conditions, so concatenating runs violates the H₀′ premise
that the calibration and monitoring windows come from the same regime. This is the same phenomenon as the M1 session
`173247` (Sprint 8), and it sharpens the deployment rule: **calibrate ν₀ per run (per continuous traverse), not per
session.** Reported as a falsified prediction with its cause, not as a fault detection.

*Honest limitation:* the within-run control could not be built for 4 of the 11 sessions (their longest single run yields
fewer than 12 cycles after the drop-first/drop-last trim), so the 2/11 figure is a lower bound over the 7 sessions where
the control exists.

### P-LH — the side of the natural asymmetry, and a falsified candidate cause
Side needs the **signed** mean difference (the energy cannot give it). The result is strikingly stable:
**LH − RH foot force is negative in 11/11 sessions (−0.04 … −0.14 standardised) and LF − RF is positive in 11/11
(+0.03 … +0.09)** — one consistent asymmetry pattern across three sites and two dates two months apart.

The hind pair does carry the larger share of Π⁻ energy in most January sessions, consistent with P-LH. **But the
candidate cause is falsified:** the March sessions carry **no foot-IMU board at all**, and their LH−RH asymmetry is if
anything *larger* (−0.10 … −0.11) than January's (−0.05 … −0.14, median −0.089). So the board and its cable are **not**
the source of the left-hind asymmetry; the remaining candidates are the payload placement and the robot's own wear.
This is exactly the sort of "known-true-cause" test the pre-registration asked for, and it came out negative — recorded.

## R2 — cross-period reproducibility (figure `e20_r2_cross_period.png`, tables `e20_r2_nu0.csv`, `e20_r2_cross.csv`)

| quantity | January (8 sessions) | March (3 sessions) |
|---|---|---|
| ν₀ range | 7.2 – 21.7 | 9.9 – 21.9 |
| within-site sd of ν₀ | site A 4.53, site B 3.11 | site C 6.38 |
| between-date ratio mean(Mar)/mean(Jan) | **1.18** | |

**Prediction 3 confirmed:** the two-month change of the mean asymmetry level (×1.18) is *smaller* than the
session-to-session spread within a single site and day (sd 3.1–6.4 on a mean of ≈ 14). The natural asymmetry of this
robot is therefore reproducible over two months to within its own run-to-run variability — the strongest reproducibility
statement in the project so far (Sprint 8's M1 evidence was single-day).

Cross-session H₀′ transfer (calibrate on session *i*, monitor session *j*) is **poor**, as the R5 diagnosis predicts:
window-reject 0.32 (Jan-calibrated → March) and 0.49 (March-calibrated → Jan), and same-site (0.33) is barely better
than different-site (0.37). Deployment consequence: ν₀ is reproducible as a *level*, but an H₀′ calibration is **not**
portable across runs/sessions — calibrate locally, compare ν₀ globally.

## R4 — estimator value (figure `e20_r4_estimator.png`, table `e20_r4_estimator.csv`)

The sprint plan made R4 conditional on a joint stream. There is none — **but the high-level API publishes
`foot_position_body`, which is exactly the body-frame contact point a contact-aided InEKF consumes**, so the experiment
ran after all (`estimate/pi_gating` with `use_provided_feet`, which now tolerates a robot with no joint columns).

| estimator | median ATE vs RTK | median ATE / path | nominal foot-measurement reject rate |
|---|---:|---:|---:|
| no gating | 2.96 m | 6.6 % | 0.000 |
| literature threshold (0.4 m/s + cov ×10) | 2.99 m | 6.5 % | **0.406** |
| GeoFDI-πᵢ hard | 2.96 m | 6.6 % | 0.000 |
| GeoFDI-πᵢ soft | 2.96 m | 6.6 % | **0.019** |

**The FAR contrast is decisive on real outdoor data: the fixed 0.4 m/s foot-speed threshold discards ~41 % of the
nominal contact measurements, the calibrated per-event gate ~2 % (≈ α).** This is the real-robot confirmation of the e16
simulation result — and it closes the gap left in Sprint 8, where Leg-KILO's slow walking trot was too gentle to expose
the threshold's over-rejection. **Honest null:** gating does *not* improve ATE here (2.96 vs 2.99 m); these straight
segments contain no slip severe enough to matter, so the gate's value on this corpus is the FAR guarantee, not accuracy.

*P-RTK caveat:* the reported dropped fraction (0.00–0.04) is low because the monitoring run was chosen to maximise
fix-OK coverage; the honest per-session RTK quality is the `fix_ok` column of §4 (site B 0.51–0.57 vs site C 1.00), which
does show the predicted site ordering.

## R6 — foot IMU as an independent phase check

Median touch-down phase error **21.2 % of a gait period** over the 7 usable January sessions, against a pre-registered
validation threshold of 10 % ⇒ **the check fails as stated**. Candidate causes, not separated here: (a) the foot-IMU
impact detector fires on both touch-down *and* lift-off (halving the period would map a lift-off onto ≈ 50 % error, and
a mixture would land near 20 %); (b) the LH contact flag is derived from a foot-force threshold whose onset lags the
mechanical impact; (c) a genuine phase-estimator error. Recorded as an open item; it does not affect R1–R5, which do not
use the foot IMU. The board is single-leg and was never a mirror channel.

## R3 — payload

Not run as a nuisance contrast: the payload was **constant (≈ 5 kg) across all 11 sessions**, so the corpus has no
payload variation. Recorded as a corpus-wide condition instead (see `go2_session_meta_form.md`).
