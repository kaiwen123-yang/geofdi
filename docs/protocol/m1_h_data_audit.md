# M1 hardware data audit — the four `H:\m1_data` rosbag2 sessions of 2026-08-10 (Sprint 8 Block D1)

Date: 2026-08-16 · Tooling: `scripts/m1_bag_tools/` (`fix_metadata.py --check`, `bag_inventory.py --decode`), `geofdi.io.m1_rosbag`
(sqlite3 + `rosbags` typestore, no ROS), `scripts/m1_h_audit.py` (tables/figures in `$GEOFDI_DATA_ROOT/results/m1_h_audit/`) ·
Status: **audited, mapping fixed, all four ingested (see §12)**. This is the project's first own hardware corpus.

## 0. Provenance and access

* Source: `H:\m1_data` (USB volume "TECLAST", 141 GB used) — four rosbag2 directories written 2026-08-10 18:12–18:39 local
  (the recordings themselves start 17:20–17:32). Sessions are named by the operator `m1_static_*` / `m1_walk_*`; note that
  **"walk" means the robot drove on its wheels** (§4, §9) — there is no leg-stepping gait in any of them.
* `/mnt/h` was **not mounted in WSL** at audit time and `sudo mount -t drvfs H: /mnt/h` needs a password the agent cannot
  supply. The data were therefore staged **read-only through Windows** (`robocopy H:\m1_data G:\...\scratch\h_stage\m1_data`,
  exit code 1 = files copied, nothing else) and every H: original was hashed on the Windows side (`Get-FileHash -Algorithm
  SHA256`, list `scratch/h_stage/m1_data_source_sha256.txt`). The staged copies match those hashes bit for bit, and the
  ingest checksums (§12) are verified against the same list — so the raw copies are provably identical to the H: source.
  H: was never written. (When the user mounts H:, `sha256sum -c` against that list can be repeated on `/mnt/h/m1_data`.)
* Rules kept: no `/mnt/g`, `/mnt/h` literal in code (only this document and the sprint spec); everything below is
  reproducible with `scripts/m1_h_audit.py --out $GEOFDI_DATA_ROOT/results/m1_h_audit $GEOFDI_DATA_ROOT/data/raw/m1/{audit,nominal}/m1_*`.

## 1. Directory structure, format, size, recording time

| session | format | files | size | recorded (local, first joint-state header stamp) | duration on disk | readable |
|---|---|---|---:|---|---:|---|
| `m1_static_20260810_172037` | rosbag2 v5, sqlite3, CDR, no compression | `_0.db3`, `_1.db3`, `metadata.yaml` | 2.99 GiB | 2026-08-10 17:20:38 | 72.0 s | all |
| `m1_walk_20260810_172847` | same | `_0`, `_1`, `metadata.yaml` | 2.94 GiB | 17:28:47 | 70.6 s | all |
| `m1_walk_20260810_173028` | same | `_0`, `_1`, `_2`, `metadata.yaml` | 5.46 GiB | 17:30:29 | 129.9 s | all |
| `m1_walk_20260810_173247` | same | `_0`, `_1`, `_2` (0 B), `_3` (0 B), `metadata.yaml` | 3.73 GiB | 17:32:48 | 151.2 s per metadata; **76.1 s readable** | `_0` whole; `_1` partially corrupt (SQLite "database disk image is malformed" — btree pages ≥ 353 414 unreadable): a rowid-ordered scan recovers the first 26 559 rows ≈ 31 s of it; `_2`/`_3` are 0-byte |

Session 4 is a **source-level** truncation (the staged copy hashes equal the H: hashes; the 0-byte files are dated 18:39,
i.e. the copy to H: was cut off), not a transfer problem here. `fix_metadata.py --check` now reports it as "REPAIR NEEDED:
unreadable db3 … bag truncated" instead of crashing; the raw copy keeps the source files as they are (faithful copy);
`geofdi.io.m1_rosbag` reads whatever is readable and records the truncation in `extract_report.json`.
Sizes: ≈ 40 MB/s of data, dominated by the two RoboSense point clouds (96×900 points at 10 Hz each).

The three "walk" sessions are **consecutive segments of one continuous drive**: the vendor odometry ends `172847` at
(5.94, 9.47) m, starts `173028` there, ends it at (18.71, 15.40) and starts `173247` there (odom frame never reset).
Path lengths 30.0 / 82.5 / 48.9 m (readable part), i.e. ≈ 161 m of driving on 2026-08-10, plus 72 s of standing.

## 2. Bag reading and metadata

* Reader: `pip install rosbags` (pure Python) — used for CDR decoding only; the `.db3` files are opened directly with
  `sqlite3` (`geofdi.io.m1_rosbag.iter_messages`), so no `metadata.yaml` consistency and no ROS environment is required.
  This sidesteps the zenoh-recording metadata issue entirely (these bags are stock rosbag2 v5 / `humble` schema anyway:
  `fix_metadata.py --check` = OK for sessions 1–3; ROS-style type names, QoS strings present).
* Time base: `header.stamp` (sensor time) for every topic; receive stamps kept as `t_bag`. Header stamps lead the bag
  receive stamps by 0.3–0.9 ms (joint states, central IMU), ≈ 13 ms (the two lidar IMUs) and ≈ 8.5 ms (odometry) —
  all from one clock (no drift over the sessions), so cross-topic alignment on header stamps is safe to ≈ 1 ms.
* Extraction (`geofdi.io.m1_rosbag.extract_bag_session`, ≈ 3–5 s per bag): `joint_states.csv`, `imu.csv`, `imu_front.csv`,
  `imu_rear.csv`, `odom.csv` (+ `cmd.csv` if messages exist — none here), `meta.yaml`, `extract_report.json` under
  `$GEOFDI_DATA_ROOT/data/processed/m1/<session>/`. `load_m1_session(<raw bag dir>)` does this automatically, so
  `run_pipeline.sh <raw session> --robot m1 --mode rolling` works directly on the ingested bag.

## 3. Topics per session

All four sessions carry the same 11 topics (session 1: 9 — no odometry/cmd topics were created while standing):

| topic | type | rate (measured) | in the sessions | decoded / used |
|---|---|---:|---|---|
| `/joint_shm_controller/joint_states` | `sensor_msgs/JointState` | **200.0 Hz** (dt jitter 0.07 ms, max gap ≤ 10 ms, monotone) | all | 16 names, position/velocity/effort → `joint_states.csv` |
| `/imu_driver/imu_central` | `sensor_msgs/Imu` (frame `imu_link`) | **200.0 Hz** (jitter 0.01–0.05 ms) | all | accel (g), gyro (rad/s), orientation constant identity → `imu.csv` |
| `/front_lidar/imu`, `/rear_lidar/imu` | `sensor_msgs/Imu` (`rslidar_head`/`rslidar_tail`) | 200 Hz (jitter 0.1–0.5 ms, gaps ≤ 30 ms) | all | RoboSense internal IMUs, y-up mounting → `imu_front/rear.csv` (cross-check only) |
| `/odom/mc_odom` | `nav_msgs/Odometry` odom→base_link | 47.6 Hz median (irregular, gaps ≤ 37 ms) | walk sessions (0 msgs while standing) | vendor motion-controller odometry → `odom.csv` (reference, **not ground truth**, §7) |
| `/odom/slam_odom` | `nav_msgs/Odometry` | — | topic present, **0 messages** everywhere | no SLAM / localisation reference on 2026-08-10 |
| `/cmd_vel` | `geometry_msgs/Twist` | — | topic present, **0 messages** everywhere | no command signal → fallback segmentation (§10) |
| `/tf` | `tf2_msgs/TFMessage` odom→base_link | ≈ 28 Hz | all | duplicate of `mc_odom` pose |
| `/tf_static` | `tf2_msgs/TFMessage` map→odom (identity) | ≈ 28 Hz (re-published) | all | no imu_link / lidar extrinsics recorded |
| `/front_lidar`, `/rear_lidar` | `sensor_msgs/PointCloud2` 96×900 | 10 Hz | all | not extracted (LiDAR odometry reference = future option, §7) |

Per-session tables (message counts, measured rates, jitter, gaps, header−bag skew, first-stamp offsets) are in
`results/m1_h_audit/audit_tables.md` §timing and `inventory.md`. Timestamps are strictly monotone on every topic; the
first-stamp offsets between topics are ≤ 60 ms (start-up order), no gaps > 37 ms anywhere.

## 4. Determination checklist (定案清单)

| item | status | finding |
|---|---|---|
| joint count | **fixed** | 16 (`name[]` length 16 in every message; position/velocity/effort arrays all 16) |
| `names` verbatim (order = truth) | **fixed** | `fl1_hip_roll fl2_hip_pitch fl3_knee_pitch fl4_foot fr1_hip_roll fr2_hip_pitch fr3_knee_pitch fr4_foot bl1_hip_roll bl2_hip_pitch bl3_knee_pitch bl4_foot br1_hip_roll br2_hip_pitch br3_knee_pitch br4_foot` — leg order fl, fr, bl, br; joint order hip_roll (ABAD), hip_pitch (HIP), knee_pitch (KNEE), foot = **wheel** (identical to the legacy August bags) |
| q / dq presence & units | **fixed** | rad, rad/s (leg joints ±1.9 rad; wheel rate 4–8 rad/s while driving; static noise 3.8e-4 rad quantum) |
| efforts presence | **fixed** (values) / **vendor** (semantics) | present, ±50 peak; crouch knee ≈ 14–15, hip ≈ 8, hip_roll ≈ 4–6, wheel ≈ 1 while rolling; distribution/scale observations in §8 — **current vs torque not decidable from data alone** (`efforts_semantics: unknown` stays) |
| wheel encoder | **fixed** | angle **wrapped to [−π, π)** (2π jumps every revolution: 43–142 per session), NOT unwrapped; rate continuous; sign flips between left and right legs (vendor frame) |
| IMU topic / rate / units | **fixed** | `/imu_driver/imu_central` 200 Hz; accel in **g** (norm 0.987 at rest), gyro rad/s; orientation constant (1,0,0,0), covariances zero |
| IMU frame | **fixed by regression** (§6) | sensor axes = (x right, y back, z down); `R_body_from_sensor = [[0,−1,0],[−1,0,0],[0,0,−1]]` in `io/m1_mapping.yaml`; mounting offset from base origin **unknown** (no tf) |
| command | **absent** | `/cmd_vel` empty in all sessions → fallback rule §10 |
| contact quantity | **no such channel** | none (wheeled robot; wheel effort is the only proxy) |
| motor temperature | **no such channel** | not published in these bags (nor in the legacy ones) |
| localisation / reference pose | **partial** | `/odom/slam_odom` empty; `/odom/mc_odom` vendor odometry present at 47.6 Hz (§7) — reference for relative metrics only |
| timestamps | **fixed** | monotone; header−receive skew 0.3–0.9 ms (joints, IMU); cross-topic offsets ≤ 60 ms at start, no gaps > 37 ms |
| bag integrity | **fixed** | sessions 1–3 whole; session 4 truncated at the source (76 s of 151 s readable) |
| locomotion mode | **fixed** | **wheeled rolling** in all "walk" sessions (§9); standing in the static session; no leg-stepping stride anywhere |

## 5. Mapping decision (`io/m1_mapping.yaml`)

* Names: the candidate `fl1..fl4 → ABAD/HIP/KNEE/WHEEL` **matches** (`fl1_hip_roll` etc. — index 1 = hip roll, 2 = hip
  pitch, 3 = knee pitch, 4 = wheel); the mapping now stores `name_pattern: "{leg}{index}{suffix}"` with the verified
  suffixes and the loader also accepts the bare `fl1` form (SDK path / rehearsals). `unverified: false`, with a
  `verification:` block listing what was fixed and what stays open (efforts semantics, hand sign test, IMU offset).
* Signs: the vendor frame mirrors **all four joints** between left and right legs (session means, fl/fr and bl/br:
  hip_roll −0.52/+0.54, hip_pitch +1.86/−1.86, knee +1.60/−1.63, wheel rate −6.5/+6.8; efforts likewise), exactly the
  legacy §4a pattern. Because the GeoFDI mirror representation is diag(−1,+1,+1,+1) (roll flips, pitch-axis joints keep
  their sign), the vendor→uniform-axis conversion for the RIGHT legs is **`per_leg_sign RF = RH = [+1, −1, −1, −1]`** — the
  earlier note in the yaml ("[−1,−1,−1,−1]") would have flipped ABAD twice; corrected and recorded. After the loader the
  mirror check is clean: `q_ABAD` −0.520 vs sign·RF −0.537, `q_HIP` 1.856 vs 1.857, `q_KNEE` 1.605 vs 1.630, wheel rate
  −6.50 vs −6.79 (session `173028`; the residual differences are the real ε: right knee 0.02–0.05 rad more flexed, right
  wheels ≈ 4 % faster on a net-left-turning drive).
* IMU: `frame: sensor_RBD`, `R_body_from_sensor`, `accel_units: g` (→ ×9.80665), `gyro_units: rad/s`, orientation flagged
  constant. Odometry conventions recorded (`twist_linear_frame: odom`).
* What the mapping does NOT claim: the absolute positive direction of each joint (Day-0 §1.3 hand test with photos is
  still the way to fix it — only the L/R relation, which is all R⁻ needs, is verified); efforts semantics.

## 6. IMU frame determination

Regression of the (smoothed) central-IMU signals on kinematic quantities derived from the vendor odometry, per walk session
(`audit_tables.md` §IMU): sensor accel [g] ≈ fwd·(0.05, **−1.2 … −1.3**, 0) + lat·(**−1.5 … −1.9**, 0.1, 0.2–0.5) +
(0.01, 0.01, **−0.987**); sensor gyro ≈ yaw-rate·(0, 0, **−1.18 … −1.22**). Hence forward acceleration lands on −y,
leftward centripetal acceleration on −x, yaw-left rate on −z, gravity reaction on −z: the sensor frame is **(x right,
y back, z down)**, a proper rotation of the body FLU frame (180° about the body (1,−1,0)/√2 axis), with the standard
specific-force convention. The two lidar IMUs (y up) agree with the central one on the yaw-rate slope (1.17–1.23),
i.e. **the vendor odometry's yaw rate is ≈ 18–22 % low** relative to all three gyros (its heading is probably wheel-
odometry based) — one more reason to treat `mc_odom` as a reference, not truth. Rest: |a| = 0.987 g (a 1.3 % scale
shortfall to remember when feeding an observer), accel std 0.03/0.014/0.014 m/s², gyro bias (1.3e-3, 1e-4, 5e-5) rad/s,
gyro std ≈ 2e-3 rad/s (static session).

## 7. Odometry (`/odom/mc_odom`)

odom→base_link, 47.6 Hz median (dt jitter 0.6 ms), pose covariance all zeros, base height z ≈ 0.535–0.556 m, roll/pitch
std ≈ 0.5°; `twist.linear` is expressed in the **odom** frame (corr 0.999 with the world velocity, ≈ 0 with the body
velocity — not REP-105); `twist.angular.z` agrees with the pose yaw rate (0.88–0.93). Speed while driving: median
0.47 / 0.70 / 0.72 m/s, max 0.72 / 1.00 / 0.94 m/s (sessions 2/3/4). Effective wheel radius from speed / mean wheel rate:
**0.097 m** in all three sessions (the sim manifest uses 0.096 m — consistent, and it shows the odometry translation is
wheel-odometry-based). It is the only pose reference in these bags: use it for relative metrics (drift over straight
runs, loop consistency), and note that an independent reference (KISS-ICP on the recorded point clouds) is possible later.

## 8. Efforts — observations only (no conclusion; vendor confirmation pending)

* Static crouch (motors enabled): knee 13.5 / −10.1 / −11.5 / +11.5, hip −5.2 / +3.8 / +2.8 / −4.8, hip_roll −4.9 / −6.3 /
  −5.9 / −4.2, wheels ≤ 1.1 (fl, fr, bl, br). Rolling at 0.7 m/s: knee ≈ ±14–15, hip ≈ ±8–9.5, hip_roll ≈ ±4–5.5 (session
  means), wheel ≈ ±0.5–1.3; peaks up to 46 (hip_roll, during posture corrections) and 8.5 (wheel).
* Scale: a knee holding ≈ 15 in a 92° crouch is what a torque of 15 N·m would look like for a ≈ 50 kg robot with a
  ≈ 0.25 m thigh (≈ 120 N × 0.25 m × sin) — but a current reading of 15 A on a geared actuator with Kt ≈ 1 N·m/A is
  indistinguishable from the data. Effort noise at rest: std 0.3–1.9 (legs), 0.02–0.06 (wheels); the wheel effort
  regressed on wheel acceleration and rate explains only R² ≈ 0.1–0.25 (a ≈ 0.01–0.02 eff·s², b ≈ 0–0.19 eff·s/rad,
  constant ≈ ∓0.6–1.3 with the vendor sign) — a rolling-resistance-like offset plus noise, no clean back-EMF term.
* Effort/velocity correlation is weak on all joints (≤ 0.2). Nothing here distinguishes "estimated torque" from
  "current": **`efforts_semantics: unknown` stays until the vendor answers** (the sim world's `torque` semantics are not
  transferred to hardware).

## 9. Per-session verdicts

| session | regime (1-s windows) | usable for | not usable for | verdict |
|---|---|---|---|---|
| `m1_static_20260810_172037` (72 s) | standing 100 % (knee excursion ≤ 0.02 rad, wheels still) | rest-noise floors (§6, `audit_tables.md` rest-noise table), encoder quantum, IMU biases, mirrored-channel noise comparison (Part 0 A4 audit); note the **standing posture is asymmetric** (hip_roll fl −0.35 vs fr +0.61, knee 1.50 vs −1.76 — the vendor stand pose, not a calibration issue: the rolling posture is symmetric to 0.02–0.05 rad) | R⁻ tests (no motion, asymmetric pose), InEKF (no odometry), DeLaN | **archive / audit** → `raw/m1/audit/` |
| `m1_walk_20260810_172847` (70.6 s, 30 m) | rolling 90 %, standing 10 %; speed median 0.47 m/s | rolling H₀′ (5 straight runs, 30.6 s → ≈ 28 one-second blocks), rolling InEKF vs fixed-foot vs ESKF against `mc_odom` | DeLaN training (too short), naive H₀ at scale | **rolling H₀′ + InEKF (small)** → `raw/m1/nominal/` |
| `m1_walk_20260810_173028` (129.9 s, 82.5 m) | rolling 98 %; speed median 0.70 m/s | rolling H₀′ (12 straight runs, 83.2 s → ≈ 78 blocks) — the best session; InEKF comparison (long yaw loops, 304° heading range) | DeLaN training alone (needs ≥ 20 min) | **primary rolling H₀′ + InEKF session** → `raw/m1/nominal/` |
| `m1_walk_20260810_173247` (76.1 s readable, 48.9 m) | rolling 96 % | rolling H₀′ (5 runs, 38.4 s → ≈ 36 blocks); InEKF (76 s) | the lost 75 s | **rolling H₀′ + InEKF (truncated)** → `raw/m1/nominal/` |
| all three walk sessions | 153 s of straight rolling in 22 runs (2.4–14 s), ≈ 140 one-second blocks | pooled H₀′ calibration | equivariant DeLaN (D3.3 needs ≥ 20 min of nominal rolling — **not available on 2026-08-10**: 4.6 min in total) | |

## 10. Straight-segment fallback (no command signal)

`phase.registration.straight_mask_kinematic` (used automatically by `run_pipeline.sh` when `cmd.csv` is absent): a row is
straight rolling when, over a 0.5-s moving window, RMS|ω_z| < 0.15 rad/s (body yaw rate from the central IMU), RMS of the
left−right mean wheel-rate difference < 1.0 rad/s (≈ 0.10 m/s at r = 0.097 m) and mean |wheel rate| > 2 rad/s
(≈ 0.19 m/s); runs shorter than 2 s are dropped. Thresholds were placed between the driving and turning modes of the
measured distributions (RMS|ω_z| q50/q75/q90 ≈ 0.015 / 0.1–0.2 / 0.4–0.66 rad/s; RMS(L−R) q75 ≈ 0.6–2.4 rad/s; mean
|wheel| q25 ≈ 3.4–5.7 rad/s). Yield: 5 / 12 / 5 runs, 30.6 / 83.2 / 38.4 s (43 / 64 / 50 % of the sessions); the
excluded parts are the turning manoeuvres — the operator turned by *skid-steering with brief posture/leg adjustments*
(wheel-speed chatter ±2 m/s, knee excursions 0.2–0.4 rad, yaw-rate impulses 1–2 rad/s), visible in
`audit_timeline_*.png`. Within the kept runs |ω_z| ≤ 0.3 rad/s peak, RMS ≤ 0.06, wheel spread ≤ 0.12 m/s.

## 11. For the theory intake (numbers to carry)

* Real ε_dyn candidates on hardware (stable mirror residuals while rolling straight): knee angle asymmetry 0.02–0.05 rad
  (right more flexed), hip_roll 0.02–0.03 rad, wheel-rate asymmetry ≈ 4 % of the session mean (turning bias); efforts
  L/R differ by 0.4–1.1 (knee, hip_roll) — i.e. the "stably asymmetric healthy loop" of H₀′ is the hardware regime, as
  predicted in Sprint 7 Block W.
* ν₀ (H₀′ calibration asymmetry functional, first-half calibration, L = 1 s): **13.7 ± 7.3 / 2.8 ± 1.7 / 9.8 ± 5.1**
  (172847 / 173028 / 173247), significantly positive on every session (ν₀ / bootstrap-std ≈ 1.7–1.9).
* Session-mean gyro z is nonzero on every drive (−0.05 … −0.09 rad/s in the sensor frame): the operator's loops, not a bias
  (static bias 1e-4).
* IMU accel scale 0.987 g at rest; odometry yaw rate −18…−22 % vs the gyros: both are in the InEKF error budget.

## 13. First hardware runs (Block D3; results `results/pipeline/m1real_*`, `results/e18_m1_real/e18-20260816`)

Pre-registration `docs/protocol/m1_real_preregistration.md` (committed 2026-08-16 14:05, before any run). Settings as
pre-declared (N = 64, window = 10, α = 0.05, M = 512, seed 0, kinematic fallback segmentation, L = 1 s then 2 s).

**D3.1 — R⁻ H₀ / H₀′** (first real-robot readout, figure `results/e18_m1_real/e18-20260816/e18_real_h0_h0prime.png`):

| session | K (L=1) | naive H₀ p | H₀ e-proc | lag1 (L=1 → L=2) | H₀′ differenced p (L=1) | H₀′ per-window e-proc max | ν₀ ± std |
|---|---:|---:|---:|---:|---:|---:|---|
| 172847 | 28 | 0.002 | 34 (alarm) | 0.65 → 0.23 | 0.37 (in band) | 0.98 (no alarm) | 13.7 ± 7.3 |
| 173028 (primary) | 74 | 0.006 | 1462 (alarm) | 0.52 → 0.23 | 0.094 (in band) | 2.6 (no alarm) | 2.8 ± 1.7 |
| 173247 | 36 | 0.006 | 223 (alarm) | 0.79 → 0.55 | **0.002 (rejects)** | 5.8 (no alarm) | 9.8 ± 5.1 |

Against the pre-registration: prediction 1 (**naive H₀ rejects**) is confirmed on all three (as designed — the healthy
loop is stably asymmetric + blocks correlated at L = 1 s; even at L = 2 s the whole-session flip test still rejects,
p 0.002–0.043, but the H₀ e-process no longer alarms on the two shorter sessions and lag-1 drops to 0.23–0.55, confirming
prediction 4). Prediction 3 (**sequential H₀′ monitor: no alarm within any session**) holds on all three — the e-process
stays below 1/α = 20 everywhere (0.98 / 2.6 / 5.8). Prediction 2 (H₀′ differenced first-vs-second-half in band on ALL
three) is **partially falsified**: it holds on 172847 and the primary 173028 but **rejects on 173247 (p = 0.002)**. The
differenced rejection is a real *slow drift* of the asymmetry level over that 76-s session, not a fault — the increase is
concentrated in the right-front knee/ABAD channels (normalised anti-symmetric energy of q_RF_KNEE rises 0.6 → 3.6, q_RF_ABAD
0.6 → 2.9 between halves) and tracks the odometry turning bias (the operator's later runs curved more). Reported as
predicted-falsified: the deployable sequential H₀′ monitor did not alarm, but the two-window differenced test is sensitive
to this within-session non-stationarity — a note for the deployment (calibrate ν₀ on a short leading window, monitor
sequentially; the differenced half-vs-half test over-reads slow drift). No parameter was changed after seeing the data.

**D3.2 — rolling InEKF vs fixed-foot vs ESKF on hardware** (`experiments/e18_m1_real`, figure `e18_inekf_real.png`):
the heading-independent, pre-registered metrics are decisive — **path length recovered 0.99 / 1.00 / 0.99** (rolling
RIEKF) vs **0.04 / 0.03 / 0.04** (fixed-foot RIEKF/ESKF), and **per-run traveled-distance error median 0.6–1.4 %**
(rolling) vs **98 %** (fixed) — the real-robot version of the e10 sim result (rolling 0.71 m vs fixed 13.4 m). The rolling
ESKF matches the rolling RIEKF (path 1.05–1.08, arclen-err 0.6–1.4 %): the error-parametrisation effect is second order at
these rates, as pre-registered. The *vector* metrics (whole-session RMSE 17–48 m, per-run vector displacement ratio ≈ 1)
are dominated by a **shared yaw-corruption artifact**: during the operator's skid-steer turns the four wheels skid, which
violates BOTH contact models, and the four fixed body-frame contact points over-constrain the yaw update and drag it away
from the gyro (identical for all four filters, including the e10-validated fixed-foot ones) — the estimated heading ends
150–300° from the gyro-integrated heading. This is exactly the failure that per-stance πᵢ gating (Sprint 8 Block G) is
meant to remove (gate the kinematic update out while the wheel rolls-constraint residual is large), and it is called out
as such. Falsification clause (rolling per-run error < half the fixed-foot's) is met decisively on the heading-independent
metric (0.6–1.4 % vs 98 %). Reference is the vendor odometry, not ground truth (its yaw rate is 18–22 % low, §6).

**D3.3 — DeLaN not run** (pre-declared: 4.6 min nominal rolling < 20 min minimum). Residual channel stays off.

## 12. Ingest record

Sessions ingested from the hash-verified staged copies with `scripts/ingest_session.sh` (payload checksums over the db3
files + `metadata.yaml`, filled `meta.yaml`, catalog rows) and verified against the H: source hashes:

| session | raw path | payload fingerprint `sha256(8)` | source-hash check |
|---|---|---|---|
| `m1_static_20260810_172037` | `raw/m1/audit/` | `5ee91cfb` | OK (all db3 + metadata.yaml equal the H: Get-FileHash list) |
| `m1_walk_20260810_172847` | `raw/m1/nominal/` | `ab623e3e` | OK (all db3 + metadata.yaml equal the H: Get-FileHash list) |
| `m1_walk_20260810_173028` | `raw/m1/nominal/` | `2566ec71` | OK (all db3 + metadata.yaml equal the H: Get-FileHash list) |
| `m1_walk_20260810_173247` | `raw/m1/nominal/` | `4fc3c926` | OK (all db3 + metadata.yaml equal the H: Get-FileHash list) |

Processed derivatives (CSV extraction) live in `data/processed/m1/<session>/` and are regenerated on demand by the loader.

## 14. Diagnosis of the `173247` H₀′ rejection (Sprint 9 B5) — and a correction to §13

Sprint 8 §13 attributed the H₀′ half-vs-half rejection on `m1_walk_20260810_173247` (p = 0.002) to *"a real slow drift of
the asymmetry level"*. Sprint 9 re-examined it with the shape classifier and the pooled-vs-within-run control developed
on the much larger Go2 corpus (`go2_quadric_audit.md` §R5), and **that attribution was too strong**:

| session | straight runs | ν trajectory (10-block windows) | drift score | jump score | shape | H₀′ differenced p (pooled) | (within longest run) |
|---|---:|---|---:|---:|---|---:|---:|
| `…173247` | 5 (longest 12.4 s) | 28.6 → 9.1 → 34.6 | 0.83 | 2.34 | non-monotone excursion | **0.002** | 0.025 |
| `…173028` | 12 (longest 12.1 s) | 21.2 → 7.9 → 5.7 → 15.3 → 7.4 → 13.4 → 12.3 | 0.70 | 2.66 | boundary-jump | 0.092 | 0.025 |
| `…172847` | 5 (longest 14.1 s) | 12.7 → 26.6 | 4.00 | 2.00 | (only 2 windows) | 0.369 | 0.066 |

**Conclusion 1 — it is not a monotone drift.** The ν trajectory of `173247` falls and then rises by a factor of ~3.8
between consecutive windows (drift score 0.83, i.e. the linear trend explains less than one standard deviation across the
record). The Sprint-8 wording is corrected to: *a large between-window change of the asymmetry level*, whose shape matches
the **between-run condition change** mechanism established on the Go2 corpus, not a slow within-session drift.

**Conclusion 2 — the M1 corpus cannot settle it, and this is a data-collection limit, not an analysis gap.** The
within-run control is decisive on Go2 (H₀′ alarms 8/11 pooled vs 2/11 within one run) because Go2 runs are 30–60 s. On
this M1 corpus the longest straight run is 12–14 s = 12–14 one-second blocks, so a within-run differenced test has only
~6 pairs; it returns p ≈ 0.025–0.066 on *all three* sessions regardless of their pooled verdict, i.e. it is too weak to
discriminate. **Settling it needs continuous straight M1 runs of ≥ 60 s** (a lane long enough not to require a turn),
which is added to the robot-day list.

**Conclusion 3 — the deployment rule is unchanged and now doubly supported:** calibrate ν₀ **per continuous run**, not per
session, and use the sequential monitor rather than the half-vs-half differenced test for deployment. The half-vs-half
test over-reads a between-run change on both platforms.

Condition cross-check (unchanged from §13): the increase is concentrated in the right-front knee/ABAD channels and tracks
the odometry turning bias, consistent with a heading/ground-dependent asymmetry rather than a component fault.
