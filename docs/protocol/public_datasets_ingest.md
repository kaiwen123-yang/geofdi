# Public legged-robot datasets — download & ingest record (`raw/public/`)

Date: 2026-08-15. Downloads to `$GEOFDI_DATA_ROOT/scratch/` (Google Drive via `gdown`, GitHub via
`git`), then `scripts/ingest_session.sh` into `raw/public/<dataset>/<session>/` (payload checksums,
`meta.yaml`, catalog rows). Dataset-level `SOURCE.md` (URL, version, license, sha256 of the
originals) sits in each `raw/public/<dataset>/`. Audits were done with the pure-python `rosbags`
reader (`~/venvs/geofdi`), no ROS 1 install needed. `liu-a1-fault` has its own audit:
`docs/protocol/liu_a1_audit.md`.

## street-a1 — Cerberus "street" sequence (Unitree A1)

| item | value |
|---|---|
| origin | S. Yang et al., *Cerberus: Low-Drift Visual-Inertial-Leg Odometry for Agile Locomotion*, ICRA 2023; repo <https://github.com/ShuoYangRobotics/Cerberus> (GPL-3.0), datasets folder <https://drive.google.com/drive/folders/13GsFDaBkDrslOl9BfE4AJnOn3ECDXVnc> |
| file | `Street/street.bag` (Drive id `1rVQW3VPx9WwpJAh8vWKELD0eW9yI_8Vu`, 5.66 GB, ROS 1 bag, recorded 2022-04-10) + `street_trajectory.jpg` |
| robot / scene | Unitree A1, suburban street, 260 m in 590 s (0.44 m/s average per the README) |
| duration | 585.5 s, 544 182 messages, 4 topics |
| `/hardware_a1/joint_foot` | `sensor_msgs/JointState`, 16 entries: `FL0 FL1 FL2 FR0 FR1 FR2 RL0 RL1 RL2 RR0 RR1 RR2` (hip, thigh, calf per leg; Unitree order FL, FR, RL, RR) + `FL_foot FR_foot RL_foot RR_foot`. Joint `position`/`velocity` populated; joint `effort` **all zero** (no torques). Foot entries: `velocity` = contact flag (0/1), `effort` = foot force (raw, ≈ 0–630), `position` = 0. Header stamps every 2.115 ms (≈ 473 Hz, jitter 0.02 ms); 263 308 msgs over 585.5 s (≈ 450 Hz average → ≈ 5 % dropped/gapped). |
| `/hardware_a1/imu` | `sensor_msgs/Imu` 263 308 msgs (≈ 473 Hz), accel in m/s² (z ≈ 9.6), gyro rad/s |
| cameras | `/camera_forward/infra{1,2}/image_rect_raw` 8 783 msgs each (15 Hz, RealSense IR) |
| gait | trot, period ≈ 0.59 s (267 samples at 473 Hz, thigh autocorrelation) |
| ground truth | none in this bag (no mocap; the repo's other sequences carry `/mocap_node/...`) |
| session | `raw/public/street-a1/cerberus_street/` |
| GeoFDI use | healthy trot on real hardware with mirror pairs (FL↔FR, RL↔RR), contacts and IMU at ≈ 470 Hz — nominal-symmetry (H0) calibration and Gate-1-style controller checks on a second platform; no torques, no faults. |

## legkilo-go1 — Leg-KILO dataset (Unitree Go1)

| item | value |
|---|---|
| origin | G. Ou et al., *Leg-KILO: Robust Kinematic-Inertial-Lidar Odometry for Dynamic Legged Robots*, IEEE RA-L 2024; dataset repo <https://github.com/ouguangjun/legkilo-dataset> (no license stated), Google Drive folder <https://drive.google.com/drive/folders/1Egpj7FngTTPCeQDEzlbiK3iesPPZtqiM> (Baidu mirror `pan.baidu.com/s/1ue5_3OwELXK8n7A_I-HWJw?pwd=kilo`) |
| files | 7 ROS 1 bags (9.5 GB): corridor 3.18 GB, park 2.61, slope 1.80, indoor 0.86, grass 0.64, rotation 0.24, running 0.22; ground truth `*_tum.txt` (TUM format) and `*_original.csv` for corridor/indoor/park(ing)/running |
| topics (all bags) | `/high_state` `unitree_legged_msgs/HighState` (Go1 SDK high-level state: 20 `motorState` slots — 12 used — with `q dq ddq tauEst temperature`; `footForce[4]`; on-board `imu` (quaternion, gyro, accel, rpy); `mode`, `gaitType`, `velocity`, `yawSpeed`, `bodyHeight`, `footPosition2Body`, own `stamp`); `/imu_raw` `sensor_msgs/Imu` (`imu_link`, m/s²); `/state_SDK` `nav_msgs/Odometry` (`sdk_odom → sdk_base_link_3d`, Unitree built-in estimator); `/points_raw` `sensor_msgs/PointCloud2` (Velodyne VLP-16, ≈ 10 Hz). Message definitions are embedded in the bags, so `HighState` decodes with `rosbags` after `get_types_from_msg`. |
| rates | HighState / state_SDK / imu_raw ≈ 498 Hz average (grass.bag: 764 Hz); recorded in bursts (bag-time median dt 0.03 ms, mean 2 ms) — use `HighState.stamp` / `header.stamp`, not bag time |
| sequences | corridor 445.9 s ("8"-shaped, few features), park 403.7 s (dynamic objects), indoor 119.9 s, running 32.7 s (≈ 1.5 m/s), slope 311.9 s (> 6 m height change), rotation 38.1 s (in place), grass 91.6 s (uneven); total 1 443.8 s ≈ 24 min |
| gait | trot; running.bag thigh autocorrelation peaks at 280/560 ms (period ≈ 0.56 s) |
| sessions | `raw/public/legkilo-go1/legkilo_{corridor,grass,indoor,park,rotation,running,slope}/` (bag + its ground-truth files; the second Drive file named `indoor_original.csv` has 4 376 rows = corridor's `corridor_tum.txt` and was stored as `legkilo_corridor/indoor_original_2.csv` — most likely corridor's original GT) |
| GeoFDI use | healthy trot/run on real hardware with `q dq tauEst` (torques!), foot forces, IMU at ≈ 500 Hz over varied terrain (flat corridor, slope, grass) — the best public nominal set for A3 (environment) and A4 (noise) style audits and for e02/e03-type illustrations; no faults. |

## Catalog

Rows are in `docs/data_catalog.md` (one per session, `sha256(8)` = payload fingerprint).
