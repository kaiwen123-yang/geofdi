# Data catalog

One row per ingested raw session. Rows are appended automatically by
`scripts/ingest_session.sh`; fill `gait` / `terrain` / `notes` by hand when the
session name does not encode them. `sha256(8)` is the first 8 hex chars of the
sha256 of the session's `checksums.sha256` (a stable payload fingerprint).

| date | session | category | gait | terrain | sha256(8) | notes |
|------|---------|----------|------|---------|-----------|-------|
| 2026-08-15 | grufd-ftc_84ca180 | public/liu-a1-fault | trot | flat (provenance unresolved) | 859e7b42 | Liu et al. RA-L 2025 GRUFD-FTC, git 84ca180; 10 CSV, 100 Hz rows (README says 50), knee faults η∈{0.4,0.6}; audit: docs/protocol/liu_a1_audit.md |
| 2026-08-15 | cerberus_street | public/street-a1 | trot | suburban street, outdoor | b34a01a5 | Cerberus (Yang et al. ICRA 2023) street.bag, 585.5 s; A1 joint_foot 12 q/dq (effort zero) + 4 foot contact/force, IMU ~473 Hz, IR stereo 15 Hz; no GT; record: docs/protocol/public_datasets_ingest.md |
| 2026-08-15 | legkilo_corridor | public/legkilo-go1 | trot | indoor corridor, 8-shaped path | 0c432f42 | Leg-KILO (Ou et al. RA-L 2024) 445.9 s; Go1 HighState q/dq/tauEst/footForce/IMU ~500 Hz, VLP-16 10 Hz; GT corridor_tum.txt (+indoor_original_2.csv); record: docs/protocol/public_datasets_ingest.md |
| 2026-08-15 | legkilo_grass | public/legkilo-go1 | trot | outdoor grass, uneven | dd983c59 | Leg-KILO 91.6 s; HighState/imu ~764 Hz in this bag; no GT |
| 2026-08-15 | legkilo_indoor | public/legkilo-go1 | trot | indoor, static non-uniform | bb9a6f6b | Leg-KILO 119.9 s; GT indoor_tum.txt / indoor_original.csv |
| 2026-08-15 | legkilo_park | public/legkilo-go1 | trot | semi-open parking lot, dynamic objects | 38ea9f08 | Leg-KILO 403.7 s; GT parking_tum.txt / parking_original.csv |
| 2026-08-15 | legkilo_rotation | public/legkilo-go1 | trot (in place) | flat | 871abf34 | Leg-KILO 38.1 s rotation in place; no GT |
| 2026-08-15 | legkilo_running | public/legkilo-go1 | running trot ~1.5 m/s | flat, short circle | eb6df5f3 | Leg-KILO 32.7 s; GT running_tum.txt / running_original.csv; period ~0.56 s |
| 2026-08-15 | legkilo_slope | public/legkilo-go1 | trot | outdoor slope, >6 m height change | eb5b196c | Leg-KILO 311.9 s; no GT |
