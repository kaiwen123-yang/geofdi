# M1 (wheeled-legged, GENISOM zsm-1w / MATRiX `zgws`) — Day-0 protocol

Purpose: turn the wheeled M1 into a GeoFDI data source in one session — fix the channel semantics that the simulator
world (`m1_wheeled(_sym)`, Sprint 7 W1) could only assume, record the first nominal corpus for the rolling mode
(Σ = G), and run the pipeline once. **No fault is injected on Day 0.** Everything recorded goes through
`scripts/ingest_session.sh` (raw is immutable; catalog row per session).

## 0. Before leaving the lab
- `run_pipeline.sh` runs on the synthetic rehearsal session (Sprint 7 W4): `scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/sim/m1_rehearsal/20260816_rolling_1.0_flat_out_rep01 --robot m1 --mode rolling`.
- Laptop: `GEOFDI_DATA_ROOT` exported, venv `~/venvs/geofdi`, `PYTHONPATH=` cleared (host ROS leaks), free disk ≥ 50 GB.
- SDK/recording path decided: GENISOM GEN2 SDK `JointStateData{names, positions, velocities, efforts}` + IMU message
  → CSV export in the layout of `io/m1_mapping.yaml` (`joint_states.csv` with `<name>_pos/_vel/_eff` columns and, if
  the low-level command is available, `<name>_cmd`; `imu.csv`; `cmd.csv` with the body velocity command; `meta.yaml`),
  **or** rosbag2 of `/joint_states` + IMU + cmd (as the August legacy bags) converted with `scripts/m1_bag_tools`.

## 1. Channel census (30 min) — archive the raw names verbatim
1. Record 30 s of standing + 30 s of slow rolling with **everything** on: `ros2 topic list -t`, `ros2 topic hz` per topic,
   `ros2 bag record -a`; if SDK: dump one `JointStateData` and one IMU message with the `names` array *as strings*.
2. Decide and write into `meta.yaml` / `io/m1_mapping.yaml` (flip `unverified: false` only after §2):
   - joint count and names (expected 16: `fl1..fl4, fr1..fr4, bl1..bl4, br1..br4`; index 4 = wheel — candidate mapping
     1 = hip roll (ABAD), 2 = hip pitch (HIP), 3 = knee (KNEE), 4 = wheel (WHEEL));
   - wheel encoder: absolute angle (unbounded, wraps?) or only velocity; units (rad, rad/s);
   - `efforts` semantics: `unknown | current_estimate | torque` and units — this decides whether `tau_meas` is a torque
     (the residual channel needs a torque-consistent quantity; the *commanded* torque, if the vendor exposes it, is
     what the residual R⁻ uses — Sprint 4 Finding 1);
   - IMU: frame (base origin? offset?), quaternion order (wxyz/xyzw), gyro/accel units, gravity sign convention
     (specific force +g at rest on z?);
   - contact quantity: none expected on the wheeled M1 (`c_*` = NaN); wheel motor current is the proxy;
   - motor temperatures if published (they go to `temp_*`, never into Z);
   - clocks: message stamps vs receive time; rate (expected 200 Hz).
3. Sign test (**per joint, by hand, with photos**): move each joint slowly in the positive uniform-axis direction
   (ABAD: roll the leg outwards on the LEFT legs is +y rotation … define with a picture; HIP/KNEE: pitch forward = +y
   rotation about the lateral axis; WHEEL: forward rolling = +y rate) and read the sign of the reported position/velocity.
   Fill `signs.per_leg_sign` in `io/m1_mapping.yaml`. The August legacy bags suggested **all four joints flip between
   mirror legs** (vendor frame): expect `RF/RH = [-1,-1,-1,-1]` — but *measure* it (Part 0 Table tab:joint-signs remark).
4. Static rest, 10 min, robot on the ground and powered (motors enabled, no command): IMU biases and noise floors,
   encoder noise, effort noise; mirrored-channel noise comparison (A4 audit, Part 0 §audit).

## 2. Nominal corpus (rolling, ~60 min)
Flat, level floor (mark a 20 m straight lane; direction OUT = away from the door, BACK = towards it).
| block | what | reps | notes |
|---|---|---|---|
| R1 | straight rolling 0.5 m/s, ≥ 30 s each, OUT and BACK | ≥ 10 per direction | keep the joystick command constant (the pipeline cuts blocks on the command plateau) |
| R2 | 1.0 m/s | ≥ 10 per direction | |
| R3 | 2.0 m/s (or the max the lane allows) | ≥ 10 per direction | |
| R4 | in-place turning, both senses, 20 s | 3 + 3 | not a GeoFDI element (Σ-breaking by command); for the mapping/sign check |
| R5 | if the vendor stack has a stepping/trot gait: trot in place ≥ 10 × 30 s | ≥ 10 | Σ ⊂ G × S¹ mode; phase from `phase/estimator.py` (knee/hip signal) |
| N1 | symmetric payload (1 kg centred on the trunk) rolling 1.0 m/s | 5 + 5 | nuisance row |
| N2 | temperature sweep: repeat R2 after 20 min of driving | 5 + 5 | `motor_temp_start/end` in meta |
Session names: `<YYYYMMDD>_rolling_<speed>_flat_<out|back>_repNN` (D001). Every session: `ingest_session.sh <dir> m1/nominal <name>`
then `run_pipeline.sh` — read `report.md`. **The primary test in rolling mode is H₀′ (asymmetry CHANGE), not naive H₀.**
On a healthy real robot naive H₀ (the whole-session flip test) is EXPECTED to reject — the healthy loop is stably
asymmetric (real ε_dyn) and fixed-duration blocks are serially correlated at short L (Sprint 7 Block W). This was
confirmed on the first M1 hardware corpus (2026-08-10): all three rolling sessions rejected naive H₀ (p 0.002–0.006)
while the sequential H₀′ e-process stayed silent (max < 1/α) — `docs/protocol/m1_h_data_audit.md` §13. Read: the H₀′
differenced/per-window p and its e-process (should not alarm within a healthy session), ν₀ (the robot's stable asymmetry
level: expect > 0 on hardware — that is the H₀′ regime, not a failure), and the lag-1 block correlation (raise L from 1 s
to 2 s if positive). Note the half-vs-half differenced test can over-read a slow within-session asymmetry drift
(session 173247), so prefer the sequential monitor for deployment.

## 3. Gate 4 — joint-level command access (needed for the residual channel and later fault injection)
With the robot **suspended (legs free) or lying on a box**: verify that joint-level torque/position commands can be sent
and the corresponding command is logged (`<name>_cmd`), one joint at a time, small amplitude. Record which interface
(SDK low-level / ROS topic), latency, and whether the vendor controller can be bypassed. No fault injection on Day 0.

## 4. First command
```
scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/m1/nominal/<session> --robot m1 --mode rolling
```
(`--residual delan_equiv` once a nominal corpus exists to train the M1 rolling DeLaN on hardware data; the sim model
`models/delan_m1/equiv_rolling_v1` is NOT transferable.)

## 5. Fill-in list (write into `docs/protocol/m1_day0_wheeled_report.md` at the end of the day)
joint names/order; wheel encoder type; efforts semantics + units; IMU frame/order/units; contact proxy; temperature
availability; clock/rate; per-joint sign table (with photos); rest-noise table; corpus inventory (sessions × directions ×
speeds); Gate-4 result; first `report.md` numbers (R⁻ p, ν₀).
