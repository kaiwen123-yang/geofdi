# Unitree Go2 — Day-0 protocol (trot, Σ = {(e,0), (g_s, ½)})

Purpose: first nominal trot corpus of the user's Go2 and the channel-semantics check for the CycloneDDS `LowState`
path; pipeline run once. **No fault is injected on Day 0.**

## 0. Before leaving the lab
- `run_pipeline.sh` runs on the synthetic rehearsal session (Sprint 7 W4): `scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/sim/go2_rehearsal/20260816_trot_0.5_flat_out_rep01 --robot go2 --mode trot`.
- Recording path: `ros2 bag record -a` on the Go2 topics (or the unitree_sdk2 python client subscribing to
  `rt/lowstate` at 500 Hz + `rt/lowcmd`) → CSV export in the layout of `io/go2_mapping.yaml`: `lowstate.csv` with
  `q_0..q_11, dq_0..dq_11, tau_est_0..tau_est_11, temp_0..temp_11, imu_qw..imu_qz, gyro_x..z, acc_x..z, foot_force_0..3`
  (Unitree motor order FR, FL, RR, RL × hip, thigh, calf), optional `lowcmd.csv` (`tau_0..tau_11` feed-forward torque or
  q_des/kp/kd), `cmd.csv` (high-level velocity command), `meta.yaml`.

## 1. Channel census (30 min)
1. `ros2 topic list -t`, `ros2 topic hz /lowstate` (expect 500 Hz), full `bag record -a` for 60 s standing + 60 s trot.
2. Decide and record (`io/go2_mapping.yaml` `unverified: false` only after the sign test):
   - motor_state count (20; first 12 = legs) and order (FR, FL, RR, RL — Unitree; GeoFDI mapping index
     `[3,4,5,0,1,2,9,10,11,6,7,8]`), units rad / rad/s / N·m;
   - `tau_est` semantics (estimated output torque → `tau_meas`); whether the low-level command torque is available
     (`lowcmd`): the residual channel needs τ_cmd (Sprint 4 Finding 1);
   - IMU: quaternion order (w,x,y,z expected), gyro/accel units, mounting frame (IMU offset −0.0256, 0, 0.0423 m in the
     URDF; the mirror action needs the body frame with y left);
   - foot_force[4]: order FR, FL, RR, RL, units (raw counts?) → contact threshold for `c_*` (pipeline default 20);
   - motor temperatures → `temp_*` (not in Z); clock/rate.
3. Sign test per joint (hand-move with motors damped, photos): hip roll +x, thigh/calf +y in the URDF convention on
   all four legs (the sim world = go2_description URDF, uniform-axis). Fill `signs.per_leg_sign`.
4. Static rest 10 min (motors on, standing) and suspended legs 5 min: noise floors, mirrored-channel comparison (A4).

## 2. Nominal corpus (trot, ~60 min)
| block | what | reps | notes |
|---|---|---|---|
| T1 | trot in place (zero command) ≥ 30 s | ≥ 10 | the strictly symmetric world; Gate 1/1b material |
| T2 | straight trot 0.3 / 0.5 / 0.8 m/s, ≥ 30 s, OUT and BACK | ≥ 10 per direction and speed | paired-direction protocol (A3) |
| T3 | standing 60 s | 3 | zero-motion reference for the residual floor |
| T4 | slow walk if the stack offers it | 5 | walk mode (Σ element (g_s, ½) still usable, Part 0 §gait) |
| N1 | symmetric payload 1 kg centred, T2 at 0.5 m/s | 5 + 5 | nuisance row |
| N2 | temperature sweep (repeat T2 after 20 min) | 5 + 5 | |
Session names `<YYYYMMDD>_trot_<speed>_flat_<out|back>_repNN`; `ingest_session.sh <dir> go2/nominal <name>`; then
`run_pipeline.sh` — read `report.md`: R⁻ whole-session p, window QQ, e-process, H₀′ (ν₀). Gate 1 (A2): the pipeline's
mirrored-command gap needs τ_cmd (`lowcmd`); Gate 1b (A5): the report's H₀′ ν₀ trend across sessions.

## 3. Gate 4 — joint-level command access
Robot on a stand / suspended: send small joint-level commands through `lowcmd` (one joint, small amplitude), confirm the
logged command matches, note the interface, latency and whether the sport-mode controller must be stopped.

## 4. First command
```
scripts/run_pipeline.sh $GEOFDI_DATA_ROOT/data/raw/go2/nominal/<session> --robot go2 --mode trot
```
(`--residual analytic|delan_equiv` needs the contact wrench (foot-force scalar → estimated wrench along the leg, or the
Pinocchio contact model) — hardware path TODO; the pipeline reports the residual channel as unavailable until then.)

## 5. Fill-in list (`docs/protocol/go2_day0_report.md`)
motor order/units; tau_est semantics; lowcmd availability; IMU frame/order/units; foot-force units + threshold; temps;
clock/rate; per-joint sign table (photos); rest-noise table; corpus inventory; Gate-4 result; first `report.md` numbers.
