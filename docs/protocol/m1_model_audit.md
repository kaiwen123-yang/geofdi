# M1 model audit — MATRiX / GENISOM open-source resources vs the point-foot M1 (Sprint 4, Block M)

Date 2026-08-15. Repositories cloned to `~/research/third_party/` (outside the geofdi repo):

| repo | commit | license | content relevant to M1 |
|---|---|---|---|
| `zsibot/matrix` (MATRiX simulator) | 6ec0b354 (2026-08-10, "Update v1.0.7") | BSD-3-Clause (ZsiBot 2025) | docs + demo media only; the runtime (UE5 + MuJoCo, robot MJCFs) is a Baidu-pan binary package. Robot keys: `go2`, `go2w`, `xgb` (= zsl-1, L1 point-foot), `xg2`, `xgw`, `xgw2`, `zgws` (= zsm-1w, **wheeled M1**), `zgwt`, `zgwsarm`. No point-foot M1 key. |
| `zsibot/MATRiX_Python_SDK` | 7c63590c (2026-05-14) | (repo README; part of the MATRiX release) | **MJCF models in git**: `model/zgws/zgws.xml` (wheeled M1, 16 joints, STL meshes 56 MB), `model/xgb` (L1 point-foot + `xg_b.urdf`), `go2`, `go2w`, `xgw`, custom scenes |
| `zsibot/genisom_model` | e6aa98e2 (2026-04-23) | BSD-3-Clause | URDF + meshes for `zsl-1` and `zsl-1w` (L1 series) — **no M1** |
| `zsibot/genisom_robot_sdk` (GEN2 SDK, M1 and later) | 02c05949 (2026-08-06) | (see repo) | joint naming of the real robot: `fl1..fl4`, `fr1..fr4`, `bl1..bl4`, `br1..br4` (wheel-leg, 4 joints per leg); `JointStateData{names, positions, velocities, efforts}` — order given at runtime by `names` |
| `zsibot/genisom_roamerx_open` (RoamerX Open) | 51d1e9d2 (2026-04-23) | BSD-3-Clause | navigation stack; `robots_dog_msgs`: `LowState.motor_state[20]`, `LegControlData.q[12]`, `LowLevelRobotState.q[12]` (Unitree-style layout; no joint-order statement) |

**Verdict on availability**: the only public M1 model is the WHEELED `zgws` MJCF (MATRiX_Python_SDK). There is no
point-foot M1 model in any public GENISOM repository. The user's machine and the uploaded STEP are point-foot.

## `zgws` (wheeled M1) MJCF — extracted structure
- 18 bodies, 17 joints (free + 16 hinges), 16 torque motors (`actuatorfrcrange` ±150 N·m, ctrlrange unset in the source),
  total mass **38.82 kg** (base 17.03; per leg: abad 0.198, hip/thigh 2.875, knee/calf 0.863 (RAR: 0.860), wheel 1.511).
- Joint order (MJCF and sensors): **FR (`FAR_*`), FL (`FBL_*`), RR (`RAR_*`), RL (`RBL_*`)** × `ABAD` (axis x, hip roll),
  `HIP` (axis y, hip pitch), `KNEE` (axis y), `FOOT` (axis y, wheel, unlimited). "A" = right side, "B" = left side.
  Sensor names use the Unitree convention (`FR_hip_pos`, `FR_thigh_pos`, `FR_calf_pos`, `FR_foot_pos`, …).
- Ranges (rad): ABAD right [−0.697, 0.523] / left [−0.523, 0.697] (mirror-consistent); HIP front [−2.442, 2.791] /
  rear [−2.791, 2.442]; KNEE ±2.791; wheel free.
- Kinematics (m): base → abad (±0.2698, ±0.065, 0); abad → hip pitch (±0.0587 fwd/back, ±0.045, 0) ⇒ hip pitch axis at
  x = ±0.3285, |y| = 0.110; hip → knee (0, ±0.0522, −0.26); knee → wheel axle (0, ∓0.0088, −0.28); wheel radius ≈ 0.096
  (mesh extents 0.192); base mesh extents 0.833 × 0.230 × 0.208.
- Inertias given (diaginertia + quat) for every link. IMU site at the base origin (0, 0, 0). No keyframe. No joint
  damping/armature/frictionloss in the source (all 0).
- **Chiral details** (relevant to H₀ / ε_dyn): base com y = +3.4 mm and base inertia quaternion with products;
  RAR knee link 3.3 g lighter than the other three (0.8598 vs 0.8631, inertia 0.01048/0.01019/0.00047 vs
  0.01053/0.01024/0.00048); everything else mirror-consistent.

## STEP dimension check (`中狗点足3D模型图.stp`, 61.7 MB, Creo AP203, sha256 d41a411f…4298b)
`scripts/m1_step_dims.py` (cadquery 2.7 / OCP 7.8; parse 600 s): units mm; bounding box 0.461 (x, lateral) ×
0.548 (y, DOWN: −0.088 … +0.460) × 0.893 (z, longitudinal); 6408 cylindrical faces; joint axes from clustered
cylinder axes (radius ≥ 12 mm):

| quantity | STEP (point-foot machine) | `zgws` MJCF (wheeled) | deviation |
|---|---|---|---|
| hip-roll (ABAD) axis lateral half-spacing | ±0.065 (z-parallel axes, r 0.055, 70/67 faces) | 0.065 | **0 %** |
| hip-pitch axis longitudinal half-spacing | ±0.3295 (x-parallel axes at y ≈ 0, r 0.046, 63/56 faces) | 0.2698 + 0.0587 = 0.3285 | **+0.3 %** |
| thigh length (hip-pitch axis → knee axis) | knee axes at (y 0.1836, z ±0.1457): √(0.1838² + 0.1838²) = **0.260** (thigh at 45°) | 0.260 | **0 %** |
| calf (knee axis → foot) | foot-pad axes (r 0.030) at (y 0.413–0.415, z ±0.375–0.379): **0.325** | 0.28 to the wheel axle (+ wheel r 0.096) | **+16 % ⚠** (different end effector) |
| foot | lateral-axis pad, r ≈ 0.030 | wheel r 0.096, mass 1.51 kg | different |
| lateral offset of the hip-pitch axis (0.110 in zgws) | not resolved by axis clustering (axis lines are projected; would need face-extent analysis) | 0.110 | — |
| overall length | 0.893 incl. feet | base mesh 0.833 | n/a |

**Reading**: the MATRiX wheeled model represents the real M1's trunk, hip-roll and hip-pitch layout and the thigh
to < 0.5 %; the calf/foot differ (wheel vs point foot; calf 0.325 vs 0.28 m). A point-foot M1 model must therefore be
derived from `zgws` by (i) removing the wheel joints, (ii) calf 0.325 m + foot pad r ≈ 0.03, (iii) new calf/foot masses
(unknown: the STEP has no mass), (iv) confirming the hip lateral offset. Everything else can be taken over.

## What was put into the repo (`src/geofdi/sim/assets/m1/`, `scripts/build_m1_candidate.py`)
- `zgws_source.xml` — verbatim MATRiX MJCF (BSD-3, GENISOM AI / ZsiBot; meshes not copied — 56 MB STL, on request
  from the third_party clone).
- `m1_wheeled_headless.xml` (+scene) — mesh-free derivative: base box from the STL extents, existing box collisions,
  wheels as cylinders (r 0.096), S0-style sensors, standing keyframe.
- `m1_pointfoot_candidate.xml` (+scene) — **candidate v0.1** point-foot M1: wheel joints removed (12 hinges), calf
  0.325 m and foot sphere r 0.03 from the STEP, wheel-link masses kept as placeholders (flagged), ±150 N·m motors.
  Checks (`results/m1_model_audit/m1_candidate_checks.json`): 60 s standing PD hold — stands (z 0.358–0.420 m, final
  0.377; roll ≤ 1.0°, pitch ≤ 12.3° transient from the keyframe with the longer calf, then settled); mirror check
  (mirrored initial state + mirrored PD hold, 3 s): q residual 1.5e-2 rad, base 1.3 cm — the model is **chiral** (base
  com_y, RAR knee mass, base products) — an ε_dyn candidate for M1 to be revisited once masses are real.
- **Not run**: e01 (no formal H₀ experiments on M1 until the Day-0 audit fills the real joint sign table).

## Joint table candidate for theory `00_notation` (sim side; real-robot audit to confirm)
| index | real-robot SDK name (wheel-leg) | MATRiX `zgws` joint | point-foot candidate | axis | mirror partner | sign under mirror (candidate) |
|---|---|---|---|---|---|---|
| 0–2 | `fr1, fr2, fr3` (+ `fr4` wheel) | `FAR_ABAD/HIP/KNEE(_FOOT)` | FR HAA/HFE/KFE | x / y / y | `fl*` | HAA −1, HFE +1, KFE +1 (uniform-axis MJCF; the vendor firmware may flip all four — legacy bags §4a) |
| 3–5 | `fl1, fl2, fl3` | `FBL_*` | FL | x / y / y | `fr*` | same |
| 6–8 | `rr1, rr2, rr3` (`br*` in the SDK) | `RAR_*` | RR | x / y / y | `rl*` | same |
| 9–11 | `rl1, rl2, rl3` (`bl*`) | `RBL_*` | RL | x / y / y | `rr*` | same |

MJCF/sensor leg order is FR, FL, RR, RL (Unitree-style); the GeoFDI telemetry order is LF, RF, LH, RH — a fixed
permutation `(1, 0, 3, 2)` in the mapping table. Sign convention on the real robot is *not* derivable from the sim
model (the legacy M1 bags showed all four joints flipping between mirror legs, i.e. vendor-frame signs) — Day-0 audit item.

## Open items
- Point-foot calf/foot masses and the hip lateral offset (STEP has no mass; offset needs face-extent analysis or the
  vendor URDF for the point-foot machine — ask GENISOM for the `zsm-1` (non-wheeled) description).
- Real joint sign/offset table (Day-0 audit) before any M1 H₀ experiment.
- MATRiX runtime (Baidu pan) not fetched — not needed for the model audit; would only add the embedded controller.
