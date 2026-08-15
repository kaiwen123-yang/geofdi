# go2_urdf — Unitree `go2_description` (URDF) as the second GeoFDI simulation world

## Source and license
- `Go2_URDF.zip` uploaded by the project owner on 2026-08-15 (sha256 `690d6863…852c5`, see `PROVENANCE.json`);
  contents = the ROS package `go2_description` v1.0.0 (SolidWorks-to-URDF export, URDF dated 2023-10-11).
- `package.xml` declares `<license>BSD</license>`; Unitree publishes `go2_description` in `unitree_ros` under the
  BSD-3-Clause license. Kept verbatim: `urdf/go2_description.urdf` (+ the two collision-model PNGs), `xacro/`,
  `config/`, `meshes/*.dae`, `package.xml`, `CMakeLists.txt`.
- Dropped: `.history/` (editor backups), `dae/` (byte-identical duplicates of `meshes/`; `dae/base.dae` ==
  `meshes/trunk.dae`), `launch/`. Per-file sha256 in `PROVENANCE.json`.
- Visual meshes are converted dae → stl by `scripts/build_go2_urdf_worlds.py` (trimesh) into
  `$GEOFDI_DATA_ROOT/data/assets/go2_urdf/meshes_stl/` (hashes in `PROVENANCE.json`). The headless MJCF does not
  reference them (collision primitives only); attach them via `meshdir` if rendering is ever needed.

## Generated worlds (`mjcf/`, by `scripts/build_go2_urdf_worlds.py` + `geofdi.sim.urdf2mjcf`)
| file | what |
|---|---|
| `go2_urdf.xml` | ORIGINAL: every number from the URDF (explicit `fullinertia`, com), incl. base ixy = 1.2166e-4, ixz = 1.4849e-3, iyz = −3.12e-5, com_y = 0, and the URDF's FL-calf collision cylinder (r 0.012, x 0.008, pitch −0.21 vs r 0.013 / 0.01 / −0.20 on the other three legs). Mirror-chiral by construction: t01 fails at 4.9e-3 (ε_dyn measurement, Block G). |
| `go2_urdf_sym.xml` | SYMMETRIZED about y = 0: base I ← (I + E I E)/2 (ixy = iyz = 0; ixz kept), com_y = 0; left-leg collision primitives = mirror image of the right-leg ones. Leg inertials are exact mirrors in the URDF already (checked numerically). t01 passes (7e-11). |
| `scene_go2_urdf*.xml` | flat-floor scenes (floor friction as S0). |

Common to both: torque motors with the URDF effort limits (hip/thigh 23.7 N·m, calf 45.43 N·m; velocity limits
30.1 / 15.7 rad/s are not enforced by MuJoCo motors — recorded in the manifest), joint ranges from the URDF,
joint dynamics from `xacro/const.xacro` (damping 0.01, friction 0.2) plus armature 0.01 (rotor inertia, menagerie
value; a URDF cannot express it), IMU site at the URDF `imu` link (−0.02557, 0, 0.04232), sensors and keyframe as
in the S0 world, S0 foot-contact class (soft menagerie feet; `foot_contact: stiff` switch at load).

## Joint order / naming (== S0 telemetry manifest)
| telemetry | URDF joint | axis | range [rad] | effort [N·m] |
|---|---|---|---|---|
| LF_HAA / LF_HFE / LF_KFE | FL_hip_joint / FL_thigh_joint / FL_calf_joint | x / y / y | ±1.0472 / [−1.5708, 3.4907] / [−2.7227, −0.83776] | 23.7 / 23.7 / 45.43 |
| RF_* | FR_* | same | same | same |
| LH_* | RL_* | same | [−0.5236, 4.5379] for the thigh | same |
| RH_* | RR_* | same | same as LH | same |

Uniform-axis convention on all four legs (S = diag(−1, +1, +1) mirror), identical to the menagerie world, so the
C₂ representation (`assets/channels_m1.yaml`) is unchanged.

## Differences to the menagerie world (`assets/unitree_go2/go2_sym.xml`, S1–S3 baseline)
- Calf: menagerie merges calf + foot + calf-lower parts into one body of 0.2414 kg (com z −0.141); the URDF has calf
  0.154 kg + foot 0.040 kg (calf-lower links massless) → 0.194 kg per calf, total robot 15.019 kg vs 15.207 kg.
- Inertias given as `fullinertia` straight from the URDF (menagerie: eigen-decomposed `diaginertia` + quat).
- Joint damping 0.01 (URDF/xacro) vs 2.0 (menagerie): the URDF world's symmetric trot orbit is stable but attracts
  much more slowly (half-period mirror residual 1e-3 rad at 5 s, 6e-5 at 10 s, 1e-7 at 20 s → use 20 warm-up cycles)
  and has an under-damped body-pitch / hind-knee mode with a period of ≈ 2 gait cycles (see the Block G MANIFEST:
  negatively correlated cross-cycle mirror differences → conservative flip test). `SimConfig.joint_damping` overrides
  the value at load time (diagnostic).
- Everything else (kinematics, joint ranges, motor limits, foot geometry, IMU site) is identical.
