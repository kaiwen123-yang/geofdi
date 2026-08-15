# Vendored Unitree Go2 (MuJoCo Menagerie)

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2
Commit: `da76818e269b82289eba39808e2fb91d679d6994` (fetched 2026-08-15, sparse checkout). License: BSD-3-Clause (`LICENSE`).

Files:
- `go2.xml`, `LICENSE`, `CHANGELOG.md` — verbatim copies for provenance (`go2.xml` references the
  30 MB of visual OBJ meshes in `assets/`, which are **not** vendored; run
  `scripts/fetch_menagerie_go2.sh` to fetch them if you want to render).
- `go2_sym.xml` — the model GeoFDI loads: visual meshes stripped (headless), the FL calf upper collision
  cylinder symmetrized to the FR/RL/RR values and the base inertia projected onto its mirror-symmetric part
  (source ixy = 1.2e-4, iyz = -3.1e-5 — the two left/right asymmetries in the source; leg masses, inertias and
  joint conventions mirror exactly to 1e-19), IMU sensors added at the `imu` site.
- `scene_flat.xml` — flat ground plane + `go2_sym.xml`.

Joint conventions (uniform-axis): abduction axis (1,0,0), thigh/calf axis (0,1,0) for all four legs, so the
sagittal mirror acts on a leg's (HAA, HFE, KFE) as diag(-1, +1, +1). Actuators are torque motors
(±23.7 N·m hips/thighs, ±45.43 N·m calves). Leg order in the model: FL, FR, RL, RR (GeoFDI's M1 schema
uses LF, RF, LH, RH — mapped in `geofdi.sim.telemetry`).
