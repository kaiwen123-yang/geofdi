"""Wheeled-M1 telemetry schema (Sprint 7 Block W1): 16 joints = LF, RF, LH, RH x ABAD, HIP, KNEE, WHEEL.

Row = one control step: t, blk (rolling block index, -1 during warm-up / non-straight segments), theta (gait phase in
stepping mode, NaN in rolling mode), per joint
q/dq/tau_cmd/tau_meas (16 each, GeoFDI order), IMU a[3]/w[3] body frame, wheel-ground contact c[4], temp surrogate per
leg, commanded speed v_cmd, and NOT-in-Z diagnostics (base pose/velocity, wheel contact wrench fc/cp/tc).

Manifest (feeds groups.c2 unchanged): joint channels partner with the mirror leg's joint with sign JOINT_SIGN
(ABAD roll -1; HIP/KNEE/WHEEL pitch-axis +1 — the wheel rate is the y-component of a pseudovector and keeps its sign);
the WHEEL *angle* q is unbounded (rolling) and is excluded from the data element (`in_Z: false`), its rate and torque
stay; IMU polar/axial as for the Go2; contacts permute.

MJCF (MATRiX zgws) order is FAR, FBL, RAR, RBL x ABAD, HIP, KNEE, FOOT ("A" = right, "B" = left) = RF, LF, RH, LH; the
GeoFDI order is LF, RF, LH, RH — MJCF_TO_GEOFDI is that permutation (index i of the GeoFDI vector = MJCF index
MJCF_TO_GEOFDI[i]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("ABAD", "HIP", "KNEE", "WHEEL")
MIRROR_LEG = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
JOINT_SIGN = {"ABAD": -1, "HIP": +1, "KNEE": +1, "WHEEL": +1}
IMU_ACC_SIGN = {"x": +1, "y": -1, "z": +1}
IMU_GYRO_SIGN = {"x": -1, "y": +1, "z": -1}
MJCF_LEG = {"LF": "FBL", "RF": "FAR", "LH": "RBL", "RH": "RAR"}          # GeoFDI leg -> MATRiX prefix
MJCF_JOINT = {"ABAD": "ABAD", "HIP": "HIP", "KNEE": "KNEE", "WHEEL": "FOOT"}
MJCF_LEG_ORDER = ("FAR", "FBL", "RAR", "RBL")
MJCF_JOINT_ORDER = ("ABAD", "HIP", "KNEE", "FOOT")
MODEL_JOINTS = [f"{l}_{j}_JOINT" for l in MJCF_LEG_ORDER for j in MJCF_JOINT_ORDER]           # MJCF order (16)
GEOFDI_JOINTS = [f"{MJCF_LEG[l]}_{MJCF_JOINT[j]}_JOINT" for l in LEGS for j in JOINTS]         # GeoFDI order (16)
MJCF_TO_GEOFDI = np.array([MODEL_JOINTS.index(n) for n in GEOFDI_JOINTS])                       # geofdi_vec = mjcf_vec[MJCF_TO_GEOFDI]
GEOFDI_TO_MJCF = np.argsort(MJCF_TO_GEOFDI)
WHEEL_GEOMS = {l: f"{MJCF_LEG[l]}_wheel" for l in LEGS}
WHEEL_R = 0.096
NJ = 16


def joint_channels() -> list[str]:
    return [f"{grp}_{leg}_{j}" for grp in ("q", "dq", "tau_cmd", "tau_meas") for leg in LEGS for j in JOINTS]


def build_manifest(sim_meta: dict | None = None) -> dict:
    ch = []
    for grp in ("q", "dq", "tau_cmd", "tau_meas"):
        for leg in LEGS:
            for j in JOINTS:
                ch.append({"name": f"{grp}_{leg}_{j}", "group": grp, "leg": leg, "joint": j, "kind": "scalar-signed",
                           "partner": f"{grp}_{MIRROR_LEG[leg]}_{j}", "sign": JOINT_SIGN[j],
                           "in_Z": not (grp == "q" and j == "WHEEL")})           # wheel angle unbounded -> excluded
    for ax in "xyz":
        ch.append({"name": f"imu_a_{ax}", "group": "imu_acc", "leg": None, "joint": None, "kind": "polar", "frame": "body",
                   "partner": f"imu_a_{ax}", "sign": IMU_ACC_SIGN[ax], "in_Z": True})
    for ax in "xyz":
        ch.append({"name": f"imu_w_{ax}", "group": "imu_gyro", "leg": None, "joint": None, "kind": "axial", "frame": "body",
                   "partner": f"imu_w_{ax}", "sign": IMU_GYRO_SIGN[ax], "in_Z": True})
    for leg in LEGS:
        ch.append({"name": f"c_{leg}", "group": "contact", "leg": leg, "joint": None, "kind": "scalar-magnitude",
                   "partner": f"c_{MIRROR_LEG[leg]}", "sign": +1, "in_Z": True})
    for leg in LEGS:
        ch.append({"name": f"temp_{leg}", "group": "temp", "leg": leg, "joint": None, "kind": "scalar-magnitude",
                   "partner": f"temp_{MIRROR_LEG[leg]}", "sign": +1, "in_Z": False})
    ch.append({"name": "blk", "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    ch.append({"name": "theta", "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    ch.append({"name": "v_cmd", "group": "command", "leg": None, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    for name in ("base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz", "base_vx", "base_vy", "base_vz"):
        ch.append({"name": name, "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    for leg in LEGS:
        for pref, grp in (("fc", "contact_force"), ("cp", "contact_point"), ("tc", "contact_torque")):
            for ax in "xyz":
                ch.append({"name": f"{pref}_{ax}_{leg}", "group": grp, "leg": leg, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    return {"schema": "geofdi-m1-wheeled-telemetry-v1", "robot": "m1_wheeled", "leg_order": list(LEGS), "joint_order": list(JOINTS),
            "gait_group": {"G": "C2 sagittal reflection", "Sigma_rolling": "{(e,0),(g_s,0)}  (pure reflection, no phase)",
                           "Sigma_stepping": "{(e,0),(g_s,1/2)}", "delta_theta": 0.0},
            "mjcf_joint_order": MODEL_JOINTS, "geofdi_joint_order": GEOFDI_JOINTS, "channels": ch, "sim": sim_meta or {}}


def z_channel_names(manifest: dict) -> list[str]:
    return [c["name"] for c in manifest["channels"] if c["in_Z"]]


def all_columns() -> list[str]:
    return (["t", "blk", "theta"] + joint_channels() + [f"imu_a_{a}" for a in "xyz"] + [f"imu_w_{a}" for a in "xyz"]
            + [f"c_{l}" for l in LEGS] + [f"temp_{l}" for l in LEGS] + ["v_cmd"]
            + ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz", "base_vx", "base_vy", "base_vz"]
            + [f"fc_{ax}_{l}" for l in LEGS for ax in "xyz"] + [f"cp_{ax}_{l}" for l in LEGS for ax in "xyz"] + [f"tc_{ax}_{l}" for l in LEGS for ax in "xyz"])


def write_manifest_yaml(path: Path) -> None:
    """Export the manifest as the repo YAML (sim/manifests/m1_wheeled.yaml); the python builder is the source of truth."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("# generated by geofdi.sim.telemetry_m1.write_manifest_yaml — do not edit by hand\n" + yaml.safe_dump(build_manifest(), sort_keys=False))


def write_run(out_dir: Path, df: pd.DataFrame, manifest: dict, config: dict) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "telemetry.parquet", index=False)
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
