"""M1-schema telemetry: column layout, channel manifest (feeds groups.c2), parquet I/O.

Row = one control step: t, theta (gait phase from the controller clock), per joint q/dq/tau_cmd/tau_meas
(leg order LF, RF, LH, RH; joint order HAA, HFE, KFE), IMU a[3]/w[3] in the BODY frame (noise is injected
in the body frame, before any mirroring), contact c[4], temp surrogate per leg; plus diagnostic base-state
columns that are NOT part of the data element Z (flagged in_Z: false in the manifest).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
MIRROR_LEG = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
JOINT_SIGN = {"HAA": -1, "HFE": +1, "KFE": +1}       # uniform-axis convention (theory Table tab:joint-signs)
IMU_ACC_SIGN = {"x": +1, "y": -1, "z": +1}           # polar vector a -> E a,      E = diag(1,-1,1)
IMU_GYRO_SIGN = {"x": -1, "y": +1, "z": -1}          # axial vector w -> -E w  (eq:imu-action)


def joint_channels() -> list[str]:
    cols = []
    for grp in ("q", "dq", "tau_cmd", "tau_meas"):
        for leg in LEGS:
            for j in JOINTS:
                cols.append(f"{grp}_{leg}_{j}")
    return cols


def build_manifest(include_groups=("q", "dq", "tau_cmd", "tau_meas", "imu_acc", "imu_gyro", "contact", "temp"),
                   sim_meta: dict | None = None) -> dict:
    """Channel manifest: for every channel its mirror partner and sign under the sagittal reflection g_s.

    (rho(g_s) Z)[partner] = sign * Z[channel]  — i.e. the mirrored robot's `partner` channel carries this
    channel's value times `sign`. Joint channels partner with the same joint of the mirror leg with the
    joint sign; IMU components partner with themselves (polar: E, axial: -E); contacts/temps permute legs.
    """
    ch = []
    for grp in ("q", "dq", "tau_cmd", "tau_meas"):
        if grp not in include_groups:
            continue
        for leg in LEGS:
            for j in JOINTS:
                ch.append({"name": f"{grp}_{leg}_{j}", "group": grp, "leg": leg, "joint": j,
                           "kind": "scalar-signed", "partner": f"{grp}_{MIRROR_LEG[leg]}_{j}",
                           "sign": JOINT_SIGN[j], "in_Z": True})
    if "imu_acc" in include_groups:
        for ax in "xyz":
            ch.append({"name": f"imu_a_{ax}", "group": "imu_acc", "leg": None, "joint": None, "kind": "polar",
                       "frame": "body", "partner": f"imu_a_{ax}", "sign": IMU_ACC_SIGN[ax], "in_Z": True})
    if "imu_gyro" in include_groups:
        for ax in "xyz":
            ch.append({"name": f"imu_w_{ax}", "group": "imu_gyro", "leg": None, "joint": None, "kind": "axial",
                       "frame": "body", "partner": f"imu_w_{ax}", "sign": IMU_GYRO_SIGN[ax], "in_Z": True})
    if "contact" in include_groups:
        for leg in LEGS:
            ch.append({"name": f"c_{leg}", "group": "contact", "leg": leg, "joint": None, "kind": "scalar-magnitude",
                       "partner": f"c_{MIRROR_LEG[leg]}", "sign": +1, "in_Z": True})
    if "temp" in include_groups:
        # Temperature surrogate: a slow monotone nuisance (theory Remark rem:temp) — recorded for audits, NOT a
        # per-cycle mirror atom: its within-cycle drift makes Z(theta) - Z(theta+1/2) systematically nonzero.
        for leg in LEGS:
            ch.append({"name": f"temp_{leg}", "group": "temp", "leg": leg, "joint": None, "kind": "scalar-magnitude",
                       "partner": f"temp_{MIRROR_LEG[leg]}", "sign": +1, "in_Z": False})
    for name in ("base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz", "base_vx", "base_vy", "base_vz"):
        ch.append({"name": name, "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic",
                   "partner": None, "sign": None, "in_Z": False})
    for leg in LEGS:                    # ground-truth foot world positions (sim diagnostics for the InEKF experiments)
        for ax in "xyz":
            ch.append({"name": f"foot_{ax}_{leg}", "group": "diagnostic", "leg": leg, "joint": None, "kind": "diagnostic",
                       "partner": None, "sign": None, "in_Z": False})
    for leg in LEGS:                    # controller reference (sim-only; M1's controller is opaque) -> R+ channel
        for j in JOINTS:
            ch.append({"name": f"qref_{leg}_{j}", "group": "qref", "leg": leg, "joint": j, "kind": "scalar-signed",
                       "partner": f"qref_{MIRROR_LEG[leg]}_{j}", "sign": JOINT_SIGN[j], "in_Z": False})
    return {"schema": "geofdi-m1-telemetry-v1", "leg_order": list(LEGS), "joint_order": list(JOINTS),
            "gait_group": {"G": "C2 sagittal reflection", "Sigma": "{(e,0),(g_s,1/2)}", "delta_theta": 0.5},
            "channels": ch, "sim": sim_meta or {}}


def z_channel_names(manifest: dict) -> list[str]:
    return [c["name"] for c in manifest["channels"] if c["in_Z"]]


def all_columns() -> list[str]:
    return (["t", "theta"] + joint_channels() + [f"imu_a_{a}" for a in "xyz"] + [f"imu_w_{a}" for a in "xyz"]
            + [f"c_{l}" for l in LEGS] + [f"temp_{l}" for l in LEGS]
            + ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz", "base_vx", "base_vy", "base_vz"]
            + [f"foot_{ax}_{l}" for l in LEGS for ax in "xyz"]
            + [f"qref_{l}_{j}" for l in LEGS for j in JOINTS])


def write_run(out_dir: Path, df: pd.DataFrame, manifest: dict, config: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "telemetry.parquet", index=False)
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def read_run(run_dir: Path):
    run_dir = Path(run_dir)
    df = pd.read_parquet(run_dir / "telemetry.parquet")
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    return df, manifest
