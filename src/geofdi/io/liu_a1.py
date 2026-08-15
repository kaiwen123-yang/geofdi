"""Loader for the Liu et al. (RA-L 2025, GRUFD-FTC) Unitree A1 joint-partial-failure dataset (Sprint 7 Block E).

CSV rows (no header, 69 numeric fields + trailing comma; 100 Hz, docs/protocol/liu_a1_audit.md §3):
  0-2 body angles (yaw, pitch, roll) | 3-5 body rates (wx, wy, wz) | 6-17 eta (torque retention, 1 = healthy) |
  18-29 q | 30-41 q_des | 42-53 dq | 54-65 dq_des | 66-68 body command (vx, vy, wz)
Official joint order (paper): 0-2 LF, 3-5 LH, 6-8 RF, 9-11 RH x (hip, thigh, calf) -> GeoFDI LF, RF, LH, RH x
HAA, HFE, KFE. Uniform-axis convention is consistent with the data (mirror hips have opposite mean, audit §2), so the
manifest signs are HAA -1 / HFE +1 / KFE +1. Data element channels: q, q_des, dq, dq_des (48; joint kinds), body pitch
angle (+1), roll angle (-1, axial), body rates (wx -1, wy +1, wz -1). Excluded: yaw angle (unbounded heading), the
command (exogenous; used for segmentation), eta (labels).

    df, manifest, labels = load_liu_file(path)          # df: GeoFDI-style frame with t (s), theta NaN, channels above
    episodes = fault_episodes(labels)                    # list of (row_start, row_end, joints, eta) fault episodes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
MIRROR_LEG = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
JOINT_SIGN = {"HAA": -1, "HFE": +1, "KFE": +1}
OFFICIAL_LEG_BLOCK = {"LF": 0, "LH": 3, "RF": 6, "RH": 9}          # start index of the (hip, thigh, calf) block in the 12-vector
GEOFDI_INDEX = [OFFICIAL_LEG_BLOCK[l] + j for l in LEGS for j in range(3)]     # geofdi joint i <- dataset joint GEOFDI_INDEX[i]
CALF_INDEX = {2: "LF", 5: "LH", 8: "RF", 11: "RH"}
DT = 0.01


def build_manifest_liu() -> dict:
    ch = []
    for grp in ("q", "q_des", "dq", "dq_des"):
        for leg in LEGS:
            for j in JOINTS:
                ch.append({"name": f"{grp}_{leg}_{j}", "group": grp, "leg": leg, "joint": j, "kind": "scalar-signed",
                           "partner": f"{grp}_{MIRROR_LEG[leg]}_{j}", "sign": JOINT_SIGN[j], "in_Z": True})
    ch.append({"name": "body_pitch", "group": "body_angle", "leg": None, "joint": None, "kind": "polar", "partner": "body_pitch", "sign": +1, "in_Z": True})
    ch.append({"name": "body_roll", "group": "body_angle", "leg": None, "joint": None, "kind": "axial", "partner": "body_roll", "sign": -1, "in_Z": True})
    for ax, sg in (("x", -1), ("y", +1), ("z", -1)):
        ch.append({"name": f"imu_w_{ax}", "group": "imu_gyro", "leg": None, "joint": None, "kind": "axial", "partner": f"imu_w_{ax}", "sign": sg, "in_Z": True})
    for name in ("body_yaw", "cmd_vx", "cmd_vy", "cmd_wz"):
        ch.append({"name": name, "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic", "partner": None, "sign": None, "in_Z": False})
    return {"schema": "geofdi-liu-a1-v1", "leg_order": list(LEGS), "joint_order": list(JOINTS),
            "gait_group": {"G": "C2 sagittal reflection", "Sigma": "{(e,0),(g_s,1/2)}", "delta_theta": 0.5}, "channels": ch,
            "sim": {"source": "Liu et al. RA-L 2025 GRUFD-FTC dataset (Gazebo simulation, legged_control NMPC+WBC)", "rate_hz": 100}}


def load_liu_file(path: str | Path):
    X = np.loadtxt(path, delimiter=",", usecols=range(69))
    n = len(X); t = np.arange(n) * DT
    df = pd.DataFrame({"t": t, "theta": np.nan})
    for grp, base in (("q", 18), ("q_des", 30), ("dq", 42), ("dq_des", 54)):
        for gi, leg in enumerate(LEGS):
            for j, jn in enumerate(JOINTS):
                df[f"{grp}_{leg}_{jn}"] = X[:, base + GEOFDI_INDEX[3 * gi + j]]
    df["body_yaw"] = X[:, 0]; df["body_pitch"] = X[:, 1]; df["body_roll"] = X[:, 2]
    df["imu_w_x"] = X[:, 3]; df["imu_w_y"] = X[:, 4]; df["imu_w_z"] = X[:, 5]
    df["cmd_vx"] = X[:, 66]; df["cmd_vy"] = X[:, 67]; df["cmd_wz"] = X[:, 68]
    eta = X[:, 6:18]                                       # dataset joint order (official)
    labels = pd.DataFrame(eta, columns=[f"eta_{k}" for k in range(12)]); labels["t"] = t
    return df, build_manifest_liu(), labels


def fault_episodes(labels: pd.DataFrame) -> list[dict]:
    """Contiguous rows with a constant faulty pattern (joint indices in the dataset order and their eta)."""
    eta = labels[[f"eta_{k}" for k in range(12)]].to_numpy(); faulty = eta < 0.999
    eps = []; prev = None; start = 0
    for i in range(len(eta)):
        key = (tuple(int(j) for j in np.where(faulty[i])[0]), tuple(float(v) for v in np.round(eta[i, faulty[i]], 3)))
        if key != prev:
            if prev is not None and len(prev[0]):
                eps.append({"row_start": start, "row_end": i, "joints": list(prev[0]), "eta": list(prev[1])})
            prev = key; start = i
    if prev is not None and len(prev[0]):
        eps.append({"row_start": start, "row_end": len(eta), "joints": list(prev[0]), "eta": list(prev[1])})
    for e in eps:
        legs = [CALF_INDEX.get(j, f"j{j}") for j in e["joints"]]
        e["legs"] = legs
        if len(legs) == 1:
            e["cls"] = "single"
        else:
            pair = tuple(sorted(legs))
            e["cls"] = {("LF", "RF"): "mirror", ("LH", "RH"): "mirror", ("LF", "LH"): "same_side", ("RF", "RH"): "same_side",
                        ("LF", "RH"): "diagonal", ("LH", "RF"): "diagonal"}.get(pair, "other")
    return eps


def command_segments(df: pd.DataFrame, tol: float = 1e-6) -> list[dict]:
    """Maximal runs of constant body command (vx, vy, wz); 'straight' = vy = wz = 0 (vx may be 0: trot in place)."""
    cmd = df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(); ch = np.where(np.any(np.abs(np.diff(cmd, axis=0)) > tol, axis=1))[0] + 1
    bounds = np.concatenate([[0], ch, [len(df)]]); segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        vx, vy, wz = cmd[a]; segs.append({"row_start": int(a), "row_end": int(b), "vx": float(vx), "vy": float(vy), "wz": float(wz), "straight": bool(abs(vy) < tol and abs(wz) < tol)})
    return segs
