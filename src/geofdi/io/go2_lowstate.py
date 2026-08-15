"""Unitree Go2 LowState session loader (Sprint 7 Block W3): CSV export of the CycloneDDS LowState (motor_state[20],
IMU, foot_force) -> GeoFDI Go2 telemetry frame (sim.telemetry schema) + manifest + report.

    df, manifest, report = load_go2_session(session_dir)
    write_go2_session(session_dir, df_geofdi, meta)          # export a simulated rollout in the LowState layout (rehearsal)

Mapping (io/go2_mapping.yaml, `unverified: true`): Unitree motor order FR, FL, RR, RL x hip, thigh, calf -> GeoFDI
LF, RF, LH, RH x HAA, HFE, KFE; tau_est -> tau_meas (an output-torque estimate; the commanded torque comes from LowCmd
if recorded, else NaN); foot_force -> contact flag c (force > threshold); phase column NaN (the kinematic phase
estimator fills it downstream). Missing columns -> NaN + listed.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..sim.telemetry import JOINTS, LEGS, all_columns, build_manifest


def load_mapping(path: str | Path | None = None) -> dict:
    p = Path(path) if path else resources.files("geofdi.io").joinpath("go2_mapping.yaml")
    return yaml.safe_load(Path(p).read_text())


def load_go2_session(session_dir: str | Path, mapping: dict | None = None, contact_force_thresh: float = 20.0):
    session_dir = Path(session_dir); mp = mapping or load_mapping(); files = mp["session_files"]
    meta = yaml.safe_load((session_dir / files["meta"]).read_text()) if (session_dir / files["meta"]).exists() else {}
    ls = pd.read_csv(session_dir / files["lowstate"])
    cmd_t = pd.read_csv(session_dir / files["lowcmd"]) if (session_dir / files["lowcmd"]).exists() else None
    t = ls["t"].to_numpy(); n = len(ls)
    df = pd.DataFrame(np.nan, index=np.arange(n), columns=all_columns()); df["t"] = t; df["theta"] = np.nan
    idx = mp["geofdi"]["motor_index_in_geofdi_order"]; fidx = mp["geofdi"]["foot_index_in_geofdi_order"]
    missing, found = [], []
    for gi, leg in enumerate(LEGS):
        for gj, j in enumerate(JOINTS):
            k = idx[3 * gi + gj]; sgn = float(mp["signs"]["per_leg_sign"][leg][gj])
            for src, grp in (("q", "q"), ("dq", "dq"), ("tau_est", "tau_meas")):
                col = f"{src}_{k}"
                if col in ls.columns:
                    df[f"{grp}_{leg}_{j}"] = sgn * ls[col].to_numpy(); found.append(col)
                else:
                    missing.append(col)
            if cmd_t is not None and f"tau_{k}" in cmd_t.columns:
                df[f"tau_cmd_{leg}_{j}"] = sgn * np.interp(t, cmd_t["t"].to_numpy(), cmd_t[f"tau_{k}"].to_numpy()); found.append(f"lowcmd tau_{k}")
        col = f"temp_{idx[3 * gi]}"
        if col in ls.columns:
            df[f"temp_{leg}"] = ls[[f"temp_{idx[3 * gi + gj]}" for gj in range(3) if f"temp_{idx[3 * gi + gj]}" in ls.columns]].mean(axis=1).to_numpy()
        fcol = f"foot_force_{fidx[gi]}"
        if fcol in ls.columns:
            df[f"c_{leg}"] = (ls[fcol].to_numpy() > contact_force_thresh).astype(float); found.append(fcol)
        else:
            missing.append(fcol)
    for ax, src in zip("xyz", ("acc_x", "acc_y", "acc_z")):
        df[f"imu_a_{ax}"] = ls[src].to_numpy() if src in ls.columns else np.nan
    for ax, src in zip("xyz", ("gyro_x", "gyro_y", "gyro_z")):
        df[f"imu_w_{ax}"] = ls[src].to_numpy() if src in ls.columns else np.nan
    for q, src in zip(("base_qw", "base_qx", "base_qy", "base_qz"), ("imu_qw", "imu_qx", "imu_qy", "imu_qz")):
        df[q] = ls[src].to_numpy() if src in ls.columns else np.nan
    if cmd_t is None:
        missing.append("lowcmd.csv (tau_cmd)")
    manifest = build_manifest(sim_meta={"source": meta.get("source", "unknown"), "robot": "go2", "rate_hz": meta.get("rate_hz"),
                                        "tau_meas_semantics": mp["torque"]["tau_est_semantics"], "mapping_unverified": bool(mp.get("unverified", True)),
                                        "session": str(session_dir), "notes": meta.get("notes", ""), "phase": "kinematic estimator (phase/estimator.py)"})
    report = {"n_rows": int(n), "duration_s": float(t[-1] - t[0]) if n else 0.0, "found": found, "missing": missing,
              "mapping_unverified": bool(mp.get("unverified", True)), "rate_hz_estimate": float(1.0 / np.median(np.diff(t))) if n > 2 else float("nan")}
    return df, manifest, report


def write_go2_session(session_dir: str | Path, df: pd.DataFrame, meta: dict, mapping: dict | None = None, foot_force_from_contact: float = 60.0) -> None:
    """Export a GeoFDI Go2 telemetry frame (sim rollout) in the LowState CSV layout: Unitree motor order, tau_est =
    tau_meas, foot_force = contact flag x foot_force_from_contact (or the recorded contact normal force if fc_z_* exists)."""
    session_dir = Path(session_dir); session_dir.mkdir(parents=True, exist_ok=True); mp = mapping or load_mapping()
    idx = mp["geofdi"]["motor_index_in_geofdi_order"]; fidx = mp["geofdi"]["foot_index_in_geofdi_order"]
    ls = pd.DataFrame({"t": df["t"].to_numpy()}); cmd = pd.DataFrame({"t": df["t"].to_numpy()})
    inv = {idx[i]: i for i in range(12)}                                   # motor k -> geofdi joint index
    for k in range(12):
        gi, gj = divmod(inv[k], 3); leg, j = LEGS[gi], JOINTS[gj]; sgn = float(mp["signs"]["per_leg_sign"][leg][gj])
        ls[f"q_{k}"] = sgn * df[f"q_{leg}_{j}"].to_numpy(); ls[f"dq_{k}"] = sgn * df[f"dq_{leg}_{j}"].to_numpy()
        ls[f"tau_est_{k}"] = sgn * df[f"tau_meas_{leg}_{j}"].to_numpy(); ls[f"temp_{k}"] = df[f"temp_{leg}"].to_numpy() if f"temp_{leg}" in df else 0.0
        cmd[f"tau_{k}"] = sgn * df[f"tau_cmd_{leg}_{j}"].to_numpy()
    for q, src in zip(("imu_qw", "imu_qx", "imu_qy", "imu_qz"), ("base_qw", "base_qx", "base_qy", "base_qz")):
        ls[q] = df[src].to_numpy()
    for ax in "xyz":
        ls[f"gyro_{ax}"] = df[f"imu_w_{ax}"].to_numpy(); ls[f"acc_{ax}"] = df[f"imu_a_{ax}"].to_numpy()
    for gi, leg in enumerate(LEGS):
        k = fidx[gi]
        ls[f"foot_force_{k}"] = df[f"fc_z_{leg}"].to_numpy() if f"fc_z_{leg}" in df else foot_force_from_contact * df[f"c_{leg}"].to_numpy()
    ls.to_csv(session_dir / mp["session_files"]["lowstate"], index=False, float_format="%.7g")
    cmd.to_csv(session_dir / mp["session_files"]["lowcmd"], index=False, float_format="%.7g")
    (session_dir / mp["session_files"]["meta"]).write_text(yaml.safe_dump(meta, sort_keys=False))
