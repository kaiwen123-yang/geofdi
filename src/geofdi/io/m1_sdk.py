"""Wheeled-M1 session loader (Sprint 7 Block W1): GENISOM SDK-style recordings -> GeoFDI telemetry frame + manifest.

A session directory holds `joint_states.csv` (t + <name>_pos/_vel/_eff per SDK joint name, any column order),
`imu.csv` (t, qw..qz, wx..wz, ax..az), optional `cmd.csv` (t, vx, vy, wz) and `meta.yaml`. Joints are reordered by
NAME through `m1_mapping.yaml` (candidate mapping, `unverified: true`); channels that are missing in the file are filled
with NaN and listed in the report; `efforts_semantics` (unknown | current_estimate | torque) is taken from meta.yaml (or
the mapping default) and written into the manifest so that downstream code knows whether tau_meas is a torque.

    df, manifest, report = load_m1_session(session_dir)
    write_m1_session(session_dir, df_geofdi, meta)          # inverse: export a GeoFDI frame in the SDK layout (rehearsals)
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..sim.telemetry_m1 import JOINTS, LEGS, all_columns, build_manifest


def load_mapping(path: str | Path | None = None) -> dict:
    p = Path(path) if path else resources.files("geofdi.io").joinpath("m1_mapping.yaml")
    return yaml.safe_load(Path(p).read_text())


def sdk_name(mapping: dict, leg: str, joint: str) -> str:
    inv_leg = {v: k for k, v in mapping["sdk"]["legs"].items()}; inv_idx = {v: k for k, v in mapping["sdk"]["index"].items()}
    return mapping["sdk"]["name_pattern"].format(leg=inv_leg[leg], index=inv_idx[joint])


def load_m1_session(session_dir: str | Path, mapping: dict | None = None):
    session_dir = Path(session_dir); mp = mapping or load_mapping()
    files = mp["session_files"]
    meta = yaml.safe_load((session_dir / files["meta"]).read_text()) if (session_dir / files["meta"]).exists() else {}
    js = pd.read_csv(session_dir / files["joint_states"])
    imu = pd.read_csv(session_dir / files["imu"]) if (session_dir / files["imu"]).exists() else None
    cmd = pd.read_csv(session_dir / files["cmd"]) if (session_dir / files["cmd"]).exists() else None
    t = js["t"].to_numpy(); n = len(js)
    df = pd.DataFrame(np.nan, index=np.arange(n), columns=all_columns()); df["t"] = t; df["blk"] = -1.0; df["theta"] = np.nan
    missing = []; found = []
    for leg in LEGS:
        for j in JOINTS:
            name = sdk_name(mp, leg, j); sgn = float(mp["signs"]["per_leg_sign"][leg][JOINTS.index(j)])
            for suf, grp in (("_pos", "q"), ("_vel", "dq"), ("_eff", "tau_meas")):
                col = f"{name}{suf}"
                if col in js.columns:
                    df[f"{grp}_{leg}_{j}"] = sgn * js[col].to_numpy(); found.append(col)
                else:
                    missing.append(col)
    # commanded torque is not part of the SDK state message: tau_cmd stays NaN unless the session provides <name>_cmd
    for leg in LEGS:
        for j in JOINTS:
            col = f"{sdk_name(mp, leg, j)}_cmd"
            if col in js.columns:
                df[f"tau_cmd_{leg}_{j}"] = float(mp["signs"]["per_leg_sign"][leg][JOINTS.index(j)]) * js[col].to_numpy(); found.append(col)
    if imu is not None:
        ti = imu["t"].to_numpy()
        def interp(c):
            return np.interp(t, ti, imu[c].to_numpy()) if c in imu else np.full(n, np.nan)
        for ax, c in zip("xyz", ("ax", "ay", "az")):
            df[f"imu_a_{ax}"] = interp(c)
        for ax, c in zip("xyz", ("wx", "wy", "wz")):
            df[f"imu_w_{ax}"] = interp(c)
        for q, c in zip(("base_qw", "base_qx", "base_qy", "base_qz"), ("qw", "qx", "qy", "qz")):
            df[q] = interp(c)
    else:
        missing.append("imu.csv")
    if cmd is not None:
        df["v_cmd"] = np.interp(t, cmd["t"].to_numpy(), cmd["vx"].to_numpy())
    else:
        df["v_cmd"] = np.nan; missing.append("cmd.csv (v_cmd)")
    for leg in LEGS:                                   # no contact sensor on the wheeled M1: flag unknown -> NaN
        df[f"c_{leg}"] = np.nan; df[f"temp_{leg}"] = np.nan
    eff = meta.get("efforts_semantics", mp["sdk"].get("efforts_semantics", "unknown"))
    manifest = build_manifest(sim_meta={"source": meta.get("source", "unknown"), "robot": "m1_wheeled", "rate_hz": meta.get("rate_hz"),
                                        "efforts_semantics": eff, "mapping_unverified": bool(mp.get("unverified", True)),
                                        "session": str(session_dir), "notes": meta.get("notes", "")})
    report = {"n_rows": int(n), "duration_s": float(t[-1] - t[0]) if n else 0.0, "found": found, "missing": missing,
              "efforts_semantics": eff, "mapping_unverified": bool(mp.get("unverified", True)),
              "rate_hz_estimate": float(1.0 / np.median(np.diff(t))) if n > 2 else float("nan")}
    return df, manifest, report


def write_m1_session(session_dir: str | Path, df: pd.DataFrame, meta: dict, mapping: dict | None = None,
                     with_cmd_torque: bool = True) -> None:
    """Export a GeoFDI M1 telemetry frame in the SDK session layout (rehearsal sessions from the simulator)."""
    session_dir = Path(session_dir); session_dir.mkdir(parents=True, exist_ok=True); mp = mapping or load_mapping()
    js = pd.DataFrame({"t": df["t"].to_numpy()})
    for leg in LEGS:
        for j in JOINTS:
            name = sdk_name(mp, leg, j); sgn = float(mp["signs"]["per_leg_sign"][leg][JOINTS.index(j)])
            js[f"{name}_pos"] = sgn * df[f"q_{leg}_{j}"].to_numpy(); js[f"{name}_vel"] = sgn * df[f"dq_{leg}_{j}"].to_numpy()
            js[f"{name}_eff"] = sgn * df[f"tau_meas_{leg}_{j}"].to_numpy()
            if with_cmd_torque:
                js[f"{name}_cmd"] = sgn * df[f"tau_cmd_{leg}_{j}"].to_numpy()
    # shuffle the column order (deterministically) to exercise the name-based reorder
    cols = list(js.columns[1:]); rng = np.random.default_rng(0); rng.shuffle(cols); js = js[["t"] + cols]
    js.to_csv(session_dir / mp["session_files"]["joint_states"], index=False, float_format="%.7g")
    imu = pd.DataFrame({"t": df["t"], "qw": df["base_qw"], "qx": df["base_qx"], "qy": df["base_qy"], "qz": df["base_qz"],
                        "wx": df["imu_w_x"], "wy": df["imu_w_y"], "wz": df["imu_w_z"], "ax": df["imu_a_x"], "ay": df["imu_a_y"], "az": df["imu_a_z"]})
    imu.to_csv(session_dir / mp["session_files"]["imu"], index=False, float_format="%.7g")
    if "v_cmd" in df:
        pd.DataFrame({"t": df["t"], "vx": df["v_cmd"], "vy": 0.0, "wz": 0.0}).to_csv(session_dir / mp["session_files"]["cmd"], index=False, float_format="%.7g")
    (session_dir / mp["session_files"]["meta"]).write_text(yaml.safe_dump(meta, sort_keys=False))
