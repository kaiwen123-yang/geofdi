"""Wheeled-M1 session loader (Sprint 7 Block W1; hardware conventions verified in Sprint 8 Block D).

A session directory holds `joint_states.csv` (t + <name>_pos/_vel/_eff per joint name, any column order), `imu.csv`
(t, qw..qz, wx..wz, ax..az), optional `odom.csv` / `cmd.csv` and `meta.yaml`; a raw rosbag2 directory (metadata.yaml +
*.db3, as recorded by the vendor stack) is accepted too — it is extracted once with `geofdi.io.m1_rosbag` into
`$GEOFDI_DATA_ROOT/data/processed/m1/<session>/` (the raw directory is never written) and the CSVs are read from there.

Joints are reordered by NAME through `m1_mapping.yaml` (verified names `fl1_hip_roll` .. `br4_foot`; the bare `fl1` ..
form is accepted as a fallback); channels missing in the file are NaN and listed in the report. Conventions are switched by
`meta.yaml` (`sign_convention`, `imu_convention`; default: `source: sim*` -> GeoFDI uniform-axis / body-frame m/s^2, else
vendor): hardware sessions get the per-leg mirror signs of the mapping, the IMU rotation `R_body_from_sensor` and the g ->
m/s^2 conversion; sim rehearsals (written by write_m1_session) are read back unchanged. `efforts_semantics` (unknown |
current_estimate | torque) is taken from meta.yaml (or the mapping default) and written into the manifest.

    df, manifest, report = load_m1_session(session_dir)
    write_m1_session(session_dir, df_geofdi, meta)          # inverse: export a GeoFDI frame in the SDK layout (rehearsals)
"""
from __future__ import annotations

import os
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..sim.telemetry_m1 import JOINTS, LEGS, all_columns, build_manifest


def load_mapping(path: str | Path | None = None) -> dict:
    p = Path(path) if path else resources.files("geofdi.io").joinpath("m1_mapping.yaml")
    return yaml.safe_load(Path(p).read_text())


def sdk_name(mapping: dict, leg: str, joint: str, bare: bool = False) -> str:
    """Joint name in the recording: '<leg><index><suffix>' (e.g. fl1_hip_roll); bare=True gives '<leg><index>' (fl1)."""
    inv_leg = {v: k for k, v in mapping["sdk"]["legs"].items()}; inv_idx = {v: k for k, v in mapping["sdk"]["index"].items()}
    idx = inv_idx[joint]; suffix = "" if bare else mapping["sdk"].get("suffix", {}).get(idx, "")
    return mapping["sdk"]["name_pattern"].format(leg=inv_leg[leg], index=idx, suffix=suffix)


def _conventions(meta: dict, mp: dict) -> tuple[str, str]:
    src = str(meta.get("source", "unknown"))
    default = "geofdi_uniform_axis" if src.startswith("sim") else "vendor"
    sign_conv = meta.get("sign_convention", default)
    imu_conv = meta.get("imu_convention", "geofdi_body_mps2" if src.startswith("sim") else "vendor")
    return sign_conv, imu_conv


def _resolve_session_dir(session_dir: Path, mp: dict) -> tuple[Path, dict | None]:
    """A raw rosbag2 directory is extracted (once) into data/processed/m1/<name>/; returns (csv_dir, extract_report)."""
    files = mp["session_files"]
    if (session_dir / files["joint_states"]).exists():
        return session_dir, None
    from .m1_rosbag import extract_bag_session, is_rosbag2
    if not is_rosbag2(session_dir):
        raise FileNotFoundError(f"{session_dir}: neither {files['joint_states']} nor a rosbag2 (metadata.yaml + *.db3) found")
    root = os.environ.get("GEOFDI_DATA_ROOT")
    out = (Path(root) / "data" / "processed" / "m1" / session_dir.name) if root else (Path(tempfile.gettempdir()) / "geofdi_m1_extract" / session_dir.name)
    rep = extract_bag_session(session_dir, out)
    return out, rep


def load_m1_session(session_dir: str | Path, mapping: dict | None = None):
    session_dir = Path(session_dir); mp = mapping or load_mapping()
    files = mp["session_files"]
    csv_dir, extract_rep = _resolve_session_dir(session_dir, mp)
    meta = (yaml.safe_load((csv_dir / files["meta"]).read_text()) or {}) if (csv_dir / files["meta"]).exists() else {}
    if csv_dir != session_dir and (session_dir / files["meta"]).exists():      # the hand-filled raw meta.yaml overrides the extractor's auto-meta
        meta.update(yaml.safe_load((session_dir / files["meta"]).read_text()) or {})
    sign_conv, imu_conv = _conventions(meta, mp)
    js = pd.read_csv(csv_dir / files["joint_states"])
    imu = pd.read_csv(csv_dir / files["imu"]) if (csv_dir / files["imu"]).exists() else None
    cmd = pd.read_csv(csv_dir / files["cmd"]) if (csv_dir / files["cmd"]).exists() else None
    odom = pd.read_csv(csv_dir / files.get("odom", "odom.csv")) if (csv_dir / files.get("odom", "odom.csv")).exists() else None
    t = js["t"].to_numpy(); n = len(js)
    df = pd.DataFrame(np.nan, index=np.arange(n), columns=all_columns()); df["t"] = t; df["blk"] = -1.0; df["theta"] = np.nan
    missing = []; found = []; name_used = {}
    for leg in LEGS:
        for j in JOINTS:
            sgn = float(mp["signs"]["per_leg_sign"][leg][JOINTS.index(j)]) if sign_conv == "vendor" else 1.0
            for suf, grp in (("_pos", "q"), ("_vel", "dq"), ("_eff", "tau_meas"), ("_cmd", "tau_cmd")):
                col = next((c for c in (f"{sdk_name(mp, leg, j)}{suf}", f"{sdk_name(mp, leg, j, bare=True)}{suf}") if c in js.columns), None)
                if col is not None:
                    df[f"{grp}_{leg}_{j}"] = sgn * js[col].to_numpy(); found.append(col); name_used[f"{leg}_{j}"] = col[: -len(suf)]
                elif suf != "_cmd":                # commanded torque is not part of the state message: NaN unless provided
                    missing.append(f"{sdk_name(mp, leg, j)}{suf}")
    imu_info = {"convention": imu_conv}
    if imu is not None:
        ti = imu["t"].to_numpy()
        def interp(c):
            return np.interp(t, ti, imu[c].to_numpy()) if c in imu else np.full(n, np.nan)
        A = np.stack([interp(c) for c in ("ax", "ay", "az")], 1); W = np.stack([interp(c) for c in ("wx", "wy", "wz")], 1)
        if imu_conv == "vendor":
            R = np.asarray(mp["imu"].get("R_body_from_sensor", np.eye(3)), dtype=float)
            g0 = 9.80665 if str(meta.get("imu_accel_units", mp["imu"].get("accel_units", "m/s^2"))).lower() == "g" else 1.0
            gs = np.pi / 180.0 if str(meta.get("imu_gyro_units", mp["imu"].get("gyro_units", "rad/s"))).lower() in ("deg/s", "dps") else 1.0
            A = (A * g0) @ R.T; W = (W * gs) @ R.T
            imu_info.update(R_body_from_sensor=R.tolist(), accel_scale=g0, gyro_scale=gs, frame=mp["imu"].get("frame"))
        for k, ax in enumerate("xyz"):
            df[f"imu_a_{ax}"] = A[:, k]; df[f"imu_w_{ax}"] = W[:, k]
        for q, c in zip(("base_qw", "base_qx", "base_qy", "base_qz"), ("qw", "qx", "qy", "qz")):
            df[q] = interp(c)
    else:
        missing.append("imu.csv")
    if odom is not None:                            # vendor odometry -> diagnostics (never in Z); overrides the (constant) IMU quaternion
        to = odom["t"].to_numpy()
        for dst, c in (("base_x", "x"), ("base_y", "y"), ("base_z", "z"), ("base_qw", "qw"), ("base_qx", "qx"), ("base_qy", "qy"), ("base_qz", "qz"),
                       ("base_vx", "vx"), ("base_vy", "vy"), ("base_vz", "vz")):
            if c in odom:
                df[dst] = np.interp(t, to, odom[c].to_numpy())
    if cmd is not None:
        df["v_cmd"] = np.interp(t, cmd["t"].to_numpy(), cmd["vx"].to_numpy())
    else:
        df["v_cmd"] = np.nan; missing.append("cmd.csv (v_cmd)")
    for leg in LEGS:                                   # no contact sensor on the wheeled M1: flag unknown -> NaN
        df[f"c_{leg}"] = np.nan; df[f"temp_{leg}"] = np.nan
    eff = meta.get("efforts_semantics", mp["sdk"].get("efforts_semantics", "unknown"))
    manifest = build_manifest(sim_meta={"source": meta.get("source", "unknown"), "robot": "m1_wheeled", "rate_hz": meta.get("rate_hz"),
                                        "efforts_semantics": eff, "mapping_unverified": bool(mp.get("unverified", True)),
                                        "sign_convention": sign_conv, "imu": imu_info, "session": str(session_dir), "csv_dir": str(csv_dir),
                                        "notes": meta.get("notes", "")})
    report = {"n_rows": int(n), "duration_s": float(t[-1] - t[0]) if n else 0.0, "found": found, "missing": missing,
              "efforts_semantics": eff, "mapping_unverified": bool(mp.get("unverified", True)), "sign_convention": sign_conv,
              "imu": imu_info, "odom_present": odom is not None, "csv_dir": str(csv_dir), "extracted_from_bag": extract_rep is not None,
              "joint_names_used": name_used, "rate_hz_estimate": float(1.0 / np.median(np.diff(t))) if n > 2 else float("nan")}
    if extract_rep is not None:
        report["extract"] = {k: extract_rep.get(k) for k in ("db3_files_read", "db3_files_skipped", "db3_files_truncated", "t0_epoch_ns", "time_source")}
    return df, manifest, report


def write_m1_session(session_dir: str | Path, df: pd.DataFrame, meta: dict, mapping: dict | None = None,
                     with_cmd_torque: bool = True) -> None:
    """Export a GeoFDI M1 telemetry frame in the SDK session layout (rehearsal sessions from the simulator)."""
    session_dir = Path(session_dir); session_dir.mkdir(parents=True, exist_ok=True); mp = mapping or load_mapping()
    meta = dict(meta); meta.setdefault("sign_convention", "geofdi_uniform_axis"); meta.setdefault("imu_convention", "geofdi_body_mps2")
    js = pd.DataFrame({"t": df["t"].to_numpy()})
    for leg in LEGS:
        for j in JOINTS:
            name = sdk_name(mp, leg, j, bare=True); sgn = 1.0          # GeoFDI convention, bare SDK-style names
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
