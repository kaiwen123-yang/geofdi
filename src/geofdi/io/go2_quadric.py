"""QUADRIC-GINS Go2 corpus loader (Sprint 9 Block Q3): three streams -> one GeoFDI telemetry frame + manifest.

A session directory (raw/go2/<day>/<name>/) holds
    highlevel/<name>.txt      `ros2 topic echo /sportmodestate` terminal transcript (io/go2_highlevel_txt.py), ~240 Hz
    foot_imu/<name>.csv       foot-mounted IMU board, 2 redundant IMUs, 200 Hz, LEFT-HIND leg only (Jan sessions only)
    fixposition/*.csv         Fixposition Vision-RTK2 products (geodetic pose/vel/ypr + variances, odometry, fusion status)

**No joint stream.** The high-level SportModeState API carries no motorState, so the C2 mirror element is built from the
foot/IMU channels instead of q/dq/tau. This is NOT the degraded "foot force + IMU" fallback: `foot_position_body` and
`foot_speed_body` are the robot's own forward kinematics of the leg joints expressed in the body frame, so they carry the
same per-leg kinematic asymmetry the joint channels would — 28 of the 34 in-Z channels are leg-resolved.

Mirror representation (sagittal reflection E = diag(1,-1,1), leg swap LF<->RF, LH<->RH):
    foot_pos_{leg}_{x,y,z}, foot_vel_{leg}_{x,y,z}   polar vectors  -> sign (+1,-1,+1), partner = mirror leg
    foot_force_{leg}                                  scalar magnitude -> sign +1, partner = mirror leg
    imu_a_{x,y,z} polar (+1,-1,+1) / imu_w_{x,y,z} axial (-1,+1,-1), self-partnered
Estimator outputs (body position/velocity/yaw_speed/rpy/quaternion), mode/gait_type and the Fixposition reference are
diagnostics (in_Z = False), following the repo convention that only measured channels enter the element.

    df, manifest, report = load_go2_quadric_session(session_dir)
    mask, info = straight_mask_go2(df)          # yaw-rate + RTK-heading-rate dual criterion
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEGS = ("LF", "RF", "LH", "RH")
MIRROR_LEG = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
UNITREE_LEG = {"LF": 1, "RF": 0, "LH": 3, "RH": 2}      # Unitree order is FR, FL, RR, RL (verified from the y-signs)
VEC_SIGN = {"x": +1, "y": -1, "z": +1}                   # polar vector under E
IMU_ACC_SIGN = {"x": +1, "y": -1, "z": +1}
IMU_GYRO_SIGN = {"x": -1, "y": +1, "z": -1}              # axial (pseudovector) under E


# ------------------------------------------------------------------------------------------ manifest
def build_go2_quadric_manifest(sim_meta: dict | None = None) -> dict:
    ch = []
    for grp, stem in (("foot_pos", "foot_pos"), ("foot_vel", "foot_vel")):
        for leg in LEGS:
            for ax in "xyz":
                ch.append({"name": f"{stem}_{leg}_{ax}", "group": grp, "leg": leg, "joint": ax, "kind": "polar",
                           "frame": "body", "partner": f"{stem}_{MIRROR_LEG[leg]}_{ax}", "sign": VEC_SIGN[ax], "in_Z": True})
    for leg in LEGS:
        ch.append({"name": f"foot_force_{leg}", "group": "contact", "leg": leg, "joint": None, "kind": "scalar-magnitude",
                   "partner": f"foot_force_{MIRROR_LEG[leg]}", "sign": +1, "in_Z": True})
    for ax in "xyz":
        ch.append({"name": f"imu_a_{ax}", "group": "imu_acc", "leg": None, "joint": None, "kind": "polar", "frame": "body",
                   "partner": f"imu_a_{ax}", "sign": IMU_ACC_SIGN[ax], "in_Z": True})
    for ax in "xyz":
        ch.append({"name": f"imu_w_{ax}", "group": "imu_gyro", "leg": None, "joint": None, "kind": "axial", "frame": "body",
                   "partner": f"imu_w_{ax}", "sign": IMU_GYRO_SIGN[ax], "in_Z": True})
    diag = ["t", "theta", "base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz",
            "base_vx", "base_vy", "base_vz", "yaw_speed", "body_height", "gait_type", "mode", "imu_temp",
            "roll", "pitch", "yaw", "foot_raise_height", "rtk_lat", "rtk_lon", "rtk_alt", "rtk_yaw",
            "rtk_ve", "rtk_vn", "rtk_vu", "rtk_pvar_x", "rtk_pvar_y", "rtk_fix_ok", "t_abs"]
    for name in diag:
        ch.append({"name": name, "group": "diagnostic", "leg": None, "joint": None, "kind": "diagnostic",
                   "partner": None, "sign": None, "in_Z": False})
    return {"schema": "geofdi-go2-quadric-highlevel-v1", "robot": "go2_quadric", "leg_order": list(LEGS),
            "gait_group": {"G": "C2 sagittal reflection", "Sigma_trot": "{(e,0),(g_s,1/2)}", "delta_theta": 0.5},
            "channels": ch, "sim": sim_meta or {}}


# ------------------------------------------------------------------------------------------ streams
def load_foot_imu(path: str | Path) -> pd.DataFrame:
    """Foot-mounted IMU board CSV: two redundant IMUs (imu_0, imu_1) + mag/press/temp, 200 Hz, LOCAL-time stamps."""
    d = pd.read_csv(path, low_memory=False)
    cols = list(d.columns)
    ts = d[cols[-1]].astype(str).str.strip().str.strip('"').str.strip()
    t = pd.to_datetime(ts, format="mixed", errors="coerce")
    out = pd.DataFrame({"t_abs": t.dt.tz_localize("Asia/Shanghai").view("int64") / 1e9})   # local -> epoch seconds
    ren = {"(imu_0)gyro_x": "g0x", "gyro_y": "g0y", "gyro_z": "g0z", "(imu_0)acc_x": "a0x", "acc_y": "a0y", "acc_z": "a0z",
           "(imu_1)gyro_x": "g1x", "(imu_1)acc_x": "a1x"}
    # positional read (the header repeats bare names for the second IMU)
    arr = d.iloc[:, :12].to_numpy(dtype=float)
    for j, name in enumerate(["g0x", "g0y", "g0z", "a0x", "a0y", "a0z", "g1x", "g1y", "g1z", "a1x", "a1y", "a1z"]):
        out[name] = arr[:, j]
    for extra in ("press", "temp", "sn"):
        if extra in d.columns:
            out[extra] = pd.to_numeric(d[extra], errors="coerce")
    return out.dropna(subset=["t_abs"]).reset_index(drop=True)


def load_fixposition(fx_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Fixposition Vision-RTK2 geodetic product + fusion status -> a reference frame on the SAME Unix clock as the
    SportModeState stream (both stamped by the recording host), so no motion-correlation alignment is needed."""
    fx_dir = Path(fx_dir)
    g = pd.read_csv(fx_dir / "user_io-out-poi_geodetic.csv", low_memory=False)
    t = g["header.stamp.secs"].to_numpy(float) + g["header.stamp.nsecs"].to_numpy(float) * 1e-9
    ref = pd.DataFrame({"t_abs": t, "lat": g["p.vector3.x"], "lon": g["p.vector3.y"], "alt": g["p.vector3.z"],
                        "pvar_x": g["p_var.vector3.x"], "pvar_y": g["p_var.vector3.y"], "pvar_z": g["p_var.vector3.z"],
                        "yaw": g["ypr.vector3.x"], "pitch": g["ypr.vector3.y"], "roll": g["ypr.vector3.z"],
                        "ve": g["v.vector3.x"], "vn": g["v.vector3.y"], "vu": g["v.vector3.z"]})
    ref = ref.dropna(subset=["t_abs"]).drop_duplicates(subset=["t_abs"]).sort_values("t_abs").reset_index(drop=True)
    # local ENU metres about the session median (a local tangent plane is enough for ATE over a few hundred metres)
    lat0, lon0 = ref["lat"].median(), ref["lon"].median()
    m_per_deg_lat = 111320.0; m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    ref["enu_e"] = (ref["lon"] - lon0) * m_per_deg_lon; ref["enu_n"] = (ref["lat"] - lat0) * m_per_deg_lat
    ref["enu_u"] = ref["alt"] - ref["alt"].median()
    info = {"n": int(len(ref)), "rate_hz": float(1.0 / np.median(np.diff(ref["t_abs"]))) if len(ref) > 2 else float("nan"),
            "lat0": float(lat0), "lon0": float(lon0), "pvar_x_q50": float(ref["pvar_x"].median()),
            "pvar_x_q90": float(ref["pvar_x"].quantile(0.9)), "duration_s": float(ref["t_abs"].iloc[-1] - ref["t_abs"].iloc[0]) if len(ref) else 0.0}
    st_path = fx_dir / "user_io-out-odom_status.csv"
    if st_path.exists():
        st = pd.read_csv(st_path, low_memory=False)
        tst = st["header.stamp.secs"].to_numpy(float) + st["header.stamp.nsecs"].to_numpy(float) * 1e-9
        # Fixposition GNSS status: 5 = RTK fixed, 8 = RTK float / other fixed-type, 0/1 = none/single -> "fix ok" = {5, 8}
        ok = np.isin(st["gnss1_status"].to_numpy(), [5, 8]).astype(float)
        ref["fix_ok"] = np.interp(ref["t_abs"], tst, ok) > 0.5
        info["fix_ok_fraction"] = float(np.mean(ref["fix_ok"]))
        info["gnss1_status_counts"] = {int(k): int(v) for k, v in st["gnss1_status"].value_counts().items()}
    else:
        ref["fix_ok"] = True; info["fix_ok_fraction"] = float("nan")
    return ref, info


# ------------------------------------------------------------------------------------------ unified session
def load_go2_quadric_session(session_dir: str | Path, cache: bool = True, max_records: int | None = None):
    """Load one ingested session into the GeoFDI schema. Caches the parsed transcript as parquet under
    $GEOFDI_DATA_ROOT/data/processed/go2/<name>.parquet (parsing 90k records from a transcript takes ~40 s)."""
    import os
    from .go2_highlevel_txt import parse_sportmodestate
    session_dir = Path(session_dir); name = session_dir.name
    txt = next((session_dir / "highlevel").glob("*.txt"))
    root = os.environ.get("GEOFDI_DATA_ROOT")
    cache_p = Path(root) / "data" / "processed" / "go2" / f"{name}.parquet" if (root and cache and max_records is None) else None
    if cache_p is not None and cache_p.exists():
        hl = pd.read_parquet(cache_p); hl_rep = {"cached": True, "n_records": len(hl)}
    else:
        hl, hl_rep = parse_sportmodestate(txt, max_records=max_records)
        if cache_p is not None:
            cache_p.parent.mkdir(parents=True, exist_ok=True); hl.to_parquet(cache_p, index=False)
    n = len(hl)
    df = pd.DataFrame({"t": hl["t"].to_numpy(), "t_abs": hl["t_abs"].to_numpy()})
    for leg in LEGS:
        li = UNITREE_LEG[leg]
        for k, ax in enumerate("xyz"):
            df[f"foot_pos_{leg}_{ax}"] = hl[f"foot_pos{3 * li + k}"].to_numpy()
            df[f"foot_vel_{leg}_{ax}"] = hl[f"foot_vel{3 * li + k}"].to_numpy()
        df[f"foot_force_{leg}"] = hl[f"foot_force{li}"].to_numpy()
        df[f"c_{leg}"] = np.nan                                     # filled below from the force threshold
    for k, ax in enumerate("xyz"):
        df[f"imu_a_{ax}"] = hl[f"acc{k}"].to_numpy(); df[f"imu_w_{ax}"] = hl[f"gyro{k}"].to_numpy()
    for k, nm in enumerate(("base_qw", "base_qx", "base_qy", "base_qz")):
        df[nm] = hl[f"quat{k}"].to_numpy()
    for k, nm in enumerate(("roll", "pitch", "yaw")):
        df[nm] = hl[f"rpy{k}"].to_numpy()
    for k, nm in enumerate(("base_x", "base_y", "base_z")):
        df[nm] = hl[f"pos{k}"].to_numpy()
    for k, nm in enumerate(("base_vx", "base_vy", "base_vz")):
        df[nm] = hl[f"vel{k}"].to_numpy()
    for nm, src in (("yaw_speed", "yaw_speed"), ("body_height", "body_height"), ("gait_type", "gait_type"),
                    ("mode", "mode"), ("imu_temp", "imu_temp"), ("foot_raise_height", "foot_raise_height")):
        df[nm] = hl[src].to_numpy()
    df["theta"] = np.nan
    # contact flags from the foot force (per-leg Otsu-free rule: above 40 % of the per-leg 90th percentile)
    for leg in LEGS:
        f = df[f"foot_force_{leg}"].to_numpy()
        df[f"c_{leg}"] = (f > 0.4 * np.nanquantile(f, 0.9)).astype(float)
    # ---- Fixposition reference resampled onto the telemetry clock
    fx_dir = session_dir / "fixposition"
    fx_info = {}
    if fx_dir.exists() and (fx_dir / "user_io-out-poi_geodetic.csv").exists():
        ref, fx_info = load_fixposition(fx_dir)
        ta = df["t_abs"].to_numpy()
        for dst, src in (("rtk_lat", "lat"), ("rtk_lon", "lon"), ("rtk_alt", "alt"), ("rtk_yaw", "yaw"),
                         ("rtk_ve", "ve"), ("rtk_vn", "vn"), ("rtk_vu", "vu"), ("rtk_pvar_x", "pvar_x"), ("rtk_pvar_y", "pvar_y"),
                         ("rtk_e", "enu_e"), ("rtk_n", "enu_n"), ("rtk_u", "enu_u")):
            df[dst] = np.interp(ta, ref["t_abs"], ref[src], left=np.nan, right=np.nan)
        df["rtk_fix_ok"] = np.interp(ta, ref["t_abs"], ref["fix_ok"].astype(float), left=0, right=0) > 0.5
        fx_info["overlap_s"] = float(np.sum(np.isfinite(df["rtk_lat"])) * np.median(np.diff(ta)))
    else:
        for c in ("rtk_lat", "rtk_lon", "rtk_alt", "rtk_yaw", "rtk_ve", "rtk_vn", "rtk_vu", "rtk_pvar_x", "rtk_pvar_y", "rtk_e", "rtk_n", "rtk_u"):
            df[c] = np.nan
        df["rtk_fix_ok"] = False
    manifest = build_go2_quadric_manifest(sim_meta={
        "source": "hardware (Unitree Go2, QUADRIC-GINS corpus)", "robot": "go2_quadric", "session": name,
        "rate_hz": hl_rep.get("rate_hz_median"), "joint_stream": False,
        "element_channels": "foot_pos(12) + foot_vel(12) + foot_force(4) + imu(6) = 34", "mapping_unverified": False})
    foot = sorted((session_dir / "foot_imu").glob("*.csv")) if (session_dir / "foot_imu").exists() else []
    report = {"session": name, "n_rows": int(n), "duration_s": float(df["t"].iloc[-1]) if n else 0.0,
              "rate_hz": hl_rep.get("rate_hz_median"), "highlevel": hl_rep, "fixposition": fx_info,
              "foot_imu_present": bool(foot), "foot_imu_path": str(foot[0]) if foot else None,
              "contact_duty": {l: float(df[f"c_{l}"].mean()) for l in LEGS}, "joint_stream": False}
    return df, manifest, report


def straight_mask_go2(df, wz_max: float = 0.15, rtk_yawrate_max_deg: float = 8.0, smooth_s: float = 1.0,
                      warmup_s: float = 3.0, min_run_s: float = 4.0, v_min: float = 0.3, require_locomotion: bool = True):
    """Straight-trot segmentation with the DUAL criterion required by the spec: the body turn rate (IMU gyro z) AND the
    Fixposition heading rate must both be small, the robot must be moving, and (by default) the Go2 must be in
    locomotion mode (`mode == 3`; mode 1 = balance-stand).

    NOTE the criterion is the **windowed MEAN** of the yaw rate, not its RMS: a trotting quadruped yaw-oscillates within
    every step cycle (RMS |w_z| ~ 0.25 rad/s on this corpus even when walking dead straight), so an RMS gate rejects
    everything. The mean over a >= 1 s window cancels the gait wobble and leaves the actual turn rate. (The wheeled M1
    has no gait wobble, which is why phase.registration.straight_mask_kinematic uses RMS there.) Where the RTK reference
    is missing or its fix is not OK the RTK half is skipped and the IMU criterion carries the segment, which is recorded.
    """
    t = df["t"].to_numpy(); dt = float(np.median(np.diff(t))); w = max(1, int(round(smooth_s / dt))); k = np.ones(w) / w
    mean = lambda x: np.convolve(np.nan_to_num(np.asarray(x, float)), k, mode="same")
    wz = np.abs(mean(df["imu_w_z"].to_numpy()))
    m = (wz < wz_max) & (t >= warmup_s)
    if require_locomotion and "mode" in df:
        m &= df["mode"].to_numpy() == 3
    used_rtk = False; rtk_skipped_frac = float("nan")
    if "rtk_yaw" in df and np.isfinite(df["rtk_yaw"]).any():
        yaw = np.deg2rad(np.unwrap(np.nan_to_num(df["rtk_yaw"].to_numpy(), nan=0.0), period=360.0))
        yr = np.abs(mean(np.gradient(yaw, t))) * 180.0 / np.pi
        ok = np.isfinite(df["rtk_yaw"].to_numpy()) & df["rtk_fix_ok"].to_numpy(bool)
        m &= (~ok) | (yr < rtk_yawrate_max_deg)
        used_rtk = True; rtk_skipped_frac = float(np.mean(~ok))
    v = mean(np.nan_to_num(np.hypot(df["base_vx"].to_numpy(), df["base_vy"].to_numpy())))
    m &= v > v_min
    idx = np.where(m)[0]; kept = np.zeros_like(m); n_runs = 0; runs = []
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if t[r[-1]] - t[r[0]] >= min_run_s:
                kept[r] = True; n_runs += 1; runs.append((round(float(t[r[0]]), 1), round(float(t[r[-1]]), 1)))
    info = {"rule": "|mean w_z| < wz_max AND (RTK |mean heading rate| < thr where fix OK) AND mean|v| > v_min AND mode==3",
            "wz_max": wz_max, "rtk_yawrate_max_deg": rtk_yawrate_max_deg, "smooth_s": smooth_s, "v_min": v_min,
            "used_rtk": used_rtk, "rtk_fix_not_ok_fraction": rtk_skipped_frac, "n_runs": int(n_runs),
            "masked_s": float(kept.sum() * dt), "fraction": float(kept.mean()), "median_speed_mps": float(np.median(v[kept])) if kept.any() else float("nan"), "runs": runs}
    return kept, info
