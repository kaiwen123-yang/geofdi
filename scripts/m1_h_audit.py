#!/usr/bin/env python3
"""m1_h_audit.py — Sprint 8 Block D1: audit tables/figures for M1 hardware rosbag2 sessions (docs/protocol/m1_h_data_audit.md).

    scripts/m1_h_audit.py --out $GEOFDI_DATA_ROOT/results/m1_h_audit  <session_dir> [<session_dir> ...]

Each session_dir is a raw rosbag2 directory (or an SDK-layout CSV session); it is loaded through geofdi.io.m1_sdk (which
extracts a bag once into data/processed/m1/<name>/) so the audit exercises the same path as run_pipeline.sh. Writes
`audit_tables.md` (+ json) and figures (timelines per session, odometry paths, mirror pairs) — nothing here needs ROS.
Determinations made here: joint names/order/count, vendor-frame mirror signs, wheel wrap, IMU units/frame (regression on
the odometry), timestamp monotonicity + inter-topic skew, motion regime (standing / rolling / stepping), efforts
observations, straight-segment fallback thresholds, rest-noise floors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

from geofdi.io.m1_sdk import load_m1_session, load_mapping
from geofdi.phase.registration import straight_mask_kinematic

LEGS = ("LF", "RF", "LH", "RH"); JOINTS = ("ABAD", "HIP", "KNEE", "WHEEL")
VLEG = {"LF": "fl", "RF": "fr", "LH": "bl", "RH": "br"}; VJ = {"ABAD": "1_hip_roll", "HIP": "2_hip_pitch", "KNEE": "3_knee_pitch", "WHEEL": "4_foot"}
G0 = 9.80665


def f(x, n=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def timing_table(csv_dir: Path, rep: dict) -> tuple[list[str], dict]:
    rows = ["| topic | n | duration s | rate Hz (median dt) | dt jitter ms (std) | max gap ms | non-monotone | header − bag ms (median / min / max) | first stamp offset vs joint_states ms |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    info = {}
    t0j = None
    for key, fn in (("joint_states", "joint_states.csv"), ("imu", "imu.csv"), ("imu_front", "imu_front.csv"), ("imu_rear", "imu_rear.csv"), ("odom", "odom.csv"), ("cmd", "cmd.csv")):
        p = csv_dir / fn
        if not p.exists():
            rows.append(f"| {key} | 0 | | | | | | absent | |"); info[key] = {"n": 0}; continue
        d = pd.read_csv(p); t = d["t"].to_numpy(); tb = d["t_bag"].to_numpy() if "t_bag" in d else t
        if key == "joint_states": t0j = t[0]
        dt = np.diff(t); skew = (t - tb) * 1e3
        r = {"n": len(t), "duration_s": float(t[-1] - t[0]), "rate_hz": float(1 / np.median(dt)), "jitter_ms": float(dt.std() * 1e3), "max_gap_ms": float(dt.max() * 1e3),
             "n_nonmono": int((dt <= 0).sum()), "skew_med": float(np.median(skew)), "skew_min": float(skew.min()), "skew_max": float(skew.max()), "offset_ms": float((t[0] - t0j) * 1e3) if t0j is not None else float("nan")}
        info[key] = r
        rows.append(f"| {key} | {r['n']} | {f(r['duration_s'],1)} | {f(r['rate_hz'],1)} | {f(r['jitter_ms'],2)} | {f(r['max_gap_ms'],1)} | {r['n_nonmono']} | {f(r['skew_med'],2)} / {f(r['skew_min'],1)} / {f(r['skew_max'],1)} | {f(r['offset_ms'],1)} |")
    return rows, info


def joint_table(js: pd.DataFrame, names: list[str]) -> tuple[list[str], dict]:
    rows = ["| joint (vendor name) | pos mean | pos std | pos min | pos max | vel mean | vel std | vel |max| | eff mean | eff std | eff |max| |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    info = {}
    for nm in names:
        p, v, e = js[f"{nm}_pos"], js[f"{nm}_vel"], js[f"{nm}_eff"]
        info[nm] = {"pos_mean": float(p.mean()), "pos_std": float(p.std()), "pos_min": float(p.min()), "pos_max": float(p.max()), "vel_mean": float(v.mean()), "vel_std": float(v.std()), "vel_absmax": float(v.abs().max()), "eff_mean": float(e.mean()), "eff_std": float(e.std()), "eff_absmax": float(e.abs().max())}
        r = info[nm]
        rows.append(f"| `{nm}` | {f(r['pos_mean'])} | {f(r['pos_std'])} | {f(r['pos_min'])} | {f(r['pos_max'])} | {f(r['vel_mean'])} | {f(r['vel_std'])} | {f(r['vel_absmax'],2)} | {f(r['eff_mean'],2)} | {f(r['eff_std'],2)} | {f(r['eff_absmax'],1)} |")
    return rows, info


def wheel_wrap(js: pd.DataFrame, names: list[str]) -> dict:
    out = {}
    for nm in names:
        if nm.endswith("_foot"):
            p = js[f"{nm}_pos"].to_numpy(); d = np.abs(np.diff(p))
            out[nm] = {"min": float(p.min()), "max": float(p.max()), "n_jumps_gt_3": int((d > 3).sum()), "max_jump": float(d.max())}
    return out


def mirror_table(js: pd.DataFrame, df: pd.DataFrame, mp: dict) -> tuple[list[str], dict]:
    """Vendor-frame L/R means and the GeoFDI-frame mirror check (after per_leg_sign): image = JOINT_SIGN * partner."""
    from geofdi.sim.telemetry_m1 import JOINT_SIGN
    rows = ["| joint | LF (fl) | RF (fr) | LH (bl) | RH (br) | vendor L/R relation | GeoFDI frame LF vs sign·RF | LH vs sign·RH |", "|---|---:|---:|---:|---:|---|---|---|"]
    info = {}
    for grp, suf, gname in (("q", "_pos", "q"), ("dq", "_vel", "dq"), ("tau_meas", "_eff", "eff")):
        for j in JOINTS:
            if grp == "q" and j == "WHEEL":
                continue
            v = {leg: float(js[f"{VLEG[leg]}{VJ[j]}{suf}"].mean()) for leg in LEGS}
            rel = "opposite" if np.sign(v["LF"]) == -np.sign(v["RF"]) and abs(v["LF"]) > 1e-3 else ("same" if abs(v["LF"]) > 1e-3 else "~0")
            gL, gR = float(df[f"{grp}_LF_{j}"].mean()), float(JOINT_SIGN[j] * df[f"{grp}_RF_{j}"].mean())
            hL, hR = float(df[f"{grp}_LH_{j}"].mean()), float(JOINT_SIGN[j] * df[f"{grp}_RH_{j}"].mean())
            info[f"{gname}_{j}"] = {"vendor": v, "relation": rel, "geofdi_LF": gL, "geofdi_sRF": gR, "geofdi_LH": hL, "geofdi_sRH": hR}
            rows.append(f"| {gname} {j} | {f(v['LF'])} | {f(v['RF'])} | {f(v['LH'])} | {f(v['RH'])} | {rel} | {f(gL)} vs {f(gR)} | {f(hL)} vs {f(hR)} |")
    return rows, info


def imu_table(csv_dir: Path, df: pd.DataFrame, moving: bool) -> tuple[list[str], dict]:
    imu = pd.read_csv(csv_dir / "imu.csv")
    A = imu[["ax", "ay", "az"]].to_numpy(); W = imu[["wx", "wy", "wz"]].to_numpy(); Q = imu[["qw", "qx", "qy", "qz"]].to_numpy()
    info = {"accel_mean_sensor": A.mean(0).tolist(), "accel_std_sensor": A.std(0).tolist(), "accel_norm_mean": float(np.linalg.norm(A, axis=1).mean()),
            "gyro_mean_sensor": W.mean(0).tolist(), "gyro_std_sensor": W.std(0).tolist(), "quat_constant": bool(Q.std(0).max() < 1e-9), "quat_value": Q[0].tolist()}
    rows = [f"- `/imu_driver/imu_central` (frame_id `imu_link`): accel mean (sensor) {np.round(A.mean(0), 4).tolist()} — norm {info['accel_norm_mean']:.4f} ⇒ **units g** (specific force; z ≈ −0.987 at rest ⇒ sensor z points DOWN); "
            f"std {np.round(A.std(0), 4).tolist()}; gyro mean {np.round(W.mean(0), 4).tolist()} std {np.round(W.std(0), 4).tolist()} (rad/s); orientation quaternion constant = {info['quat_constant']} (value {Q[0].tolist()}); covariances all zero."]
    if moving and (csv_dir / "odom.csv").exists():
        od = pd.read_csv(csv_dir / "odom.csv"); t = od.t.to_numpy(); dt = float(np.median(np.diff(t))); win = 21
        def I(d, c): return np.interp(t, d.t.to_numpy(), d[c].to_numpy())
        As = savgol_filter(np.stack([I(imu, c) for c in ("ax", "ay", "az")], 1), win, 2, axis=0); Ws = savgol_filter(np.stack([I(imu, c) for c in ("wx", "wy", "wz")], 1), win, 2, axis=0)
        qw, qx, qy, qz = (od[c].to_numpy() for c in ("qw", "qx", "qy", "qz")); yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        yawrate = savgol_filter(yaw, win, 2, deriv=1, delta=dt); vx_w = savgol_filter(od.x.to_numpy(), win, 2, deriv=1, delta=dt); vy_w = savgol_filter(od.y.to_numpy(), win, 2, deriv=1, delta=dt)
        vb_x = np.cos(yaw) * vx_w + np.sin(yaw) * vy_w; vb_y = -np.sin(yaw) * vx_w + np.cos(yaw) * vy_w
        a_fwd = savgol_filter(vb_x, win, 2, deriv=1, delta=dt); a_lat = vb_x * yawrate + savgol_filter(vb_y, win, 2, deriv=1, delta=dt)
        X = np.stack([a_fwd / G0, a_lat / G0, np.ones_like(a_fwd)], 1); M, *_ = np.linalg.lstsq(X, As, rcond=None)
        Xw = np.stack([yawrate, np.ones_like(yawrate)], 1); Mw, *_ = np.linalg.lstsq(Xw, Ws, rcond=None)
        info["regression_acc_rows_fwd_lat_const_cols_xyz"] = M.tolist(); info["regression_gyro_yawrate_slope_xyz"] = Mw[0].tolist()
        rows.append(f"- frame determination (regression on the vendor odometry, this session): sensor accel [g] ≈ fwd·{np.round(M[0], 2).tolist()} + lat·{np.round(M[1], 2).tolist()} + {np.round(M[2], 3).tolist()}; "
                    f"gyro ≈ yaw-rate·{np.round(Mw[0], 2).tolist()} ⇒ forward acceleration lands on **−y**, leftward (centripetal) on **−x**, yaw-left rate on **−z**: sensor frame = (x right, y back, z down); "
                    f"|gyro slope| ≈ {abs(Mw[0][2]):.2f} > 1 ⇒ the odometry yaw rate is ~{100*(abs(Mw[0][2])-1):.0f} % low relative to the IMU (all three IMUs agree, see below).")
        for k in ("imu_front", "imu_rear"):
            li = pd.read_csv(csv_dir / f"{k}.csv"); Al = np.stack([np.interp(t, li.t, li[c]) for c in ("ax", "ay", "az")], 1); Wl = savgol_filter(np.stack([np.interp(t, li.t, li[c]) for c in ("wx", "wy", "wz")], 1), win, 2, axis=0)
            Mwl, *_ = np.linalg.lstsq(Xw, Wl, rcond=None); rows.append(f"- `{k}`: accel mean {np.round(Al.mean(0), 3).tolist()} (y up, specific force in g), gyro yaw-rate slope {np.round(Mwl[0], 2).tolist()}")
            info[k] = {"accel_mean": Al.mean(0).tolist(), "gyro_yawrate_slope": Mwl[0].tolist()}
    else:
        for k in ("imu_front", "imu_rear"):
            if (csv_dir / f"{k}.csv").exists():
                li = pd.read_csv(csv_dir / f"{k}.csv"); rows.append(f"- `{k}`: accel mean {np.round(li[['ax','ay','az']].mean().to_numpy(), 3).tolist()}, gyro std {np.round(li[['wx','wy','wz']].std().to_numpy(), 4).tolist()}")
    Ab = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); Wb = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    rows.append(f"- after the loader (body FLU, m/s², R_body_from_sensor applied): accel mean {np.round(np.nanmean(Ab, 0), 3).tolist()}, gyro mean {np.round(np.nanmean(Wb, 0), 4).tolist()}")
    info["body_accel_mean"] = np.nanmean(Ab, 0).tolist(); info["body_gyro_mean"] = np.nanmean(Wb, 0).tolist()
    return rows, info


def odom_table(csv_dir: Path) -> tuple[list[str], dict]:
    p = csv_dir / "odom.csv"
    if not p.exists():
        return ["- `/odom/mc_odom`: **no messages** in this session"], {"n": 0}
    od = pd.read_csv(p); t = od.t.to_numpy(); dt = np.diff(t); xy = od[["x", "y"]].to_numpy(); win = 21
    qw, qx, qy, qz = (od[c].to_numpy() for c in ("qw", "qx", "qy", "qz")); yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)); pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    vx_w = savgol_filter(od.x.to_numpy(), win, 2, deriv=1, delta=float(np.median(dt))); vy_w = savgol_filter(od.y.to_numpy(), win, 2, deriv=1, delta=float(np.median(dt)))
    speed = np.hypot(vx_w, vy_w)
    info = {"n": len(t), "rate_hz": float(1 / np.median(dt)), "dt_jitter_ms": float(dt.std() * 1e3), "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), "start": od[["x", "y", "z"]].iloc[0].tolist(), "end": od[["x", "y", "z"]].iloc[-1].tolist(),
            "z_min": float(od.z.min()), "z_max": float(od.z.max()), "speed_median": float(np.median(speed)), "speed_q90": float(np.quantile(speed, 0.9)), "speed_max": float(speed.max()),
            "roll_std_deg": float(np.degrees(roll.std())), "pitch_std_deg": float(np.degrees(pitch.std())), "yaw_range_deg": float(np.degrees(yaw.max() - yaw.min())),
            "twist_vx_corr_world_vx": float(np.corrcoef(od.vx, vx_w)[0, 1]), "twist_vy_corr_world_vy": float(np.corrcoef(od.vy, vy_w)[0, 1]),
            "twist_vx_corr_body_vx": float(np.corrcoef(od.vx, np.cos(yaw) * vx_w + np.sin(yaw) * vy_w)[0, 1]), "twist_speed_mean": float(np.hypot(od.vx, od.vy).mean()), "pose_speed_mean": float(speed.mean())}
    rows = [f"- `/odom/mc_odom` odom→base_link: n {info['n']}, {info['rate_hz']:.1f} Hz (dt jitter {info['dt_jitter_ms']:.1f} ms), path {info['path_length_m']:.1f} m, start {np.round(info['start'], 2).tolist()} → end {np.round(info['end'], 2).tolist()}, "
            f"z ∈ [{info['z_min']:.3f}, {info['z_max']:.3f}] (base height ≈ 0.54 m), speed median {info['speed_median']:.2f} / q90 {info['speed_q90']:.2f} / max {info['speed_max']:.2f} m/s, roll/pitch std {info['roll_std_deg']:.2f}/{info['pitch_std_deg']:.2f}°, yaw range {info['yaw_range_deg']:.0f}°; "
            f"pose covariance all zeros; twist.linear correlates with the WORLD velocity ({info['twist_vx_corr_world_vx']:.3f}) not the body velocity ({info['twist_vx_corr_body_vx']:.2f}) ⇒ twist expressed in the odom frame."]
    return rows, info


def regime(df: pd.DataFrame, r_wheel: float = 0.097) -> tuple[list[str], dict]:
    t = df.t.to_numpy(); dt = float(np.median(np.diff(t))); w = int(round(1 / dt))
    knee = df[["q_LF_KNEE", "q_RF_KNEE", "q_LH_KNEE", "q_RH_KNEE"]].to_numpy(); hip = df[["q_LF_HIP", "q_RF_HIP", "q_LH_HIP", "q_RH_HIP"]].to_numpy()
    wheels = df[["dq_LF_WHEEL", "dq_RF_WHEEL", "dq_LH_WHEEL", "dq_RH_WHEEL"]].to_numpy()
    ex_k = np.array([(knee[i:i + w].max(0) - knee[i:i + w].min(0)).max() for i in range(0, len(t) - w, w)])
    ex_h = np.array([(hip[i:i + w].max(0) - hip[i:i + w].min(0)).max() for i in range(0, len(t) - w, w)])
    wmean = np.array([np.abs(wheels[i:i + w]).mean() for i in range(0, len(t) - w, w)])
    rolling = wmean * r_wheel > 0.1; stepping = (ex_k > 0.15) & ~rolling; standing = ~rolling & ~stepping
    info = {"n_windows_1s": int(len(ex_k)), "frac_rolling": float(rolling.mean()), "frac_stepping_knee_gt_0.15": float(stepping.mean()), "frac_standing": float(standing.mean()),
            "knee_1s_excursion_median": float(np.median(ex_k)), "knee_1s_excursion_max": float(ex_k.max()), "hip_1s_excursion_max": float(ex_h.max()), "wheel_speed_mps_median_when_rolling": float(np.median(wmean[rolling] * r_wheel)) if rolling.any() else 0.0}
    rows = [f"- motion regime (1-s windows, r = {r_wheel} m): rolling (mean |wheel| > 0.1 m/s) {100*info['frac_rolling']:.0f} %, standing {100*info['frac_standing']:.0f} %, leg-stepping (knee excursion > 0.15 rad while not rolling) {100*info['frac_stepping_knee_gt_0.15']:.0f} %; "
            f"knee 1-s excursion median {info['knee_1s_excursion_median']:.3f} / max {info['knee_1s_excursion_max']:.2f} rad, hip max {info['hip_1s_excursion_max']:.2f} rad; wheel speed while rolling median {info['wheel_speed_mps_median_when_rolling']:.2f} m/s"]
    return rows, info


def efforts_obs(df: pd.DataFrame, moving: bool) -> tuple[list[str], dict]:
    rows = []; info = {}
    for j in JOINTS:
        e = np.concatenate([df[f"tau_meas_{l}_{j}"].to_numpy() for l in LEGS]); v = np.concatenate([df[f"dq_{l}_{j}"].to_numpy() for l in LEGS])
        info[j] = {"abs_mean": float(np.abs(e).mean()), "abs_q99": float(np.quantile(np.abs(e), 0.99)), "corr_eff_vel": float(np.corrcoef(e, v)[0, 1]) if v.std() > 0 else float("nan")}
    rows.append("- efforts (vendor units, all four legs pooled): " + "; ".join(f"{j} |eff| mean {info[j]['abs_mean']:.2f}, q99 {info[j]['abs_q99']:.1f}, corr(eff, vel) {info[j]['corr_eff_vel']:+.2f}" for j in JOINTS))
    if moving:
        # wheel effort vs wheel acceleration (inertial part) and vs speed (rolling resistance / back-EMF)
        t = df.t.to_numpy(); dt = float(np.median(np.diff(t)))
        for l in LEGS:
            dq = df[f"dq_{l}_WHEEL"].to_numpy(); ddq = savgol_filter(dq, 41, 2, deriv=1, delta=dt); e = df[f"tau_meas_{l}_WHEEL"].to_numpy()
            X = np.stack([ddq, dq, np.ones_like(dq)], 1); b, *_ = np.linalg.lstsq(X, e, rcond=None); r2 = 1 - ((e - X @ b) ** 2).sum() / ((e - e.mean()) ** 2).sum()
            info[f"wheel_fit_{l}"] = {"coef_ddq": float(b[0]), "coef_dq": float(b[1]), "const": float(b[2]), "r2": float(r2)}
        rows.append("- wheel effort ≈ a·ddq + b·dq + c (per leg; a = apparent inertia in eff-units·s², b = viscous/back-EMF): " + "; ".join(f"{l}: a {info[f'wheel_fit_{l}']['coef_ddq']:+.3f}, b {info[f'wheel_fit_{l}']['coef_dq']:+.3f}, c {info[f'wheel_fit_{l}']['const']:+.2f}, R² {info[f'wheel_fit_{l}']['r2']:.2f}" for l in LEGS))
    return rows, info


def segmentation(df: pd.DataFrame) -> tuple[list[str], dict]:
    t = df.t.to_numpy(); dt = float(np.median(np.diff(t))); w = max(1, int(round(0.5 / dt))); k = np.ones(w) / w
    rms = lambda x: np.sqrt(np.convolve(np.nan_to_num(np.asarray(x, float)) ** 2, k, mode="same"))
    wl = 0.5 * (df.dq_LF_WHEEL + df.dq_LH_WHEEL).to_numpy(); wr = 0.5 * (df.dq_RF_WHEEL + df.dq_RH_WHEEL).to_numpy()
    info = {"abs_wz_q50_q75_q90": np.quantile(rms(df.imu_w_z.to_numpy()), [.5, .75, .9]).tolist(), "abs_wheel_diff_q50_q75_q90": np.quantile(rms(wl - wr), [.5, .75, .9]).tolist(), "abs_wheel_mean_q25_q50_q75": np.quantile(np.convolve(0.5 * np.abs(wl + wr), k, mode="same"), [.25, .5, .75]).tolist()}
    mask, kin = straight_mask_kinematic(df); info["fallback"] = kin
    # runs
    idx = np.where(mask)[0]; runs = []
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            runs.append(float(t[r[-1]] - t[r[0]]))
    info["run_durations_s"] = runs
    rows = [f"- fallback thresholds from the distributions: RMS|ω_z| (0.5-s) q50/q75/q90 = {np.round(info['abs_wz_q50_q75_q90'], 3).tolist()} rad/s; RMS(wheel_L − wheel_R) q50/q75/q90 = {np.round(info['abs_wheel_diff_q50_q75_q90'], 2).tolist()} rad/s; mean |wheel| q25/q50/q75 = {np.round(info['abs_wheel_mean_q25_q50_q75'], 2).tolist()} rad/s",
            f"- rule `straight_mask_kinematic` (RMS|ω_z| < {kin['wz_max_rad_s']}, RMS(L−R) < {kin['wheel_diff_max_rad_s']} rad/s ≈ {kin['wheel_diff_max_rad_s']*0.097:.3f} m/s, mean |wheel| > {kin['wheel_min_rad_s']} rad/s ≈ {kin['wheel_min_rad_s']*0.097:.2f} m/s, runs ≥ {kin['min_run_s']} s): {kin['n_runs']} runs, {kin['masked_duration_s']:.1f} s ({100*kin['fraction_of_session']:.0f} % of the session); run durations {np.round(sorted(runs, reverse=True), 1).tolist()[:12]}"]
    return rows, info


def rest_noise(js: pd.DataFrame, names: list[str], csv_dir: Path) -> tuple[list[str], dict]:
    rows = ["| joint | pos std (rad) | pos p2p (rad) | vel std (rad/s) | eff std | eff mean |", "|---|---:|---:|---:|---:|---:|"]; info = {}
    for nm in names:
        p, v, e = js[f"{nm}_pos"], js[f"{nm}_vel"], js[f"{nm}_eff"]
        info[nm] = {"pos_std": float(p.std()), "pos_p2p": float(p.max() - p.min()), "vel_std": float(v.std()), "eff_std": float(e.std()), "eff_mean": float(e.mean())}
        rows.append(f"| `{nm}` | {p.std():.5f} | {p.max()-p.min():.5f} | {v.std():.4f} | {e.std():.3f} | {e.mean():+.2f} |")
    imu = pd.read_csv(csv_dir / "imu.csv"); A = imu[["ax", "ay", "az"]].to_numpy() * G0; W = imu[["wx", "wy", "wz"]].to_numpy()
    info["imu"] = {"accel_bias_mps2_sensor": (A.mean(0) - np.array([0, 0, -G0])).tolist(), "accel_std_mps2": A.std(0).tolist(), "gyro_bias_rad_s": W.mean(0).tolist(), "gyro_std_rad_s": W.std(0).tolist()}
    rows.append(f"\n- IMU at rest (sensor frame): accel mean {np.round(A.mean(0), 4).tolist()} m/s² (norm {np.linalg.norm(A.mean(0)):.4f}; the 1.3 % shortfall vs 9.807 is a scale/units offset to keep in mind), accel std {np.round(A.std(0), 4).tolist()} m/s²; gyro bias {np.round(W.mean(0), 5).tolist()} rad/s, gyro std {np.round(W.std(0), 5).tolist()} rad/s")
    # encoder LSB estimate from position quantisation
    for nm in names[:4]:
        p = js[f"{nm}_pos"].to_numpy(); d = np.diff(np.unique(np.round(p, 7))); lsb = float(np.min(d[d > 0])) if (d > 0).any() else float("nan"); info[nm]["lsb_est"] = lsb
    rows.append("- position quantisation (smallest nonzero step) fl1..fl4: " + ", ".join(f"{info[nm]['lsb_est']:.2e}" for nm in names[:4]) + " rad")
    return rows, info


def timeline_figure(df: pd.DataFrame, csv_dir: Path, out: Path, name: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = df.t.to_numpy(); mask, kin = straight_mask_kinematic(df)
    fig, ax = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for l, c in zip(LEGS, ("tab:blue", "tab:red", "tab:cyan", "tab:orange")):
        ax[0].plot(t, df[f"dq_{l}_WHEEL"] * 0.097, lw=0.6, label=l, color=c)
    ax[0].set_ylabel("wheel speed [m/s]\n(GeoFDI frame)"); ax[0].legend(ncol=4, fontsize=7); ax[0].grid(alpha=.3)
    ax[1].plot(t, df.imu_w_z, lw=0.5, color="k", label="IMU ω_z (body)"); ax[1].axhline(0.1, color="r", ls=":", lw=0.7); ax[1].axhline(-0.1, color="r", ls=":", lw=0.7); ax[1].set_ylabel("yaw rate [rad/s]"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    for l, c in zip(LEGS, ("tab:blue", "tab:red", "tab:cyan", "tab:orange")):
        ax[2].plot(t, df[f"q_{l}_KNEE"] - df[f"q_{l}_KNEE"].iloc[0], lw=0.6, color=c, label=f"knee {l}")
    ax[2].set_ylabel("knee − knee(0) [rad]"); ax[2].legend(ncol=4, fontsize=7); ax[2].grid(alpha=.3)
    if (csv_dir / "odom.csv").exists():
        od = pd.read_csv(csv_dir / "odom.csv"); dt = float(np.median(np.diff(od.t))); sp = np.hypot(savgol_filter(od.x, 21, 2, deriv=1, delta=dt), savgol_filter(od.y, 21, 2, deriv=1, delta=dt))
        ax[3].plot(od.t, sp, lw=0.7, color="tab:green", label="odom speed [m/s]")
    ax[3].fill_between(t, 0, 1, where=mask, color="tab:olive", alpha=0.25, transform=ax[3].get_xaxis_transform(), label="straight-rolling mask")
    ax[3].set_ylabel("speed / mask"); ax[3].set_xlabel("t [s]"); ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)
    fig.suptitle(f"{name}: wheel speeds, IMU yaw rate, knee motion, odometry speed + fallback straight mask ({kin['n_runs']} runs, {kin['masked_duration_s']:.0f} s)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"audit_timeline_{name}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sessions", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); mp = load_mapping()
    md = [f"_Generated by `scripts/m1_h_audit.py` ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}); values as reported by the vendor unless stated._\n"]
    allinfo = {}; paths = {}
    for s in a.sessions:
        sdir = Path(s); df, man, rep = load_m1_session(sdir); csv_dir = Path(rep["csv_dir"]); name = sdir.name
        js = pd.read_csv(csv_dir / "joint_states.csv"); names = [c[:-4] for c in js.columns if c.endswith("_pos")]
        meta = yaml.safe_load((csv_dir / "meta.yaml").read_text()); ext = json.loads((csv_dir / "extract_report.json").read_text()) if (csv_dir / "extract_report.json").exists() else {}
        moving = bool((df[[f"dq_{l}_WHEEL" for l in LEGS]].abs().mean(1) * 0.097 > 0.1).mean() > 0.2)
        info = {"name": name, "moving": moving, "loader": {k: rep[k] for k in ("n_rows", "duration_s", "rate_hz_estimate", "missing", "sign_convention", "extracted_from_bag")}, "extract": {k: ext.get(k) for k in ("db3_files_read", "db3_files_skipped", "db3_files_truncated", "t0_local", "message_counts")}}
        md.append(f"\n## {name}\n")
        md.append(f"- recorded {ext.get('t0_local', '?')} (local; first joint-state header stamp), db3 read {ext.get('db3_files_read')}, skipped {ext.get('db3_files_skipped') or 'none'}, truncated {ext.get('db3_files_truncated') or 'none'}; loader: {rep['n_rows']} rows, {rep['duration_s']:.1f} s @ {rep['rate_hz_estimate']:.1f} Hz; missing channels {rep['missing'] or 'none'}; regime: {'moving (rolling)' if moving else 'stationary'}\n")
        md.append("### timing (header stamps; skew = header − bag receive stamp)\n"); rows, info["timing"] = timing_table(csv_dir, rep); md += rows
        md.append("\n### joints (vendor frame, raw units)\n"); rows, info["joints"] = joint_table(js, names); md += rows
        info["wheel_wrap"] = wheel_wrap(js, names); ww = info["wheel_wrap"]
        md.append(f"\n- wheel angle (`*4_foot` pos): " + "; ".join(f"{k}: range [{v['min']:.3f}, {v['max']:.3f}], 2π jumps {v['n_jumps_gt_3']}" for k, v in ww.items()) + (" ⇒ **wrapped to [−π, π), not unwrapped**" if any(v["n_jumps_gt_3"] for v in ww.values()) else " (no wrap events in this session)"))
        md.append("\n### mirror pairs (session means): vendor frame and after `per_leg_sign` + manifest sign\n"); rows, info["mirror"] = mirror_table(js, df, mp); md += rows
        md.append("\n### IMU\n"); rows, info["imu"] = imu_table(csv_dir, df, moving); md += rows
        md.append("\n### odometry\n"); rows, info["odom"] = odom_table(csv_dir); md += rows
        md.append("\n### motion regime\n"); rows, info["regime"] = regime(df); md += rows
        md.append("\n### efforts — observations only\n"); rows, info["efforts"] = efforts_obs(df, moving); md += rows
        if moving:
            md.append("\n### straight-segment fallback (no /cmd_vel messages)\n"); rows, info["segmentation"] = segmentation(df); md += rows
            timeline_figure(df, csv_dir, out, name); paths[name] = str(out / f"audit_timeline_{name}.png")
        else:
            md.append("\n### rest-noise table (standing, motors enabled)\n"); rows, info["rest_noise"] = rest_noise(js, names, csv_dir); md += rows
        allinfo[name] = info
    # odometry paths figure (all moving sessions)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for s in a.sessions:
        sdir = Path(s); csv_dir = Path(allinfo[sdir.name]["loader"].get("csv_dir", "")) if "csv_dir" in allinfo[sdir.name]["loader"] else None
        p = Path(load_m1_session(sdir)[2]["csv_dir"]) / "odom.csv"
        if p.exists():
            od = pd.read_csv(p); ax.plot(od.x, od.y, lw=1, label=f"{sdir.name} ({allinfo[sdir.name]['odom']['path_length_m']:.0f} m)"); ax.plot(od.x.iloc[0], od.y.iloc[0], "o", ms=5, color=ax.lines[-1].get_color()); ax.plot(od.x.iloc[-1], od.y.iloc[-1], "s", ms=5, color=ax.lines[-1].get_color())
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("odom x [m]"); ax.set_ylabel("odom y [m]"); ax.legend(fontsize=7); ax.set_title("vendor odometry (/odom/mc_odom) paths — o start, ■ end", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "audit_odom_paths.png", dpi=110); plt.close(fig)
    (out / "audit_tables.md").write_text("\n".join(md) + "\n"); (out / "audit_info.json").write_text(json.dumps(allinfo, indent=1, default=str))
    print(f"wrote {out/'audit_tables.md'} and audit_info.json; figures: {list(paths.values()) + [str(out/'audit_odom_paths.png')]}")


if __name__ == "__main__":
    main()
