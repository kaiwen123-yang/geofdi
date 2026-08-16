"""pi_i gating of the contact-aided invariant filter (Sprint 8 Block G).

Runs a contact-aided RIEKF (legged) or RollingRIEKF (wheeled) while gating each foot's measurement by a per-leg anomaly
weight from detect.stance_event:
  - "none"      : no gating (the plain filter).
  - "threshold" : the literature baseline -- if a stance foot's ESTIMATED world-frame velocity exceeds `foot_speed_thresh`
                  (default 0.4 m/s) its measurement covariance is inflated by `cov_inflate` (default x10). A hard variant
                  drops the measurement instead.
  - "geofdi_soft": per-leg stance-event e-process -> weight w_i = 1/(1+max(e-1,0)); the foot covariance is scaled by 1/w_i
                  (w->1 no change, w->0 covariance -> inf, i.e. the measurement is down-weighted smoothly).
  - "geofdi_hard": per-leg e-process -> hard gate; the measurement is dropped when e_i >= 1/alpha.
The literature THRESHOLD reads the estimated contact-point world speed (which the filter partly HIDES by drifting to
keep the foot "fixed") and trips at a fixed 0.4 m/s -- false-rejecting nominal touch-down/lift-off transients. The GeoFDI
gate scores the PRE-update per-foot innovation (the honest kinematic-constraint residual, not corrupted by the update) and
fires via a conformal-p / e-process calibrated on the NOMINAL event library, so its per-EVENT false-alarm rate is ~alpha.
The wheeled version scores the rolling-constraint residual. A nominal library (calibration run)
sets the conformal reference so the gate's per-EVENT false-alarm rate is ~alpha (FAR guarantee).

    res = run_gated_filter(df, kin, mode="geofdi_soft", lib=lib, ...)   # returns est path, gate log, per-step weights
    lib = build_event_library(df_nominal, kin, ...)                     # nominal per-leg score library
"""
from __future__ import annotations

import numpy as np

from ..detect.stance_event import EventLibrary, StanceEventTracker
from ..inekf.kinematics import Go2Kinematics
from ..inekf.liegroups import quat_to_rot
from ..inekf.rinekf import RIEKF
from ..sim.telemetry import JOINTS, LEGS


def _foot_world_velocity(f, h_body, h_body_prev, dt):
    """Estimated world velocity of the tracked contact point = d(R h + p)/dt using the filter state (finite difference)."""
    if h_body_prev is None:
        return 0.0
    return float(np.linalg.norm((f.R @ h_body + f.p - (f.R @ h_body_prev + f.p)) / dt))


def _score_innovation(f, leg, h, cov):
    """Per-foot innovation Mahalanobis (whitened kinematic-constraint residual) WITHOUT committing the update."""
    if leg not in f.d:
        return 0.0
    R = f.R; r = R.T @ (f.d[leg] - f.p); z = h - r                 # body-frame innovation of this foot
    S = cov + f.skf ** 2 * np.eye(3) + R.T @ f.P[6:9, 6:9] @ R      # coarse per-foot innovation cov (position block)
    try:
        return float(np.sqrt(max(z @ np.linalg.solve(S, z), 0.0)))
    except np.linalg.LinAlgError:
        return 0.0


def build_event_library(df, kin, sigma_enc=3e-3, n_legs=4, **kw):
    """Per-leg nominal stance-event score library: run the plain filter over a NOMINAL frame, and record the mean per-foot
    innovation Mahalanobis over each stance event. Conformal p-values against this library give the gate a per-event FAR ~ alpha."""
    _, info = run_gated_filter(df, kin, mode="track_only", sigma_enc=sigma_enc, **kw)
    lib = EventLibrary(n_legs)
    for ev in info["events"]:
        lib.add(ev["leg"], ev["score"])
    return lib.finalize()


def _run(df, kin, mode="none", lib=None, sigma_gyro=0.01, sigma_accel=0.1, sigma_enc=3e-3, sigma_contact=3e-3,
         sigma_kin_floor=3e-3, alpha=0.05, foot_speed_thresh=0.4, cov_inflate=10.0, threshold_hard=False, slip=None,
         use_provided_feet=False):
    # use_provided_feet=True reads the body-frame contact point from foot_{x,y,z}_{leg} columns (a real robot's own FK,
    # e.g. Leg-KILO footPosition2Body) with a fixed measurement covariance sigma_contact^2 I -- no URDF FK needed.
    # slip = dict(leg, t0, t1, vel_world=[vx,vy,vz]): a controlled world-frame foot slip that corrupts the stationary-
    # contact measurement of `leg` during [t0,t1] (the tracked contact point drifts by vel_world*(t-t0), body-framed).
    q_cols = [f"q_{l}_{j}" for l in LEGS for j in JOINTS]
    Q = df[q_cols].to_numpy(); acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    C = df[[f"c_{l}" for l in LEGS]].to_numpy() > 0.5; t = df["t"].to_numpy()
    quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy()
    pos = np.nan_to_num(df[["base_x", "base_y", "base_z"]].to_numpy()); vel = np.nan_to_num(df[["base_vx", "base_vy", "base_vz"]].to_numpy())
    FP = df[[f"foot_{ax}_{l}" for l in LEGS for ax in "xyz"]].to_numpy() if use_provided_feet else None
    R0 = quat_to_rot(quat[0]); p0 = (pos[0] + R0 @ kin.r_imu) if not use_provided_feet else np.zeros(3); v0 = np.nan_to_num(vel[0])
    f = RIEKF(R0, v0, p0, sigma_gyro=sigma_gyro, sigma_accel=sigma_accel, sigma_contact=sigma_contact, sigma_kin_floor=sigma_kin_floor)
    dt = float(np.median(np.diff(t))); prev = np.zeros(4, bool); h_prev = [None] * 4
    track = mode in ("track_only",) or mode.startswith("geofdi")
    tracker = StanceEventTracker(4, alpha=alpha) if track else None
    lib0 = lib if lib is not None else EventLibrary(4).finalize()
    est = np.zeros((len(t), 3)); W = np.ones((len(t), 4)); nees = []
    for k in range(len(t)):
        f.propagate(gyr[k], acc[k], dt)
        meas = []; w_k = np.ones(4)
        for leg in range(4):
            if C[k, leg]:
                if use_provided_feet:
                    h = FP[k, 3 * leg:3 * leg + 3]; cov = sigma_contact ** 2 * np.eye(3)
                    if not np.all(np.isfinite(h)):
                        continue
                else:
                    h, J = kin.h_and_jac(Q[k], leg); cov = J @ (sigma_enc ** 2 * np.eye(3)) @ J.T
                if slip is not None and leg == slip["leg"] and slip["t0"] <= t[k] < slip["t1"]:
                    drift_world = np.asarray(slip["vel_world"], float) * (t[k] - slip["t0"]); h = h + f.R.T @ drift_world
                if not prev[leg]:
                    f.add_contact(leg, h, cov)
                v_foot = _foot_world_velocity(f, h, h_prev[leg], dt)          # what the literature threshold reads (filter-corrupted)
                score = _score_innovation(f, leg, h, cov)                    # GeoFDI gating score: the PRE-update per-foot innovation
                if tracker is not None:
                    tracker.update(leg, True, score, lib0, t=t[k])
                if mode == "threshold":
                    if v_foot > foot_speed_thresh:
                        if threshold_hard:
                            w_k[leg] = 0.0; h_prev[leg] = h; continue
                        cov = cov * cov_inflate; w_k[leg] = 1.0 / cov_inflate
                elif mode.startswith("geofdi"):
                    e = tracker.current_e()[leg]
                    if mode == "geofdi_hard":
                        if e >= 1.0 / alpha:
                            w_k[leg] = 0.0; h_prev[leg] = h; continue
                    else:                              # geofdi_soft
                        w = 1.0 / (1.0 + max(e - 1.0, 0.0)); w_k[leg] = w; cov = cov / max(w, 1e-3)
                meas.append((leg, h, cov)); h_prev[leg] = h
            elif prev[leg]:
                f.remove_contact(leg)
                if tracker is not None:
                    tracker.update(leg, False, np.nan, lib0, t=t[k])
                h_prev[leg] = None
        prev = C[k].copy()
        if meas:
            rec = f.correct(meas, t=t[k])
            if rec is not None and np.isfinite(rec["nis"]):
                nees.append((t[k], rec["nis"] / rec["dof"]))
        est[k] = f.p; W[k] = w_k
    return f, est, {"weights": W, "nees": np.array(nees) if nees else np.zeros((0, 2)),
                    "events": tracker.events if tracker is not None else []}


def run_gated_filter(df, kin=None, mode="none", lib=None, **kw):
    """Public entry: run the (optionally gated) contact-aided RIEKF over an M1/Go2 telemetry frame. Returns (est, info)."""
    kin = kin or Go2Kinematics()
    f, est, info = _run(df, kin, mode=mode, lib=lib, **kw)
    info["p_final"] = f.p.copy()
    # ground-truth error vs the sim base pose (if present); est tracks the IMU point p, GT is base + R r_imu
    if "base_x" in df:
        gt = df[["base_x", "base_y", "base_z"]].to_numpy(); err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        info["gt_rmse_xy"] = float(np.sqrt(np.mean(err ** 2))); info["gt_end_xy"] = float(err[-1]); info["gt_err_xy"] = err
    return est, info
