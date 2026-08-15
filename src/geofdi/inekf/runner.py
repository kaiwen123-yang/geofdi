"""Run a filter (RIEKF or ESKF) over M1-schema telemetry: IMU propagation each control step, kinematic corrections
for feet in contact (contact flags from the telemetry), contact add/remove on flag edges. Uses only measured
quantities (q_meas, IMU, contacts) plus the kinematic model; ground truth is used only to initialize (R0, v0, p0)."""
from __future__ import annotations

import numpy as np

from ..sim.telemetry import JOINTS, LEGS
from .eskf import ESKF
from .kinematics import Go2Kinematics
from .liegroups import quat_to_rot
from .rinekf import RIEKF


def run_filter(df, kind: str = "riekf", sigma_gyro=0.01, sigma_accel=0.1, sigma_enc=2e-3, sigma_contact=2e-3, sigma_kin_floor=2e-3,
               correct_every: int = 1, init_from_truth: bool = True, meas_perturb=None, gyro_bias=None, kin=None,
               kick=None, gyro_noise_add: float = 0.0, rng=None, meas_cov_add=None):
    """meas_perturb(t, leg, h_body, R_true, p_true, d_true) -> h_body' lets experiments inject world-frame noise / faults into
    the kinematic measurement; gyro_bias (3,) is added to the measured gyro (fault injection).
    kick = dict(period_s, rot_deg, vel, jitter_s, start_s): at random times (period +- jitter) the estimate is perturbed by a
    random world-frame rotation (angle rot_deg, uniform axis) and a velocity offset N(0, vel^2 I) with P inflated
    accordingly (large-error regime); gyro_noise_add adds white noise (std, rad/s) to the gyro input (sensor degradation)."""
    kin = kin or Go2Kinematics()
    q_cols = [f"q_{l}_{j}" for l in LEGS for j in JOINTS]
    Q = df[q_cols].to_numpy(); acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    C = df[[f"c_{l}" for l in LEGS]].to_numpy() > 0.5; t = df["t"].to_numpy()
    quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy(); pos = df[["base_x", "base_y", "base_z"]].to_numpy()
    vel = df[["base_vx", "base_vy", "base_vz"]].to_numpy()
    R0 = quat_to_rot(quat[0]) if init_from_truth else np.eye(3)
    p0 = pos[0] + R0 @ kin.r_imu if init_from_truth else np.array([0, 0, 0.3])
    v0 = vel[0] if init_from_truth else np.zeros(3)
    Filt = RIEKF if kind == "riekf" else ESKF
    f = Filt(R0, v0, p0, sigma_gyro=float(np.hypot(sigma_gyro, gyro_noise_add)), sigma_accel=sigma_accel, sigma_contact=sigma_contact, sigma_kin_floor=sigma_kin_floor)
    dt = float(np.median(np.diff(t)))
    prev = np.zeros(4, dtype=bool)
    est = np.zeros((len(t), 3)); Rest = np.zeros((len(t), 3, 3))
    rng = np.random.default_rng(0) if rng is None else rng
    kick_times = []
    if kick:
        tk = float(kick.get("start_s", 0.0)) + kick["period_s"] * rng.uniform(0.2, 1.0)
        while tk < t[-1]:
            kick_times.append(tk); tk += kick["period_s"] + rng.uniform(-kick.get("jitter_s", 0.0), kick.get("jitter_s", 0.0))
    kick_idx = set(int(np.searchsorted(t, tk)) for tk in kick_times); f.kick_times = list(kick_times)
    gyro_extra = rng.normal(0.0, gyro_noise_add, size=gyr.shape) if gyro_noise_add > 0 else None
    for k in range(len(t)):
        gb = 0.0 if gyro_bias is None else (gyro_bias[k] if np.ndim(gyro_bias) == 2 else gyro_bias)
        g = gyr[k] + gb + (gyro_extra[k] if gyro_extra is not None else 0.0)
        if k in kick_idx:
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = np.deg2rad(kick["rot_deg"])
            f.kick(ang * ax, rng.normal(0.0, kick["vel"], size=3), sig_theta=ang, sig_v=kick["vel"])
        f.propagate(g, acc[k], dt)
        meas = []
        for leg in range(4):
            if C[k, leg]:
                h, J = kin.h_and_jac(Q[k], leg)
                if meas_perturb is not None:
                    Rt = quat_to_rot(quat[k]); pt = pos[k] + Rt @ kin.r_imu
                    h = meas_perturb(t[k], leg, h, Rt, pt)
                cov = J @ (sigma_enc**2 * np.eye(3)) @ J.T
                if meas_cov_add is not None:
                    cov = cov + meas_cov_add                       # extra BODY-frame measurement-noise model (experiments)
                if not prev[leg]:
                    f.add_contact(leg, h, cov)
                meas.append((leg, h, cov))
            elif prev[leg]:
                f.remove_contact(leg)
        prev = C[k].copy()
        if meas and (k % correct_every == 0):
            f.correct(meas, t=t[k])
        est[k] = f.p; Rest[k] = f.R
    return f, est, Rest
