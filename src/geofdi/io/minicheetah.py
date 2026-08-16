"""MIT Mini Cheetah contact dataset loader (Sprint 8 Block PUB, e17).

UMich-CURLY/deep-contact-estimator .mat files -> the GeoFDI Go2-schema telemetry frame + manifest. Fields per file
(1000 Hz): q/qd/tau_est (N,12), foot position p / velocity v / GRF F (N,12), contacts (N,4, uint8), imu_acc/imu_omega
(N,3), imu_quat (N,4, wxyz), control_time/imu_time (N,). Leg order is MIT [FR, FL, RR, RL], joints [ab/ad, hip, knee];
the GeoFDI order is [LF, RF, LH, RH]=[FL,FR,RL,RR]=cheetah[1,0,3,2] with joints [HAA, HFE, KFE]. The native Cheetah frame
already matches the GeoFDI mirror convention (ab/ad opposite between mirror legs, hip/knee same), verified on the nominal
stance means, so per_leg_sign = +1 throughout (unlike the M1 vendor frame). IMU is body FLU, specific force in m/s^2
(z ~ +9.8 at rest). tau_est -> tau_meas (the estimated joint torque); tau_cmd is not in the dataset (NaN).

    df, manifest, report = load_minicheetah(mat_path)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..sim.telemetry import JOINTS, LEGS, all_columns, build_manifest

CHEETAH_LEG = {"LF": 1, "RF": 0, "LH": 3, "RH": 2}          # GeoFDI leg -> cheetah leg index (FR,FL,RR,RL)
JOINT_IDX = {"HAA": 0, "HFE": 1, "KFE": 2}                   # cheetah per-leg joint order [abad, hip, knee]


def load_minicheetah(mat_path: str | Path):
    mat_path = Path(mat_path); m = scipy.io.loadmat(mat_path)
    q = m["q"]; qd = m["qd"]; tau = m["tau_est"]; C = m["contacts"].astype(float)
    t = m["control_time"].ravel(); n = len(q); t = t - t[0]
    df = pd.DataFrame(np.nan, index=np.arange(n), columns=all_columns()); df["t"] = t; df["theta"] = np.nan
    def col(arr, leg, j):
        return arr[:, 3 * CHEETAH_LEG[leg] + JOINT_IDX[j]]
    for leg in LEGS:
        for j in JOINTS:
            df[f"q_{leg}_{j}"] = col(q, leg, j); df[f"dq_{leg}_{j}"] = col(qd, leg, j); df[f"tau_meas_{leg}_{j}"] = col(tau, leg, j)
        df[f"c_{leg}"] = C[:, CHEETAH_LEG[leg]]; df[f"temp_{leg}"] = np.nan
    A = m["imu_acc"]; W = m["imu_omega"]; Qt = m["imu_quat"]      # body FLU, specific force m/s^2, quat wxyz
    for k, ax in enumerate("xyz"):
        df[f"imu_a_{ax}"] = A[:, k]; df[f"imu_w_{ax}"] = W[:, k]
    for k, qn in enumerate(("base_qw", "base_qx", "base_qy", "base_qz")):
        df[qn] = Qt[:, k]
    manifest = build_manifest(sim_meta={"source": "public (MIT Mini Cheetah contact dataset, UMich-CURLY/deep-contact-estimator)",
                                        "robot": "mini_cheetah", "rate_hz": float(1.0 / np.median(np.diff(t))), "sequence": mat_path.stem,
                                        "efforts_semantics": "estimated_torque (tau_est)", "mapping_unverified": False, "imu_frame": "body FLU, specific force m/s^2"})
    report = {"n_rows": int(n), "duration_s": float(t[-1] - t[0]), "rate_hz_estimate": float(1.0 / np.median(np.diff(t))),
              "contact_duty": {leg: float(df[f"c_{leg}"].mean()) for leg in LEGS}, "sequence": mat_path.stem,
              "efforts_semantics": "estimated_torque", "mapping_unverified": False}
    return df, manifest, report


def stance_trot_mask(df, wz_max=0.6, warmup_s=2.0, min_run_s=3.0):
    """Straight-trot segmentation: rows where the smoothed body yaw rate |w_z| < wz_max and all four legs are cycling
    (mean contact duty in [0.2,0.8] over a 1 s window); runs shorter than min_run_s dropped. Returns (mask, info)."""
    t = df["t"].to_numpy(); dt = float(np.median(np.diff(t))); w = max(1, int(round(0.3 / dt))); k = np.ones(w) / w
    wz = np.convolve(np.abs(np.nan_to_num(df["imu_w_z"].to_numpy())), k, mode="same")
    duty = np.mean([np.convolve(df[f"c_{l}"].to_numpy(), np.ones(int(1 / dt)) / int(1 / dt), mode="same") for l in LEGS], axis=0)
    m = (wz < wz_max) & (duty > 0.15) & (duty < 0.85) & (t >= warmup_s)
    idx = np.where(m)[0]; kept = np.zeros_like(m); n_runs = 0
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if t[r[-1]] - t[r[0]] >= min_run_s:
                kept[r] = True; n_runs += 1
    return kept, {"wz_max": wz_max, "n_runs": int(n_runs), "masked_s": float(kept.sum() * dt), "fraction": float(kept.mean())}
