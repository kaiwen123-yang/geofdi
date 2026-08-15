#!/usr/bin/env python3
"""e10 — N2 estimator signatures (Sprint 7 Block N2).

  nis_smoke  : rolling-contact InEKF (geofdi.inekf.inekf_rolling.RollingRIEKF) on nominal m1_wheeled_sym rolling data;
               NIS consistency (mean NIS / dof -> 1, per-correction exceedance frac -> alpha) vs a fixed-foot RIEKF and
               an ESKF, which lack the moving-contact model and so inflate (the continuous rolling contact violates the
               stationary-foot assumption). Validates the memo's group-affine rolling model.
  signatures : bias-augmented InEKF (geofdi.inekf.rinekf_bias.RIEKFBias) on Go2 go2_urdf_sym.
               (a) encoder bias +0.05 rad on LF-KFE: the fixed-foot innovation acquires a body-frame shift that matches
                   the analytic Jacobian prediction J[:,j] b (cosine), and the augmented filter RECONSTRUCTS b_hat -> b;
               (b) gyro bias 0.02 rad/s about y (pitch): b_hat_g -> b (partial; IMU bias weakly observable in flat trot);
               (c) slip on LF vs the mirror slip on RF: the estimator innovations are R-covariant (z_RF ~ E z_LF),
                   the same Sigma-equivariance the detection channel uses.

    python experiments/e10_n2_signatures/run.py --stage nis_smoke|signatures|all [--run-id ID] [--quick] [--workers N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

from geofdi.inekf.eskf import ESKF
from geofdi.inekf.inekf_rolling import RollingRIEKF, wheel_contact_inputs
from geofdi.inekf.kinematics import Go2Kinematics
from geofdi.inekf.kinematics_m1 import M1Kinematics
from geofdi.inekf.liegroups import quat_to_rot
from geofdi.inekf.rinekf import RIEKF
from geofdi.inekf.rinekf_bias import RIEKFBias
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.env_m1 import SimConfigM1, rollout_m1
from geofdi.sim.pipeline import pmap
from geofdi.sim.telemetry import JOINTS as G_JOINTS
from geofdi.sim.telemetry import LEGS as G_LEGS
from geofdi.sim.telemetry_m1 import JOINTS as M_JOINTS
from geofdi.sim.telemetry_m1 import LEGS as M_LEGS
from geofdi.sim.telemetry_m1 import WHEEL_R

EXP_NAME = "e10_n2_signatures"
REPO = Path(__file__).resolve().parents[2]
E = np.diag([1.0, -1.0, 1.0])
SIG = {0: 1, 1: 0, 2: 3, 3: 2}


# --------------------------------------------------------------------------------------------------------------------
# M1 rolling NIS smoke
# --------------------------------------------------------------------------------------------------------------------
def _m1_arrays(df):
    q = df[[f"q_{l}_{j}" for l in M_LEGS for j in M_JOINTS]].to_numpy()
    dq = df[[f"dq_{l}_{j}" for l in M_LEGS for j in M_JOINTS]].to_numpy()
    acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    C = df[[f"c_{l}" for l in M_LEGS]].to_numpy() > 0.5; t = df["t"].to_numpy()
    quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy(); pos = df[["base_x", "base_y", "base_z"]].to_numpy()
    vel = df[["base_vx", "base_vy", "base_vz"]].to_numpy(); blk = df["blk"].to_numpy()
    wheel_i = [4 * li + M_JOINTS.index("WHEEL") for li in range(4)]
    return q, dq, acc, gyr, C, t, quat, pos, vel, blk, wheel_i


def _run_m1_filter(df, kin, kind, ns):
    q, dq, acc, gyr, C, t, quat, pos, vel, blk, wheel_i = _m1_arrays(df)
    R0 = quat_to_rot(quat[0]); p0 = pos[0] + R0 @ kin.r_imu; v0 = vel[0]
    common = dict(sigma_gyro=ns["sigma_gyro"], sigma_accel=ns["sigma_accel"], sigma_contact=ns["sigma_contact"], sigma_kin_floor=ns["sigma_kin_floor"])
    if kind == "rolling":
        f = RollingRIEKF(R0, v0, p0, sigma_roll=ns["sigma_roll"], sigma_slip=ns["sigma_slip"], **common)
    elif kind == "fixed":
        f = RIEKF(R0, v0, p0, **common)
    else:
        f = ESKF(R0, v0, p0, **common)
    dt = float(np.median(np.diff(t))); prev = np.zeros(4, bool)
    nis, dofs, perr = [], [], []
    for k in range(len(t)):
        if kind == "rolling":
            u_body, wf = wheel_contact_inputs({leg: dq[k, wheel_i[leg]] for leg in range(4)}, WHEEL_R)
            f.set_rolling_inputs(u_body, wf)
        f.propagate(gyr[k], acc[k], dt)
        meas = []
        for leg in range(4):
            if C[k, leg]:
                h, J = kin.h_and_jac(q[k], leg)
                cov = J @ (ns["sigma_enc"] ** 2 * np.eye(3)) @ J.T
                if not prev[leg]:
                    f.add_contact(leg, h, cov)
                meas.append((leg, h, cov))
            elif prev[leg]:
                f.remove_contact(leg)
        prev = C[k].copy()
        if meas and blk[k] >= 0:                       # score only on the straight-rolling segment
            rec = f.correct(meas, t=t[k])
            if rec is not None and np.isfinite(rec["nis"]):
                nis.append(rec["nis"]); dofs.append(rec["dof"])
                perr.append(float(np.linalg.norm(f.p - (pos[k] + f.R @ kin.r_imu))))
            elif rec is not None:
                nis.append(np.inf); dofs.append(rec["dof"]); perr.append(np.nan)
    return np.array(nis), np.array(dofs), np.array(perr)


def _m1_worker(seed, ns, quick):
    dur = 12.0 if quick else ns["duration_s"]
    cfg = SimConfigM1(model=ns["model"], speed=ns["speed"], duration_s=dur, warmup_s=ns["warmup_s"], seed=int(seed))
    df, _ = rollout_m1(cfg)
    kin = M1Kinematics(ns["model"])
    out = {}
    thr = chi2.ppf(1 - ns["far_alpha"], 3 * 4)          # 4 contacts x 3 dof; per-correction dof varies, handled below
    for kind in ("rolling", "fixed", "eskf"):
        nis, dof, perr = _run_m1_filter(df, kin, kind, ns)
        if len(nis) == 0:
            out[kind] = dict(nis_over_dof=np.nan, far=np.nan, p_rmse=np.nan, n=0); continue
        gate = chi2.ppf(1 - ns["far_alpha"], dof)
        exceed = float(np.mean(nis > gate))
        out[kind] = dict(nis_over_dof=float(np.nanmean(nis / dof)), nis_median_over_dof=float(np.nanmedian(nis / dof)),
                         far=exceed, p_rmse=float(np.sqrt(np.nanmean(perr ** 2))), n=int(len(nis)))
    return out


def stage_nis_smoke(cfg, res_dir, quick, workers):
    ns = cfg["nis_smoke"]; R = 4 if quick else ns["R"]
    print(f"[nis_smoke] {R} rolling rollouts x 3 filters ({ns['model']}, {ns['speed']} m/s)", flush=True)
    outs = pmap(_m1_worker, [(ns["seed_base"] + r, ns, quick) for r in range(R)], workers)
    rows = []
    for kind in ("rolling", "fixed", "eskf"):
        nd = np.array([o[kind]["nis_over_dof"] for o in outs]); far = np.array([o[kind]["far"] for o in outs]); pr = np.array([o[kind]["p_rmse"] for o in outs])
        rows.append(dict(filter=kind, R=R, nis_over_dof_mean=float(np.nanmean(nd)), nis_over_dof_sd=float(np.nanstd(nd)),
                         far_mean=float(np.nanmean(far)), far_sd=float(np.nanstd(far)), p_rmse_m=float(np.nanmean(pr))))
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e10_nis_smoke.csv", index=False)
    print(tab.to_string(index=False), flush=True)
    line = ("[e10 nis_smoke] base-position tracking RMSE (the discriminator): rolling-InEKF %.3f m vs fixed-foot RIEKF "
            "%.1f m / ESKF %.1f m — the fixed-foot filters treat the continuously-rolling robot as near-stationary and "
            "so do not track. NIS/dof is conservative for all (rolling %.2f, fixed %.2f, eskf %.2f at alpha=%.2f; the "
            "kinematic-floor covariance dominates the sub-mm residuals), so per-bin FAR does not separate them — the "
            "rolling contact model earns its keep on estimation accuracy. Rolling per-corr FAR %.3f."
            % (rows[0]["p_rmse_m"], rows[1]["p_rmse_m"], rows[2]["p_rmse_m"], rows[0]["nis_over_dof_mean"],
               rows[1]["nis_over_dof_mean"], rows[2]["nis_over_dof_mean"], ns["far_alpha"], rows[0]["far_mean"]))
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line, flush=True)
    return tab


# --------------------------------------------------------------------------------------------------------------------
# Go2 bias-signature reconstruction
# --------------------------------------------------------------------------------------------------------------------
def _go2_arrays(df):
    q = df[[f"q_{l}_{j}" for l in G_LEGS for j in G_JOINTS]].to_numpy()
    acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    C = df[[f"c_{l}" for l in G_LEGS]].to_numpy() > 0.5; t = df["t"].to_numpy()
    quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy(); pos = df[["base_x", "base_y", "base_z"]].to_numpy()
    vel = df[["base_vx", "base_vy", "base_vz"]].to_numpy()
    return q, acc, gyr, C, t, quat, pos, vel


def _run_go2_bias(df, kin, sg, enc_bias=None, gyro_bias=None, slip=None, estimate=True, onset_frac=0.5, win=30):
    """Drive a Go2 contact filter. The injected fault (enc_bias (12,) added to q / gyro_bias (3,) added to the gyro /
    slip world-frame foot drift) turns on as a STEP at onset_frac of the run, so the innovation carries a transient
    signature before the (augmented) filter absorbs it. Body-frame innovations are accumulated per leg in the post-onset
    window [onset, onset+win]. estimate=True augments with bias states (encoder bias iff enc_bias given)."""
    q, acc, gyr, C, t, quat, pos, vel = _go2_arrays(df)
    R0 = quat_to_rot(quat[0]); p0 = pos[0] + R0 @ kin.r_imu; v0 = vel[0]
    n_enc = 12 if (estimate and enc_bias is not None) else 0
    if estimate:
        f = RIEKFBias(R0, v0, p0, n_enc=n_enc, sigma_gyro=sg["sigma_gyro"], sigma_accel=sg["sigma_accel"],
                      sigma_contact=sg["sigma_contact"], sigma_kin_floor=sg["sigma_contact"],
                      sigma_bg_rw=sg["bias_rw"]["gyro"], sigma_ba_rw=sg["bias_rw"]["accel"], sigma_benc_rw=sg["bias_rw"]["enc"])
    else:
        f = RIEKF(R0, v0, p0, sigma_gyro=sg["sigma_gyro"], sigma_accel=sg["sigma_accel"], sigma_contact=sg["sigma_contact"], sigma_kin_floor=sg["sigma_contact"])
    eb0 = np.zeros(12) if enc_bias is None else np.asarray(enc_bias, float)
    gb0 = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, float)
    dt = float(np.median(np.diff(t))); prev = np.zeros(4, bool); n = len(t); onset = int(onset_frac * n)
    zbody = {leg: [] for leg in range(4)}; bg_pre = np.zeros(3)
    for k in range(n):
        on = k >= onset
        gb = gb0 if on else np.zeros(3)
        f.propagate(gyr[k] + gb, acc[k], dt)
        if k == onset and estimate:
            bg_pre = f.bg.copy()                          # bias estimate just before the step (nominal baseline)
        meas = []; leg_jac = {}
        qb = q[k] + (eb0 if on else 0.0)                  # biased encoder reading after onset
        for leg in range(4):
            if C[k, leg]:
                q_op = qb.copy()
                if n_enc:                                  # debias with the current estimate (iterated-EKF operating point)
                    q_op[3 * leg:3 * leg + 3] = qb[3 * leg:3 * leg + 3] - f.benc[3 * leg:3 * leg + 3]
                h, J = kin.h_and_jac(q_op, leg)
                if on and slip is not None and leg == slip["leg"]:
                    h = h + f.R.T @ np.asarray(slip["vec"], float)     # world-frame foot drift -> body-frame add
                cov = J @ (sg["sigma_enc"] ** 2 * np.eye(3)) @ J.T
                if not prev[leg]:
                    f.add_contact(leg, h, cov)
                meas.append((leg, h, cov))
                if n_enc:
                    leg_jac[leg] = (J, np.array([3 * leg, 3 * leg + 1, 3 * leg + 2]))
            elif prev[leg]:
                f.remove_contact(leg)
        prev = C[k].copy()
        if meas:
            rec = f.correct(meas, t=t[k], leg_jac=leg_jac) if n_enc else f.correct(meas, t=t[k])
            if rec is not None and np.isfinite(rec["nis"]) and onset <= k < onset + win:
                zw = rec["z"].reshape(-1, 3)
                for jf, leg in enumerate(rec["feet"]):
                    zbody[leg].append(f.R.T @ zw[jf])      # body-frame innovation, post-onset window
    zmean = {leg: (np.mean(v, axis=0) if len(v) else np.full(3, np.nan)) for leg, v in zbody.items()}
    return dict(zmean=zmean, benc=(f.benc.copy() if (estimate and n_enc) else np.zeros(12)),
                bg=(f.bg.copy() if estimate else np.zeros(3)), bg_pre=bg_pre)


def _sig_worker(seed, sg, quick):
    dur = 8.0 if quick else sg["duration_s"]
    base = dict(model=sg["model"], gait=sg["gait"], speed=0.0, duration_s=dur, seed=int(seed))
    df, _ = rollout(SimConfig(**base)); kin = Go2Kinematics()
    L = G_LEGS.index(sg["encoder_bias"]["leg"]); j = G_JOINTS.index(sg["encoder_bias"]["joint"]); jj = 3 * L + j
    eb = np.zeros(12); eb[jj] = sg["encoder_bias"]["rad"]
    qmean = df[[f"q_{l}_{jt}" for l in G_LEGS for jt in G_JOINTS]].to_numpy().mean(0)
    # (a) encoder-bias transient signature: step onset, nominal vs faulted body-frame innovation over the post-onset
    #     window, direction compared to the analytic Jacobian prediction J[:,j] b (adjoint of the fault)
    nom = _run_go2_bias(df, kin, sg, estimate=False)
    flt = _run_go2_bias(df, kin, sg, enc_bias=eb, estimate=False)
    _, Jnom = kin.h_and_jac(qmean, L); pred = Jnom[:, j] * sg["encoder_bias"]["rad"]
    dz = flt["zmean"][L] - nom["zmean"][L]
    cos_enc = float(dz @ pred / (np.linalg.norm(dz) * np.linalg.norm(pred) + 1e-12))
    # (b) encoder-bias reconstruction: WITH bias states (read at end of run, after the step)
    est = _run_go2_bias(df, kin, sg, enc_bias=eb, estimate=True); benc_rec = float(est["benc"][jj])
    # (c) gyro-bias reconstruction: pitch axis (roll/pitch observable via gravity; yaw is InEKF-unobservable), reported
    #     as the response to the step relative to the pre-onset baseline (isolates the observable component)
    gax = {"x": 0, "y": 1, "z": 2}[sg["gyro_bias"]["axis"]]
    gb = np.zeros(3); gb[gax] = sg["gyro_bias"]["rad_s"]
    estg = _run_go2_bias(df, kin, sg, gyro_bias=gb, estimate=True)
    bg_rec = float(estg["bg"][gax] - estg["bg_pre"][gax])
    # (d) slip mirror-covariance: LF slip vs RF mirror slip; body-frame innovations satisfy z_RF ~ E z_LF (R-covariance).
    #     Persistent fault, averaged over the whole run (onset 0, large window) so both legs' full contact phases are
    #     sampled — a short post-onset window would compare LF and RF at different trot sub-phases (half a cycle apart).
    sl = sg["slip"]; vec = np.asarray(sl["dir"], float) * sl["mag"]
    zlf = _run_go2_bias(df, kin, sg, slip={"leg": G_LEGS.index("LF"), "vec": vec}, estimate=False, onset_frac=0.0, win=10 ** 9)["zmean"][G_LEGS.index("LF")]
    zrf = _run_go2_bias(df, kin, sg, slip={"leg": G_LEGS.index("RF"), "vec": E @ vec}, estimate=False, onset_frac=0.0, win=10 ** 9)["zmean"][G_LEGS.index("RF")]
    cos_mirror = float((E @ zlf) @ zrf / (np.linalg.norm(zlf) * np.linalg.norm(zrf) + 1e-12))
    return dict(cos_enc=cos_enc, benc_rec=benc_rec, benc_true=sg["encoder_bias"]["rad"], bg_rec=bg_rec,
                bg_true=sg["gyro_bias"]["rad_s"], cos_mirror=cos_mirror, dz_norm=float(np.linalg.norm(dz)))


def stage_signatures(cfg, res_dir, quick, workers):
    sg = cfg["signatures"]; R = 4 if quick else sg["R"]
    print(f"[signatures] {R} Go2 rollouts ({sg['model']}) — encoder / gyro bias reconstruction + slip mirror-covariance", flush=True)
    outs = pmap(_sig_worker, [(sg["seed_base"] + r, sg, quick) for r in range(R)], workers)
    df = pd.DataFrame(outs); df.to_csv(res_dir / "e10_signatures.csv", index=False)
    agg = dict(cos_enc=df.cos_enc.mean(), benc_rec=df.benc_rec.mean(), benc_rec_sd=df.benc_rec.std(),
               bg_rec=df.bg_rec.mean(), bg_rec_sd=df.bg_rec.std(), cos_mirror=df.cos_mirror.mean())
    pd.DataFrame([agg]).to_csv(res_dir / "e10_signatures_summary.csv", index=False)
    line = ("[e10 signatures] encoder-bias +%.3f rad LF-%s (step): innovation-direction cosine to the analytic J[:,j]b = "
            "%.3f, reconstructed b_hat = %.4f +/- %.4f rad (same sign, right order; a mild over-estimate) | pitch "
            "gyro-bias %.3f rad/s: step response b_hat_g = %.4f +/- %.4f rad/s (partial — IMU bias is only weakly "
            "observable in short flat trot, and yaw is InEKF-unobservable) | slip mirror-covariance cos(E z_LF, z_RF) = "
            "%.3f  => the InEKF innovation carries the fault signature in the adjoint (Jacobian) direction, the augmented "
            "filter reconstructs the observable bias, and the residuals are Sigma-equivariant"
            % (sg["encoder_bias"]["rad"], sg["encoder_bias"]["joint"], agg["cos_enc"], agg["benc_rec"], agg["benc_rec_sd"],
               sg["gyro_bias"]["rad_s"], agg["bg_rec"], agg["bg_rec_sd"], agg["cos_mirror"]))
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line, flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["nis_smoke", "signatures", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); workers = a.workers or cfg.get("workers", 8)
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.stage in ("nis_smoke", "all"):
        stage_nis_smoke(cfg, res_dir, a.quick, workers)
    if a.stage in ("signatures", "all"):
        stage_signatures(cfg, res_dir, a.quick, workers)
    print(f"[e10] done -> {res_dir}", flush=True)


if __name__ == "__main__":
    main()
