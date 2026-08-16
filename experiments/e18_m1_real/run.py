#!/usr/bin/env python3
"""e18 — first GeoFDI experiments on the M1 hardware sessions (Sprint 8 Block D3).

Stages
  inekf   : rolling RIEKF vs rolling ESKF vs fixed-foot RIEKF vs fixed-foot ESKF on the three real rolling sessions
            (IMU + measured wheel rates + a declared constant contact geometry; reference = vendor odometry, NOT truth):
            per-straight-run displacement error, whole-session position error after alignment, path length, yaw drift.
  summary : collects the run_pipeline.sh reports of the sessions (results/pipeline/m1real_<session>_L{1,2}) into one table
            + a 3-session figure (H0/H0' per-window p and e-processes).

    python experiments/e18_m1_real/run.py --stage inekf|summary|all [--quick]
Outputs -> $GEOFDI_DATA_ROOT/results/e18_m1_real/<run>/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.inekf.eskf import ESKF
from geofdi.inekf.inekf_rolling import RollingESKF, RollingRIEKF, wheel_contact_inputs
from geofdi.inekf.liegroups import quat_to_rot
from geofdi.inekf.rinekf import RIEKF
from geofdi.io.m1_sdk import load_m1_session
from geofdi.phase.registration import straight_mask_kinematic

REPO = Path(__file__).resolve().parents[2]
LEGS = ("LF", "RF", "LH", "RH")
G0 = 9.80665


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def run_filters(df: pd.DataFrame, cfg: dict, quick: bool = False):
    ns = cfg["inekf"]; t = df["t"].to_numpy(); dt = float(np.median(np.diff(t)))
    if quick:
        n = int(30 / dt); df = df.iloc[:n]; t = t[:n]
    acc = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy() / float(ns.get("accel_norm_correction", 1.0))
    gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    dqw = df[[f"dq_{l}_WHEEL" for l in LEGS]].to_numpy()
    quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy(); pos = df[["base_x", "base_y", "base_z"]].to_numpy()
    R0 = quat_to_rot(quat[0]); p0 = pos[0].copy()
    # initial velocity from the odometry position (first 0.5 s)
    k5 = int(0.5 / dt); v0 = (pos[k5] - pos[0]) / (t[k5] - t[0])
    hb = {i: np.asarray(ns["contact_points_body"][l], float) for i, l in enumerate(LEGS)}
    common = dict(sigma_gyro=ns["sigma_gyro"], sigma_accel=ns["sigma_accel"], sigma_contact=ns["sigma_contact"], sigma_kin_floor=ns["sigma_kin_floor"])
    out = {}
    for kind in ns["filters"]:
        if kind == "rolling_riekf":
            f = RollingRIEKF(R0, v0, p0, sigma_roll=ns["sigma_roll"], sigma_slip=ns["sigma_slip"], **common)
        elif kind == "rolling_eskf":
            f = RollingESKF(R0, v0, p0, sigma_roll=ns["sigma_roll"], sigma_slip=ns["sigma_slip"], **common)
        elif kind == "fixed_riekf":
            f = RIEKF(R0, v0, p0, **common)
        else:
            f = ESKF(R0, v0, p0, **common)
        cov = ns["sigma_enc"] ** 2 * np.eye(3)
        for i in range(4):
            f.add_contact(i, hb[i], cov)
        P = np.zeros((len(t), 3)); Y = np.zeros(len(t))
        for k in range(len(t)):
            if kind.startswith("rolling"):
                u_body, wf = wheel_contact_inputs({i: dqw[k, i] for i in range(4)}, ns["wheel_radius"])
                f.set_rolling_inputs(u_body, wf)
            f.propagate(gyr[k], acc[k], dt)
            f.correct([(i, hb[i], cov) for i in range(4)], t=t[k])
            P[k] = f.p; Y[k] = _yaw(f.R)
        out[kind] = {"p": P, "yaw": Y}
    # references
    qw, qx, qy, qz = quat.T
    yaw_odom = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    yaw_imu = _yaw(R0) + np.concatenate([[0.0], np.cumsum(gyr[1:, 2] * np.diff(t))])       # gyro-integrated yaw (bias 1e-4 rad/s -> negligible)
    return t, pos, yaw_odom, yaw_imu, out, df


def metrics(t, pos, yaw_odom, yaw_imu, out, df, warmup_s):
    mask, info = straight_mask_kinematic(df, warmup_s=warmup_s)
    idx = np.where(mask)[0]; runs = [r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)] if len(idx) else []
    L_odom = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())
    rows = []
    for kind, o in out.items():
        P = o["p"]; err = np.linalg.norm(P[:, :2] - pos[:, :2], axis=1)
        per_run = []
        for r in runs:
            a, b = r[0], r[-1]
            d_est = P[b, :2] - P[a, :2]; d_ref = pos[b, :2] - pos[a, :2]; L = float(np.linalg.norm(d_ref))
            arc_est = float(np.linalg.norm(np.diff(P[a:b + 1, :2], axis=0), axis=1).sum()); arc_ref = float(np.linalg.norm(np.diff(pos[a:b + 1, :2], axis=0), axis=1).sum())
            per_run.append({"t0": float(t[a]), "t1": float(t[b]), "run_len_m": L, "disp_err_m": float(np.linalg.norm(d_est - d_ref)), "ratio": float(np.linalg.norm(d_est - d_ref) / max(L, 1e-6)),
                            "arclen_est_m": arc_est, "arclen_ref_m": arc_ref, "arclen_ratio_err": float(abs(arc_est - arc_ref) / max(arc_ref, 1e-6)),
                            "yaw_drift_vs_imu_deg": float(np.degrees(np.unwrap([o["yaw"][a], o["yaw"][b]])[1] - o["yaw"][a] - (yaw_imu[b] - yaw_imu[a])))})
        Y = np.unwrap(o["yaw"])
        rows.append({"filter": kind, "rmse_xy_m": float(np.sqrt((err ** 2).mean())), "end_err_xy_m": float(err[-1]), "max_err_xy_m": float(err.max()),
                     "path_len_est_m": float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()), "path_len_odom_m": L_odom,
                     "yaw_end_vs_odom_deg": float(np.degrees(Y[-1] - Y[0] - (yaw_odom[-1] - yaw_odom[0]))), "yaw_end_vs_imu_deg": float(np.degrees(Y[-1] - Y[0] - (yaw_imu[-1] - yaw_imu[0]))),
                     "n_runs": len(per_run), "run_disp_err_median_m": float(np.median([r["disp_err_m"] for r in per_run])) if per_run else np.nan,
                     "run_ratio_median": float(np.median([r["ratio"] for r in per_run])) if per_run else np.nan, "run_ratio_max": float(np.max([r["ratio"] for r in per_run])) if per_run else np.nan,
                     "run_arclen_err_median": float(np.median([r["arclen_ratio_err"] for r in per_run])) if per_run else np.nan, "path_len_recovered_frac": float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum() / max(L_odom, 1e-6)),
                     "per_run": per_run})
    return rows, runs


def stage_inekf(cfg, res_dir, quick):
    root = Path(os.environ["GEOFDI_DATA_ROOT"]); tab = []; figs = {}
    for s in cfg["sessions"]:
        df, man, rep = load_m1_session(root / "data/raw/m1/nominal" / s)
        t, pos, yaw_odom, yaw_imu, out, df2 = run_filters(df, cfg, quick)
        rows, runs = metrics(t, pos, yaw_odom, yaw_imu, out, df2, cfg["inekf"]["warmup_s"])
        for r in rows:
            r["session"] = s; tab.append(r)
        figs[s] = (t, pos, out, runs)
        print(f"[e18 inekf] {s}: " + " | ".join(f"{r['filter']}: rmse {r['rmse_xy_m']:.2f} m, end {r['end_err_xy_m']:.2f} m, run-ratio med {r['run_ratio_median']:.3f}" for r in rows), flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_run"} for r in tab]); df.to_csv(res_dir / "e18_inekf_table.csv", index=False)
    (res_dir / "e18_inekf_per_run.json").write_text(json.dumps({r["session"] + "/" + r["filter"]: r["per_run"] for r in tab}, indent=1))
    # figure: XY paths on the primary session + per-run ratio per filter (all sessions)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    prim = cfg["primary"] if cfg["primary"] in figs else cfg["sessions"][0]; t, pos, out, runs = figs[prim]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]; ax.plot(pos[:, 0], pos[:, 1], "k-", lw=1.5, label="vendor odometry (reference)")
    for kind, c in zip(cfg["inekf"]["filters"], ("tab:blue", "tab:cyan", "tab:red", "tab:orange")):
        P = out[kind]["p"]; ax.plot(P[:, 0], P[:, 1], "-", color=c, lw=1, label=kind.replace("_", " "))
    for r in runs:
        ax.plot(pos[r, 0], pos[r, 1], color="tab:olive", lw=4, alpha=0.35)
    ax.plot(pos[0, 0], pos[0, 1], "ko", ms=6); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{prim}: estimated base paths (thick olive = straight-rolling runs)", fontsize=8)
    ax = axes[1]
    kinds = cfg["inekf"]["filters"]; xs = np.arange(len(cfg["sessions"])); w = 0.2
    for i, (kind, c) in enumerate(zip(kinds, ("tab:blue", "tab:cyan", "tab:red", "tab:orange"))):
        frac = [df[(df.session == s) & (df["filter"] == kind)]["path_len_recovered_frac"].values[0] for s in cfg["sessions"]]
        ax.bar(xs + (i - 1.5) * w, frac, w, color=c, label=kind.replace("_", " "))
    ax.axhline(1.0, color="k", ls=":", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels([s.replace("m1_walk_20260810_", "") for s in cfg["sessions"]], fontsize=8)
    ax.set_ylabel("estimated path length / odometry path length"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7); ax.set_ylim(0, 1.25)
    ax.set_title("path length recovered vs vendor odometry (pre-registered metric iii,\nheading-independent): rolling ≈ 1, fixed-foot ≈ 0", fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / "e18_inekf_real.png", dpi=110); plt.close(fig)
    lines = []
    for s in cfg["sessions"]:
        d = df[df.session == s]
        lines.append(f"[e18 inekf] {s}: " + "; ".join(f"{r['filter']} path-recovered {r['path_len_recovered_frac']:.2f} ({r['path_len_est_m']:.1f}/{r['path_len_odom_m']:.1f} m); per-run arclen-err median {r['run_arclen_err_median']:.3f}; rmse-vs-odom {r['rmse_xy_m']:.1f} m; run vector-ratio median {r['run_ratio_median']:.2f} (yaw-contaminated); yaw-end vs imu {r['yaw_end_vs_imu_deg']:+.0f}°" for _, r in d.iterrows()))
    (res_dir / "conclusions.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def stage_summary(cfg, res_dir, quick):
    root = Path(os.environ["GEOFDI_DATA_ROOT"]); rows = []
    for s in cfg["sessions"]:
        for L in ("L1", "L2"):
            f = root / "results/pipeline" / f"m1real_{s}_{L}" / "report.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text()); h = r["h0prime"]; w = h["window_test"]
            rows.append({"session": s, "L_s": r["registration"]["L_s"], "K": r["data_element"]["K"], "d": r["data_element"]["d"], "runs": r["registration"].get("segmentation", {}).get("n_runs"), "straight_s": r["registration"].get("segmentation", {}).get("masked_duration_s"),
                         "H0_p_paired": r["h0"]["whole_session_p"]["paired_energy"], "H0_p_energy": r["h0"]["whole_session_p"]["energy_distance"], "H0_window_rej": r["h0"]["window_rejection_rate"], "H0_n_windows": r["h0"]["n_windows"], "H0_eproc_max": r["h0"]["eprocess_max"], "H0_alarm": r["h0"]["eprocess_alarm_window"],
                         "lag1": r["block_correlation"]["lag1_autocorr_antisym_energy"], "H0p_diff_p": h["differenced_p_first_vs_second_half"], "nu0": h["nu0"], "nu0_boot_std": h["nu0_boot_std"], "K_cal_half": h["K_cal"],
                         "H0p_windows": w["n_windows"], "H0p_window_rej": w["window_rejection_rate"], "H0p_eproc_max": w["eprocess_max"], "H0p_alarm": w["eprocess_alarm_window"], "H0p_p_values": w["p_values"], "H0_p_values": r["h0"]["p_values"], "report_dir": str(f.parent)})
    df = pd.DataFrame(rows); df.to_csv(res_dir / "e18_pipeline_summary.csv", index=False); print(df.drop(columns=["H0p_p_values", "H0_p_values", "report_dir"]).to_string(index=False))
    # 3-session figure at L = 1 s: per-window p (H0 vs H0') and e-processes
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from geofdi.detect.evalue import eprocess
    d1 = df[df.L_s == 1.0]
    fig, axes = plt.subplots(2, len(d1), figsize=(4.2 * len(d1), 6), squeeze=False)
    for j, (_, r) in enumerate(d1.iterrows()):
        p0 = np.array(r["H0_p_values"]); p1 = np.array(r["H0p_p_values"]); n0 = len(p0); k1 = max(10, r["K"] // 3) / 10
        ax = axes[0, j]; ax.plot(np.arange(1, n0 + 1), p0, "o-", ms=4, label="H₀ naive flip"); 
        if len(p1): ax.plot(k1 + np.arange(1, len(p1) + 1), p1, "s-", ms=4, label="H₀′ vs calibration")
        ax.axhline(0.05, color="r", ls="--", lw=0.8); ax.set_ylim(-0.02, 1.02); ax.set_title(f"{r['session'].replace('m1_walk_20260810_', '')} — K={r['K']} blocks (L=1 s), lag1={r['lag1']:.2f}", fontsize=8); ax.set_xlabel("10-block window"); ax.set_ylabel("p"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax = axes[1, j]; E0, _ = eprocess(p0, 0.05); ax.semilogy(np.arange(1, n0 + 1), E0, "-o", ms=4, label="H₀ e-process")
        if len(p1): E1, _ = eprocess(p1, 0.05); ax.semilogy(k1 + np.arange(1, len(p1) + 1), E1, "-s", ms=4, label="H₀′ e-process")
        ax.axhline(20, color="r", ls="--", label="1/α"); ax.set_xlabel("window"); ax.set_ylabel("E_t"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title(f"H₀ p={r['H0_p_paired']:.3f}; H₀′ diff p={r['H0p_diff_p']:.3f}, ν₀={r['nu0']:.1f}±{r['nu0_boot_std']:.1f}", fontsize=8)
    fig.suptitle("M1 hardware 2026-08-10 — first real-robot R⁻ H₀ / H₀′ readouts (run_pipeline.sh, L = 1 s)", fontsize=9)
    fig.tight_layout(); fig.savefig(res_dir / "e18_real_h0_h0prime.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["inekf", "summary", "all"]); ap.add_argument("--quick", action="store_true"); ap.add_argument("--run-name", default=None)
    a = ap.parse_args(); cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    root = Path(os.environ["GEOFDI_DATA_ROOT"]); res_dir = root / "results" / cfg["experiment"] / (a.run_name or f"e18-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}"); res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.stage in ("inekf", "all"):
        stage_inekf(cfg, res_dir, a.quick)
    if a.stage in ("summary", "all"):
        stage_summary(cfg, res_dir, a.quick)
    print(f"[e18] results -> {res_dir}")


if __name__ == "__main__":
    main()
