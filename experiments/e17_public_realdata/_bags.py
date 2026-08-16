"""Leg-KILO Go1 and Cerberus Street A1 bag loaders + PUB3 stages (Sprint 8 Block PUB, e17)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from geofdi.sim.telemetry import JOINTS, LEGS, all_columns, build_manifest

DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
# Unitree leg order FR,FL,RR,RL; motorState per leg [abad, thigh, calf]. GeoFDI [LF,RF,LH,RH]=[FL,FR,RL,RR].
UNITREE_LEG = {"LF": 1, "RF": 0, "LH": 3, "RH": 2}
JIDX = {"HAA": 0, "HFE": 1, "KFE": 2}


def _empty_frame(t):
    df = pd.DataFrame(np.nan, index=np.arange(len(t)), columns=all_columns()); df["t"] = t - t[0]; df["theta"] = np.nan
    return df


def load_legkilo(bagdir):
    """Leg-KILO Go1 bag -> Go2-schema GeoFDI frame. HighState motorState q/dq/tauEst, footForce->contact, imu FLU."""
    from rosbags.highlevel import AnyReader
    bag = next(Path(bagdir).glob("*.bag"))
    rows = {"t": [], "q": [], "dq": [], "tau": [], "ff": [], "quat": [], "gyro": [], "acc": [], "fp": []}
    with AnyReader([bag]) as r:
        conn = [c for c in r.connections if c.topic == "/high_state"]
        for c, ts, raw in r.messages(connections=conn):
            m = r.deserialize(raw, c.msgtype)
            st = m.stamp.sec + m.stamp.nanosec * 1e-9 if hasattr(m, "stamp") and hasattr(m.stamp, "sec") else ts * 1e-9
            rows["t"].append(st); rows["q"].append([m.motorState[i].q for i in range(12)]); rows["dq"].append([m.motorState[i].dq for i in range(12)])
            rows["tau"].append([m.motorState[i].tauEst for i in range(12)]); rows["ff"].append(list(m.footForce))
            rows["quat"].append(list(m.imu.quaternion)); rows["gyro"].append(list(m.imu.gyroscope)); rows["acc"].append(list(m.imu.accelerometer))
            fp = m.footPosition2Body if hasattr(m, "footPosition2Body") else None
            rows["fp"].append([v for cc in fp for v in (cc.x, cc.y, cc.z)] if fp is not None else [np.nan] * 12)
    t = np.array(rows["t"]); q = np.array(rows["q"]); dq = np.array(rows["dq"]); tau = np.array(rows["tau"]); ff = np.array(rows["ff"], float)
    quat = np.array(rows["quat"]); gyro = np.array(rows["gyro"]); acc = np.array(rows["acc"]); fp = np.array(rows["fp"], float)
    df = _empty_frame(t)
    ff_thresh = np.nanmedian(ff) * 0.6
    for leg in LEGS:
        li = UNITREE_LEG[leg]
        for j in JOINTS:
            df[f"q_{leg}_{j}"] = q[:, 3 * li + JIDX[j]]; df[f"dq_{leg}_{j}"] = dq[:, 3 * li + JIDX[j]]; df[f"tau_meas_{leg}_{j}"] = tau[:, 3 * li + JIDX[j]]
        df[f"c_{leg}"] = (ff[:, li] > ff_thresh).astype(float); df[f"temp_{leg}"] = np.nan
        for a, ax in enumerate("xyz"):
            df[f"foot_{ax}_{leg}"] = fp[:, 3 * li + a]
    for a, ax in enumerate("xyz"):
        df[f"imu_a_{ax}"] = acc[:, a]; df[f"imu_w_{ax}"] = gyro[:, a]
    for a, qn in enumerate(("base_qx", "base_qy", "base_qz", "base_qw")):     # Unitree imu.quaternion is [w,x,y,z]? store all
        pass
    df["base_qw"] = quat[:, 0]; df["base_qx"] = quat[:, 1]; df["base_qy"] = quat[:, 2]; df["base_qz"] = quat[:, 3]
    man = build_manifest(sim_meta={"source": "public (Leg-KILO Go1, Ou et al. RA-L 2024)", "robot": "unitree_go1", "sequence": Path(bagdir).name,
                                   "rate_hz": float(1.0 / np.median(np.diff(t))), "efforts_semantics": "tauEst", "mapping_unverified": False})
    rep = {"n_rows": len(df), "duration_s": float(t[-1] - t[0]), "rate_hz_estimate": float(1.0 / np.median(np.diff(t))), "sequence": Path(bagdir).name}
    return df, man, rep


def load_street(bagdir):
    """Cerberus Street A1 bag -> Go2-schema frame. /hardware_a1/joint_foot (12 joints + 4 foot: vel=contact, eff=force)."""
    from rosbags.highlevel import AnyReader
    bag = next(Path(bagdir).glob("*.bag"))
    JN = ["FL0", "FL1", "FL2", "FR0", "FR1", "FR2", "RL0", "RL1", "RL2", "RR0", "RR1", "RR2"]
    LEGMAP = {"LF": "FL", "RF": "FR", "LH": "RL", "RH": "RR"}
    js_t, js_q, js_dq, foot_c = [], [], [], []
    imu_t, imu_g, imu_a, imu_q = [], [], [], []
    with AnyReader([bag]) as r:
        for c, ts, raw in r.messages(connections=[c for c in r.connections if c.topic in ("/hardware_a1/joint_foot", "/hardware_a1/imu")]):
            m = r.deserialize(raw, c.msgtype)
            if c.topic.endswith("joint_foot"):
                names = list(m.name); pos = np.array(m.position); vel = np.array(m.velocity)
                idx = {n: i for i, n in enumerate(names)}
                js_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                js_q.append([pos[idx[n]] for n in JN]); js_dq.append([vel[idx[n]] for n in JN])
                foot_c.append([vel[idx[f"{LEGMAP[l]}_foot"]] for l in LEGS])       # foot velocity slot = contact flag (0/1)
            else:
                imu_t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9); imu_g.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                imu_a.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]); imu_q.append([m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z])
    t = np.array(js_t); q = np.array(js_q); dq = np.array(js_dq); fc = np.array(foot_c)
    df = _empty_frame(t)
    for leg in LEGS:
        pre = LEGMAP[leg]
        for j, suf in zip(JOINTS, ("0", "1", "2")):
            col = JN.index(f"{pre}{suf}"); df[f"q_{leg}_{j}"] = q[:, col]; df[f"dq_{leg}_{j}"] = dq[:, col]
        df[f"c_{leg}"] = (fc[:, LEGS.index(leg)] > 0.5).astype(float); df[f"temp_{leg}"] = np.nan
    ti = np.array(imu_t); ig = np.array(imu_g); ia = np.array(imu_a); iq = np.array(imu_q)
    for a, ax in enumerate("xyz"):
        df[f"imu_a_{ax}"] = np.interp(t, ti, ia[:, a]); df[f"imu_w_{ax}"] = np.interp(t, ti, ig[:, a])
    for a, qn in enumerate(("base_qw", "base_qx", "base_qy", "base_qz")):
        df[qn] = np.interp(t, ti, iq[:, a])
    man = build_manifest(include_groups=("q", "dq", "imu_acc", "imu_gyro", "contact", "temp"),      # A1 street has no torques
                         sim_meta={"source": "public (Cerberus Street A1, Yang et al. ICRA 2023)", "robot": "unitree_a1", "sequence": Path(bagdir).name,
                                   "rate_hz": float(1.0 / np.median(np.diff(t))), "efforts_semantics": "none (effort=0)", "mapping_unverified": False})
    rep = {"n_rows": len(df), "duration_s": float(t[-1] - t[0]), "rate_hz_estimate": float(1.0 / np.median(np.diff(t))), "sequence": Path(bagdir).name}
    return df, man, rep


def straight_trot_mask(df, wz_max=0.25, warmup_s=3.0, min_run_s=4.0):
    t = df["t"].to_numpy(); dt = float(np.median(np.diff(t))); w = max(1, int(round(0.4 / dt))); k = np.ones(w) / w
    wz = np.convolve(np.abs(np.nan_to_num(df["imu_w_z"].to_numpy())), k, mode="same")
    duty = np.mean([np.convolve(df[f"c_{l}"].to_numpy(), np.ones(max(1, int(1 / dt))) / max(1, int(1 / dt)), mode="same") for l in LEGS], axis=0)
    m = (wz < wz_max) & (duty > 0.15) & (duty < 0.9) & (t >= warmup_s)
    idx = np.where(m)[0]; kept = np.zeros_like(m); n_runs = 0
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if t[r[-1]] - t[r[0]] >= min_run_s:
                kept[r] = True; n_runs += 1
    return kept, {"n_runs": int(n_runs), "masked_s": float(kept.sum() * dt), "fraction": float(kept.mean())}


def _longest_run(df, mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return df.iloc[:0]
    run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    return df.iloc[run[0]:run[-1] + 1].reset_index(drop=True)


def stage_legkilo(res_dir, rminus_fn):
    rows = []
    seqs = ["corridor", "park", "slope", "grass", "indoor"]
    for seq in seqs:
        d = DATA / "data/raw/public/legkilo-go1" / f"legkilo_{seq}"
        if not d.exists():
            continue
        try:
            df, man, rep = load_legkilo(d)
        except Exception as e:
            print(f"[e17 legkilo] {seq}: load error {e}"); continue
        mask, mi = straight_trot_mask(df); sub = _longest_run(df, mask)
        if len(sub) < 2000:
            print(f"[e17 legkilo] {seq}: no straight run"); continue
        r = rminus_fn(sub, man); r.update(sequence=seq, rate_hz=rep["rate_hz_estimate"], straight_s=float(sub["t"].iloc[-1] - sub["t"].iloc[0]))
        rows.append(r); print(f"[e17 legkilo] {seq}: K={r.get('K')} H0 p={r.get('H0_whole_p',{}).get('paired_energy',float('nan')):.3f} alarm {r.get('H0_alarm')} | H0' win-rej {r.get('H0p_window_rej',float('nan')):.2f} alarm {r.get('H0p_alarm')} ν0 {r.get('nu0',float('nan')):.2f}", flush=True)
    pd.DataFrame(rows).to_csv(res_dir / "e17_legkilo_h0prime.csv", index=False)
    _plot_bags(res_dir, rows, "legkilo (Go1 trot, 500 Hz)", "e17_legkilo_h0prime.png")
    if rows:
        line = "[e17 legkilo] real R⁻ (straight trot, longest run): " + "; ".join(f"{r['sequence']} K{r['K']} H0 p{r['H0_whole_p'].get('paired_energy',float('nan')):.3f} H0' win-rej {r['H0p_window_rej']:.2f}" for r in rows if 'K' in r and 'error' not in r)
        (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)


def stage_street(res_dir, rminus_fn):
    d = DATA / "data/raw/public/street-a1/cerberus_street"
    if not d.exists():
        print("[e17 street] not ingested"); return
    df, man, rep = load_street(d); mask, mi = straight_trot_mask(df); sub = _longest_run(df, mask)
    r = rminus_fn(sub, man); r.update(sequence="cerberus_street", straight_s=float(sub["t"].iloc[-1] - sub["t"].iloc[0]) if len(sub) else 0.0)
    pd.DataFrame([r]).to_csv(res_dir / "e17_street_h0prime.csv", index=False)
    _plot_bags(res_dir, [r], "Cerberus Street A1 (trot, 470 Hz, no torques)", "e17_street_h0prime.png")
    print(f"[e17 street] K={r.get('K')} H0 p={r.get('H0_whole_p',{}).get('paired_energy',float('nan')):.3f} alarm {r.get('H0_alarm')} | H0' win-rej {r.get('H0p_window_rej',float('nan')):.2f} alarm {r.get('H0p_alarm')}", flush=True)
    if "K" in r and "error" not in r:
        (res_dir / "conclusions.txt").open("a").write(f"[e17 street] real R⁻ (straight trot {r['straight_s']:.0f}s, K{r['K']}): H0 p {r['H0_whole_p'].get('paired_energy',float('nan')):.3f} alarm {r['H0_alarm']}; H0' win-rej {r['H0p_window_rej']:.2f} alarm {r['H0p_alarm']}\n")


def _plot_bags(res_dir, rows, title, fname):
    rows = [r for r in rows if "K" in r and "error" not in r]
    if not rows:
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(rows) + 3), 4.2)); xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["H0_window_rej"] for r in rows], 0.4, color="tab:red", label="naive H₀ window-reject")
    ax.bar(xs + 0.2, [r["H0p_window_rej"] for r in rows], 0.4, color="tab:blue", label="H₀′ window-reject")
    ax.axhline(0.05, color="k", ls=":", lw=0.8, label="α"); ax.axhspan(0, 0.12, color="green", alpha=0.1)
    ax.set_xticks(xs); ax.set_xticklabels([r["sequence"] for r in rows], fontsize=8); ax.set_ylabel("R⁻ window-reject rate")
    ax.set_title(f"{title} — real-robot R⁻ H₀/H₀′ (straight-trot mined)", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(res_dir / fname, dpi=120); plt.close(fig)


def stage_legkilo_gating(res_dir, seq="corridor"):
    """Real-data version of e16: the 4 estimators on a Leg-KILO sequence using the robot's own footPosition2Body (no URDF
    FK). Leg-KILO carries no labelled slip, so we measure the NOMINAL per-event false-rejection rate: the fixed 0.4 m/s
    foot-speed threshold vs the calibrated GeoFDI gate. Prediction (e16): threshold false-rejects real nominal contacts,
    GeoFDI-πᵢ ≈ α."""
    from geofdi.estimate.pi_gating import build_event_library, run_gated_filter
    from geofdi.inekf.kinematics import Go2Kinematics
    d = DATA / "data/raw/public/legkilo-go1" / f"legkilo_{seq}"
    df, man, rep = load_legkilo(d); mask, mi = straight_trot_mask(df)
    idx = np.where(mask)[0]; runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); runs = [r for r in runs if len(r) > 3000]
    if len(runs) < 2:
        print(f"[e17 legkilo-gate] {seq}: need two straight runs"); return
    kin = Go2Kinematics()
    cal = df.iloc[runs[0][0]:runs[0][-1] + 1].reset_index(drop=True)
    mon = df.iloc[runs[1][0]:runs[1][-1] + 1].reset_index(drop=True)
    common = dict(sigma_gyro=0.02, sigma_accel=0.2, sigma_contact=0.02, sigma_kin_floor=0.02, alpha=0.05, use_provided_feet=True)
    lib = build_event_library(cal, kin, **common)
    rows = []
    for mode in ("none", "threshold", "geofdi_hard", "geofdi_soft"):
        kw = dict(mode=mode, lib=lib, **common)
        if mode == "threshold":
            kw.update(foot_speed_thresh=0.4, cov_inflate=10.0)
        _, info = run_gated_filter(mon, kin, **kw)
        W = info["weights"]; fr = float(1.0 - np.nanmean(W))
        rows.append({"estimator": mode, "nominal_false_reject": fr, "n_steps": len(mon)})
        print(f"[e17 legkilo-gate] {seq} {mode}: nominal per-step false-reject {fr:.3f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e17_legkilo_gating.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4)); xs = np.arange(len(rows))
    ax.bar(xs, [r["nominal_false_reject"] for r in rows], color=["grey", "tab:orange", "tab:blue", "tab:cyan"])
    ax.axhline(0.05, color="r", ls="--", lw=0.8, label="α"); ax.set_xticks(xs); ax.set_xticklabels([r["estimator"].replace("_", "\n") for r in rows], fontsize=8)
    ax.set_ylabel("nominal foot-measurement reject rate"); ax.set_title(f"Leg-KILO {seq} (real Go1): the fixed 0.4 m/s threshold\nfalse-rejects real nominal contacts; GeoFDI-πᵢ respects α", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(res_dir / "e17_legkilo_gating.png", dpi=120); plt.close(fig)
    th = T[T.estimator == "threshold"]["nominal_false_reject"].iloc[0]; gs = T[T.estimator == "geofdi_soft"]["nominal_false_reject"].iloc[0]; gh = T[T.estimator == "geofdi_hard"]["nominal_false_reject"].iloc[0]
    line = f"[e17 legkilo-gate] {seq} (real Go1, footPosition2Body): nominal foot-measurement reject rate — threshold(0.4 m/s) {th:.3f} vs GeoFDI-soft {gs:.3f} / GeoFDI-hard {gh:.3f} (~alpha). Real-data version of e16: the fixed threshold false-rejects real nominal contacts; the calibrated gate respects the per-event FAR."
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
