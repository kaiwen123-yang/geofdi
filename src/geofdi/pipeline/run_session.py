#!/usr/bin/env python3
"""GeoFDI session pipeline (Sprint 7 Block W4) — the Day-0 first command behind scripts/run_pipeline.sh.

    python -m geofdi.pipeline.run_session <session_dir> --robot m1|go2 --mode rolling|trot [--residual off|analytic|delan_equiv]
                                          [--out DIR] [--L 1.0] [--window 10] [--alpha 0.05] [--M 512]

Steps: (1) ingest check (checksums.sha256 verified if present) -> (2) loader (name/index-based reorder, missing-channel
list, mapping unverified flag) -> (3) segmentation + registration (rolling: straight-command blocks of L s; trot:
kinematic phase estimator + cycle registration) -> (4) data element Z (NaN channels dropped and listed) -> (5) R- H0:
whole-session flip test, window p-values (QQ/KS, window rejection rate), e-process trajectory -> (6) H0': differenced
test first half vs second half + nu_0 calibration -> (7) three-channel readout if a residual model is available (M1:
equivariant DeLaN trained on rolling data; Go2: needs the contact wrench -> reported unavailable) -> report.md + figures.
Nothing is trained here; nothing needs a controller clock.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..detect.h0prime import calibrate
from ..detect.evalue import eprocess
from ..detect.permutation import hg_permutation_test, hg_permutation_tests
from ..groups.c2 import C2Rep


def _checksums_ok(session_dir: Path) -> dict:
    f = session_dir / "checksums.sha256"
    if not f.exists():
        return {"present": False, "verified": None, "mismatch": []}
    bad = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split(maxsplit=1); name = name.strip().lstrip("*").lstrip("./")
        p = session_dir / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(name)
    return {"present": True, "verified": len(bad) == 0, "mismatch": bad}


def _load(session_dir: Path, robot: str):
    if robot == "m1":
        from ..io.m1_sdk import load_m1_session
        return load_m1_session(session_dir)
    from ..io.go2_lowstate import load_go2_session
    return load_go2_session(session_dir)


def _register(df, manifest, robot: str, mode: str, L: float, N: int, warmup_s: float):
    from ..phase.registration import register_blocks, register_cycles, straight_mask
    z_names = [c["name"] for c in manifest["channels"] if c["in_Z"]]
    info = {}
    if mode == "rolling":
        mask = straight_mask(df, warmup_s=warmup_s) if "v_cmd" in df and np.isfinite(df["v_cmd"]).any() else (df["t"].to_numpy() >= warmup_s)
        info["segment_source"] = "v_cmd plateau" if "v_cmd" in df and np.isfinite(df["v_cmd"]).any() else "all rows after warm-up (no command signal)"
        Z, meta = register_blocks(df, z_names, L_s=L, N=N, mask=mask)
    else:
        from ..phase.estimator import estimate_phase
        contact = ["c_LF"] if ("c_LF" in df and np.isfinite(df["c_LF"]).any()) else None
        theta, pinfo = estimate_phase(df, joint="KFE", contact_cols=None)          # template origin (event origin reported)
        info["phase"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in pinfo.items()}
        d2 = df.copy(); d2["theta_hat"] = theta
        # drop_first covers the warm-up + the filter transient at the record start, drop_last the Hilbert-transform edge
        # transient at the record end (W3 finding: the last ~5 estimated cycles are distorted)
        Z, meta = register_cycles(d2, z_names, N=N, theta_col="theta_hat", drop_first=int(np.ceil(warmup_s / pinfo["period_s"])), drop_last=5)
        info["edge_cycles_dropped_last"] = 5
    return Z, meta, z_names, info


def run(session_dir: Path, robot: str, mode: str, residual: str = "off", out: Path | None = None, L: float = 1.0, N: int = 64,
        window: int = 10, alpha: float = 0.05, M: int = 512, warmup_s: float = 6.0, seed: int = 0):
    session_dir = Path(session_dir); out = Path(out) if out else session_dir.parent.parent.parent.parent.parent / "results" / "pipeline" / f"{robot}_{session_dir.name}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"session": str(session_dir), "robot": robot, "mode": mode, "residual": residual, "started": _dt.datetime.now().isoformat(timespec="seconds")}
    rep["ingest"] = _checksums_ok(session_dir)
    df, manifest, load_rep = _load(session_dir, robot); rep["loader"] = load_rep
    Z, meta, z_names, reg_info = _register(df, manifest, robot, mode, L, N, warmup_s); rep["registration"] = {**{k: v for k, v in meta.items() if k not in ("t_start", "row_start", "channels")}, **reg_info}
    # drop NaN channels (e.g. tau_cmd absent on hardware) from the data element
    keep = [i for i, n in enumerate(z_names) if np.isfinite(Z[:, i, :]).all()] if Z.shape[0] else list(range(len(z_names)))
    dropped = [n for i, n in enumerate(z_names) if i not in keep]
    man2 = dict(manifest); man2["channels"] = [c for c in manifest["channels"] if (c["name"] in [z_names[i] for i in keep]) or not c["in_Z"]]
    Z = Z[:, keep, :]; rep["data_element"] = {"K": int(Z.shape[0]), "d": int(Z.shape[1]), "N": int(Z.shape[2]), "dropped_nan_channels": dropped}
    if Z.shape[0] < 4:
        rep["error"] = "fewer than 4 data elements after registration"; (out / "report.json").write_text(json.dumps(rep, indent=1, default=str)); return rep
    repC2 = C2Rep(man2); K = Z.shape[0]
    # (5) R- H0
    r_all = hg_permutation_tests(Z, repC2, M=M, rng=np.random.default_rng(seed))
    nw = K // window; pw = np.empty(nw)
    for w in range(nw):
        pw[w], _ = hg_permutation_test(Z[w * window:(w + 1) * window], repC2, statistic="paired_energy", M=M, rng=np.random.default_rng([seed, w]))
    from scipy import stats
    E, alarm = eprocess(pw, alpha)
    rep["h0"] = {"whole_session_p": {k: float(v["p"]) for k, v in r_all.items()}, "window": window, "n_windows": int(nw),
                 "window_rejection_rate": float(np.mean(pw <= alpha)) if nw else float("nan"), "window_ks_p": float(stats.kstest(pw, "uniform").pvalue) if nw > 3 else float("nan"),
                 "eprocess_max": float(E.max()) if nw else float("nan"), "eprocess_alarm_window": alarm}
    # (6) H0': differenced test first half vs second half; nu_0 calibration numbers
    Kc = K // 2; Zc, Zm = Z[:Kc], Z[Kc:2 * Kc]
    pdiff, _ = hg_permutation_test(Zm - Zc, repC2, statistic="paired_energy", M=M, rng=np.random.default_rng([seed, 99]))
    cal = calibrate(Zc, repC2, n_boot=100, block_len=1, rng=np.random.default_rng(seed))
    rep["h0prime"] = {"differenced_p_first_vs_second_half": float(pdiff), "nu0": float(cal["nu0"]), "nu0_boot_std": float(cal["nu0_boot_std"]), "K_cal": int(Kc)}
    # (7) residual channel
    rep["residual_channel"] = {"requested": residual}
    if residual != "off":
        if robot == "m1" and residual == "delan_equiv":
            try:
                mdir = Path(os.environ["GEOFDI_DATA_ROOT"]) / "models" / "delan_m1" / "equiv_rolling_v1"
                from ..dynamics.delan_equiv import load_delan
                from ..phase.registration import register_blocks
                quad = load_delan(mdir, device="cpu"); LEGS = ("LF", "RF", "LH", "RH"); JOINTS = ("ABAD", "HIP", "KNEE", "WHEEL")
                from scipy.signal import savgol_filter
                res = np.zeros((len(df), 16)); dt = float(np.median(np.diff(df["t"].to_numpy())))
                for li, leg in enumerate(LEGS):
                    q = df[[f"q_{leg}_{j}" for j in JOINTS]].to_numpy().copy(); q[:, 3] = 0.0; dq = df[[f"dq_{leg}_{j}" for j in JOINTS]].to_numpy()
                    ddq = savgol_filter(dq, 7, 2, deriv=1, delta=dt, axis=0); a = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy(); tau = df[[f"tau_cmd_{leg}_{j}" for j in JOINTS]].to_numpy()
                    res[:, 4 * li:4 * li + 4] = tau - quad.predict(leg, q, dq, ddq, a)
                cols = [f"res_{l}_{j}" for l in LEGS for j in JOINTS]; dfr = pd.DataFrame(res, columns=cols); dfr["t"] = df["t"].to_numpy()
                from ..phase.registration import straight_mask
                mask = straight_mask(df, warmup_s=warmup_s)
                Zr, _ = register_blocks(dfr, cols, L_s=L, N=N, mask=mask)
                from ..sim.telemetry_m1 import JOINT_SIGN, MIRROR_LEG
                ch = [{"name": f"res_{l}_{j}", "group": "res", "leg": l, "joint": j, "kind": "scalar-signed", "partner": f"res_{MIRROR_LEG[l]}_{j}", "sign": JOINT_SIGN[j], "in_Z": True} for l in LEGS for j in JOINTS]
                repR = C2Rep({"channels": ch, "gait_group": {"delta_theta": 0.0}})
                ok = np.isfinite(Zr).all()
                if ok and Zr.shape[0] >= 4:
                    rr = hg_permutation_tests(Zr, repR, M=M, rng=np.random.default_rng([seed, 7]))
                    from ..residuals.mirror_pairs import isotypic_split
                    Zp, Zm_ = isotypic_split(Zr, repR)
                    rep["residual_channel"].update({"model": str(mdir), "K": int(Zr.shape[0]), "Rminus_on_residual_p": {k: float(v["p"]) for k, v in rr.items()},
                                                    "Rplus_energy_Pi_plus_mean": float((Zp ** 2).mean()), "Rminus_energy_Pi_minus_mean": float((Zm_ ** 2).mean()),
                                                    "per_leg_residual_rms": {l: float(np.sqrt((Zr[:, 4 * li:4 * li + 4, :] ** 2).mean())) for li, l in enumerate(LEGS)}})
                else:
                    rep["residual_channel"]["status"] = "residual has NaN (tau_cmd missing in the session?)"
            except Exception as e:      # noqa: BLE001
                rep["residual_channel"]["status"] = f"unavailable: {e}"
        else:
            rep["residual_channel"]["status"] = ("Go2 residual channels need the contact wrench (analytic observer) or the DeLaN target tau + J^T f: "
                                                 "not derivable from LowState alone — hardware path TODO" if robot == "go2" else f"{residual} not implemented for {robot}")
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    if nw:
        q = (np.arange(nw) + 0.5) / nw; axes[0].plot(q, np.sort(pw), "o-", ms=3); axes[0].plot([0, 1], [0, 1], "k:")
        axes[0].set_xlabel("uniform quantile"); axes[0].set_ylabel(f"window p (paired energy, {window} elements)"); axes[0].set_title(f"R⁻ window p-values (n={nw}, KS p={rep['h0']['window_ks_p']:.2f})", fontsize=9)
        axes[1].semilogy(np.arange(1, nw + 1), E, "-o", ms=3); axes[1].axhline(1 / alpha, color="r", ls="--", label=f"1/α = {1/alpha:.0f}"); axes[1].set_xlabel("window"); axes[1].set_ylabel("e-process E_t"); axes[1].legend(fontsize=8); axes[1].set_title("R⁻ e-process trajectory", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "rminus_qq_eprocess.png", dpi=130); plt.close(fig)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    md = [f"# GeoFDI session report — {session_dir.name} ({robot}, {mode})", "", f"- ingest checksums: {rep['ingest']}", f"- loader: {load_rep['n_rows']} rows, {load_rep['duration_s']:.1f} s, rate ≈ {load_rep['rate_hz_estimate']:.1f} Hz, mapping unverified: {load_rep['mapping_unverified']}",
          f"- missing channels: {load_rep['missing'] or 'none'}", f"- registration: {rep['registration']}", f"- data element: K = {Z.shape[0]}, d = {Z.shape[1]}, N = {Z.shape[2]}; dropped NaN channels: {dropped or 'none'}", "",
          f"## R⁻ under H₀", f"- whole-session flip test p: {rep['h0']['whole_session_p']}", f"- window rejection rate ({window}-element windows, α={alpha}): {rep['h0']['window_rejection_rate']:.3f} over {nw} windows; KS uniformity p = {rep['h0']['window_ks_p']:.3f}",
          f"- e-process max {rep['h0']['eprocess_max']:.2f}, alarm window {rep['h0']['eprocess_alarm_window']}", "", f"## H₀′ (asymmetry change)", f"- differenced test first vs second half p = {pdiff:.3f}; ν₀ = {cal['nu0']:.4f} ± {cal['nu0_boot_std']:.4f} (K_cal = {Kc})", "",
          f"## residual channel", f"- {rep['residual_channel']}", "", f"figures: rminus_qq_eprocess.png"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(f"[pipeline] report -> {out}/report.md")
    print(f"[pipeline] K={Z.shape[0]} d={Z.shape[1]} | R- whole-session p {rep['h0']['whole_session_p']} | window rejection {rep['h0']['window_rejection_rate']:.3f} (KS {rep['h0']['window_ks_p']:.2f}) | e-process max {rep['h0']['eprocess_max']:.2f} alarm {alarm} | H0' differenced p {pdiff:.3f} | residual: {rep['residual_channel'].get('status', rep['residual_channel'].get('Rminus_on_residual_p', 'off'))}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("session_dir"); ap.add_argument("--robot", choices=["m1", "go2"], required=True)
    ap.add_argument("--mode", choices=["rolling", "trot"], required=True); ap.add_argument("--residual", choices=["off", "analytic", "delan_equiv"], default="off")
    ap.add_argument("--out", default=None); ap.add_argument("--L", type=float, default=1.0); ap.add_argument("--N", type=int, default=64); ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--M", type=int, default=512); ap.add_argument("--warmup", type=float, default=6.0)
    a = ap.parse_args()
    run(Path(a.session_dir), a.robot, a.mode, a.residual, a.out, a.L, a.N, a.window, a.alpha, a.M, a.warmup)


if __name__ == "__main__":
    main()
