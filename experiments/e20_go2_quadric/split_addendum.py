#!/usr/bin/env python3
"""M0.2 — xb4 / nmb3 split experiment (Sprint 10). Pre-registration: docs/protocol/e20_split_preregistration.md.

Cuts each session's straight rows into two halves (xb4: spatially, by median RTK easting; nmb3: temporally) and runs the
H0' machinery on each half alone and on the concatenation, plus a within-longest-run halving control. Also reports the
symmetric (Pi^+) magnitude readouts, which a surface change should move even though R- should not.

    python experiments/e20_go2_quadric/split_addendum.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run as R                                                   # build_element/h0_h0prime/CFG/DATA/session_dir
from geofdi.groups.c2 import C2Rep
from geofdi.io.go2_quadric import LEGS, load_go2_quadric_session, straight_mask_go2
from geofdi.phase.estimator import estimate_phase, gait_signal_from_columns
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import isotypic_split
from scipy.signal import butter, filtfilt

CFG, DATA = R.CFG, R.DATA


def _register_rows(df, rows, man, det, min_cycles=12):
    """Phase-register a contiguous set of straight rows (one or more runs) -> element."""
    chans = [c["name"] for c in man["channels"] if c["in_Z"]]
    Zs = []
    for r in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
        if len(r) < 400:
            continue
        sub = df.iloc[r[0]:r[-1] + 1].reset_index(drop=True)
        if float(sub["t"].iloc[-1] - sub["t"].iloc[0]) < 8.0:
            continue
        try:
            theta, _ = estimate_phase(sub, contact_cols=[f"c_{l}" for l in LEGS], signal=gait_signal_from_columns(sub))
        except Exception:
            continue
        d2 = sub.copy(); d2["theta_hat"] = theta
        Z, _ = register_cycles(d2, chans, N=det["N"], theta_col="theta_hat", drop_first=det["drop_first"], drop_last=det["drop_last"])
        if Z.shape[0] and np.isfinite(Z).all():
            Zs.append(Z)
    Z = np.concatenate(Zs) if Zs else np.empty((0, len(chans), det["N"]))
    return Z, chans


def _magnitude_readouts(df, rows):
    """Symmetric (mirror-invariant) readouts a surface change should move: foot-force level and spread, and the
    >20 Hz IMU texture. These are Pi^+-type quantities, so R- is blind to them by construction."""
    t = df["t"].to_numpy(); fs = 1.0 / np.median(np.diff(t))
    b, a = butter(2, 20 / (fs / 2), btype="high")
    hi = np.abs(filtfilt(b, a, np.nan_to_num(df["imu_a_z"].to_numpy())))
    ff = np.nanmean([df[f"foot_force_{l}"].to_numpy() for l in LEGS], axis=0)
    return {"foot_force_mean": float(np.nanmean(ff[rows])), "foot_force_std": float(np.nanstd(ff[rows])),
            "imu_hf_std": float(np.nanstd(hi[rows])), "speed_mean": float(np.nanmean(np.hypot(
                np.nan_to_num(df["base_vx"].to_numpy()), np.nan_to_num(df["base_vy"].to_numpy()))[rows]))}


def main():
    det = CFG["detect"]
    res = DATA / "results" / "e20_go2_quadric" / f"split-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}"
    res.mkdir(parents=True, exist_ok=True)
    rows_out, mag_out = [], []
    for name, how in (("xb4", "spatial"), ("nmb3", "temporal")):
        df, man, rep = load_go2_quadric_session(R.session_dir(name))
        mask, minfo = straight_mask_go2(df)
        idx = np.where(mask)[0]
        if how == "spatial":
            e = df["rtk_e"].to_numpy()
            med = np.nanmedian(e[idx])
            selA = idx[e[idx] <= med]; selB = idx[e[idx] > med]
            labA, labB = f"half A (RTK easting <= {med:.1f} m)", f"half B (RTK easting > {med:.1f} m)"
        else:
            h = len(idx) // 2
            selA, selB = idx[:h], idx[h:]
            labA, labB = "first half (time)", "second half (time)"
        parts = {labA: selA, labB: selB, "concatenated (both halves)": idx}
        # within-longest-run control: halve the longest single run
        runs = [r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)]
        best = max(runs, key=len)
        parts["longest run, 1st half"] = best[:len(best) // 2]
        parts["longest run, 2nd half"] = best[len(best) // 2:]
        parts["longest run, whole"] = best
        for lab, sel in parts.items():
            Z, chans = _register_rows(df, sel, man, det)
            if Z.shape[0] < 12:
                rows_out.append({"session": name, "split": how, "part": lab, "K": int(Z.shape[0]), "note": "too few cycles"})
                print(f"[split] {name} / {lab}: K={Z.shape[0]} — too few cycles", flush=True); continue
            crep = C2Rep(man)
            r = R.h0_h0prime(Z, crep, det)
            rows_out.append({"session": name, "split": how, "part": lab, "K": int(Z.shape[0]),
                             "H0_p": r["H0_whole_p"]["paired_energy"], "H0_window_rej": r["H0_window_rej"],
                             "H0p_window_rej": r["H0p_window_rej"], "H0p_alarm": r["H0p_alarm"],
                             "H0p_eproc_max": r["H0p_eproc_max"], "nu0": r["nu0"], "nu0_boot_std": r["nu0_boot_std"],
                             "n_windows_h0p": r["H0p_n_windows"]})
            print(f"[split] {name} / {lab}: K={Z.shape[0]} | H0 p={r['H0_whole_p']['paired_energy']:.4f} | "
                  f"H0' win-rej {r['H0p_window_rej']:.2f} alarm {r['H0p_alarm']} (n={r['H0p_n_windows']}) | nu0 {r['nu0']:.2f}", flush=True)
        for lab, sel in ((labA, selA), (labB, selB)):
            mag_out.append({"session": name, "part": lab, **_magnitude_readouts(df, sel)})
    T = pd.DataFrame(rows_out); T.to_csv(res / "e20_split_table.csv", index=False)
    M = pd.DataFrame(mag_out); M.to_csv(res / "e20_split_magnitudes.csv", index=False)
    print()
    print(T.to_string(index=False))
    print()
    print(M.round(3).to_string(index=False))
    # verdict — four outcomes, distinguished honestly (and the window-count confound is stated, not hidden)
    lines = []
    for name in ("xb4", "nmb3"):
        d = T[(T.session == name) & T.H0p_window_rej.notna()]
        halves = d[~d.part.str.startswith(("concatenated", "longest"))]
        conc = d[d.part.str.startswith("concatenated")]
        lr = d[d.part.str.startswith("longest run,") & d.part.str.contains("half")]
        n_alarm = int(halves.H0p_alarm.notna().sum()); n_half = len(halves)
        conc_alarm = int(conc.H0p_alarm.notna().sum())
        mg = M[M.session == name]
        dff = (mg.foot_force_mean.max() - mg.foot_force_mean.min()) / max(mg.foot_force_mean.min(), 1e-9)
        dhf = (mg.imu_hf_std.max() - mg.imu_hf_std.min()) / max(mg.imu_hf_std.min(), 1e-9)
        sym = f"symmetric readouts differ by {100*dff:.0f} % (foot force) / {100*dhf:.0f} % (IMU texture) across the split"
        if n_alarm == 0 and conc_alarm > 0:
            v = ("CROSS-SEGMENT condition change: every half is in band, only the concatenation alarms -> P-A corrected to "
                 "'between-segment', and the per-run rule reads as per-HOMOGENEOUS-SEGMENT")
        elif n_alarm == n_half:
            v = "WITHIN-TRAVERSE anomaly everywhere: every half still alarms -> the per-segment rule is not sufficient"
        elif n_alarm > 0:
            clean = halves[halves.H0p_alarm.isna()].part.tolist(); noisy = halves[halves.H0p_alarm.notna()].part.tolist()
            v = (f"LOCALISED non-stationarity: {noisy} still alarms while {clean} is in band "
                 f"(nu0 {halves.nu0.min():.1f} vs {halves.nu0.max():.1f}) -> the session is not homogeneous, and the "
                 f"inhomogeneity sits in one identified part, not at the join")
        else:
            v = "no alarm anywhere at this resolution — inconclusive"
        lr_note = (f" [longest-run halves: {int(lr.H0p_alarm.notna().sum())}/{len(lr)} alarm, but with only "
                   f"{lr.n_windows_h0p.min():.0f}-{lr.n_windows_h0p.max():.0f} monitoring windows each the e-process has "
                   f"little chance to accumulate — NOT evidence of homogeneity]") if len(lr) else ""
        lines.append(f"[M0.2] {name}: {v}. {sym}.{lr_note}")
    mag_line = "[M0.2] symmetric readouts across the split: " + "; ".join(
        f"{r['session']} {r['part'].split('(')[0].strip()} force {r['foot_force_mean']:.1f}+-{r['foot_force_std']:.1f}, HF {r['imu_hf_std']:.2f}, v {r['speed_mean']:.2f}"
        for _, r in M.iterrows())
    (res / "conclusions.txt").write_text("\n".join(lines + [mag_line]) + "\n")
    print(); [print(l) for l in lines]; print(mag_line)
    print(f"[M0.2] results -> {res}")


if __name__ == "__main__":
    main()
