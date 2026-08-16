"""e20 stages R2 (cross-period), R4 (estimator), R5 (anomaly hunt + diagnosis), R6 (foot-IMU phase check)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geofdi.detect.evalue import eprocess
from geofdi.detect.h0prime import calibrate, h0prime_test
from geofdi.detect.permutation import hg_permutation_test
from geofdi.io.go2_quadric import LEGS, load_go2_quadric_session, straight_mask_go2
from geofdi.residuals.mirror_pairs import isotypic_split

import run as R                                        # build_element, h0_h0prime, CFG, DATA, session_dir, all_sessions
CFG, DATA = R.CFG, R.DATA
PAIRS = (("LF", "RF"), ("LH", "RH"))


# ----------------------------------------------------------------------------- side localisation (P-LH)
def signed_pair_asymmetry(Z, chans):
    """R- ranks mirror PAIRS, not sides: for a mirror pair (c, c') the anti-symmetric component obeys
    (Pi^- Z)[c'] = -sign * (Pi^- Z)[c], so the two partners carry IDENTICAL anti-symmetric energy and no side can be
    read off an energy. The side is carried by the SIGN of the mean difference (the Sprint-7 e09 fix). Here, per mirror
    pair and per channel family, we report mean_k,n ( Z[c] - sign * Z[c'] ) / scale -- positive means the LEFT member is
    the larger one -- plus the pair's anti-symmetric energy share (which IS a legitimate R- ranking)."""
    out = {}
    for L, Rg in PAIRS:
        for fam in ("foot_force", "foot_pos", "foot_vel"):
            cl = [i for i, c in enumerate(chans) if c.startswith(fam) and (c == f"{fam}_{L}" or c.startswith(f"{fam}_{L}_"))]
            cr = [i for i, c in enumerate(chans) if c.startswith(fam) and (c == f"{fam}_{Rg}" or c.startswith(f"{fam}_{Rg}_"))]
            if not cl or len(cl) != len(cr):
                continue
            # channel-wise sign under the mirror: foot_force +1 ; foot_pos/vel (x,y,z) -> (+1,-1,+1)
            sg = np.array([1.0] if fam == "foot_force" else [1.0, -1.0, 1.0])
            a = Z[:, cl, :]; b = Z[:, cr, :] * sg[None, :, None]
            d = (a - b)
            sc = np.sqrt(((a - a.mean()) ** 2).mean()) + 1e-12
            out[f"{L}{Rg}_{fam}_signed"] = float(d.mean() / sc)
            out[f"{L}{Rg}_{fam}_absmean"] = float(np.abs(d.mean(axis=(0, 2))).mean() / sc)
    return out


def pair_energy_share(Z, rep, chans):
    """Anti-symmetric energy share per mirror pair (a legitimate R- ranking: pairs, not sides), unnormalised channels."""
    _, Zm = isotypic_split(Z, rep)
    tot = float((Zm ** 2).sum()) + 1e-12
    out = {}
    for L, Rg in PAIRS:
        ix = [i for i, c in enumerate(chans) if f"_{L}" in c or f"_{Rg}" in c]
        out[f"{L}/{Rg}"] = float((Zm[:, ix, :] ** 2).sum() / tot)
    return out


# ----------------------------------------------------------------------------- R5
def stage_r5(res_dir):
    """Natural-anomaly hunt + 173247-style diagnosis. The key control: H0' computed on POOLED straight runs vs H0'
    computed WITHIN one long run. If pooled alarms while within-run does not, the asymmetry is stationary inside a run
    and varies BETWEEN runs (a condition change), which is a nuisance, not a fault."""
    det = CFG["detect"]; rows = []
    for day, name in R.all_sessions():
        Z, rep, info, chans = R.build_element(name)
        pooled = R.h0_h0prime(Z, rep, det)
        # --- within-run control: the single longest run of the session, registered on its own
        best = max((m for m in info["runs"] if "K" in m), key=lambda m: m["K"], default=None)
        within = {}
        if best is not None:
            Zr, rep2, info2, _ = R.build_element(name, min_run_s=max(8.0, best["t1"] - best["t0"] - 0.01))
            within = R.h0_h0prime(Zr, rep2, det) if Zr.shape[0] >= 12 else {"error": "too few cycles"}
        # --- nu trajectory shape (drift vs boundary jump)
        K = Z.shape[0]; win = det["window"]; nb = K // win
        nus = []
        for w in range(nb):
            c = calibrate(Z[w * win:(w + 1) * win], rep, n_boot=20, rng=np.random.default_rng([7, w]))
            nus.append(float(c["nu0"]))
        nus = np.array(nus)
        shape = "flat"
        if len(nus) >= 6:
            x = np.arange(len(nus)); sl = np.polyfit(x, nus, 1)[0]
            drift = abs(sl) * len(nus) / (np.std(nus) + 1e-9)
            jump = float(np.max(np.abs(np.diff(nus))) / (np.std(nus) + 1e-9))
            shape = "drift" if drift > 1.5 and drift > jump else ("boundary-jump" if jump > 2.5 else "flat/noisy")
        else:
            drift = jump = float("nan")
        row = {"session": name, "site": info["site"], "day": day, "K_pooled": int(K), "n_runs_used": info["n_runs_used"],
               "pooled_H0p_window_rej": pooled.get("H0p_window_rej"), "pooled_H0p_alarm": pooled.get("H0p_alarm"),
               "pooled_H0p_eproc_max": pooled.get("H0p_eproc_max"),
               "within_run_K": within.get("K"), "within_run_H0p_window_rej": within.get("H0p_window_rej"),
               "within_run_H0p_alarm": within.get("H0p_alarm"), "within_run_H0p_eproc_max": within.get("H0p_eproc_max"),
               "nu_shape": shape, "nu_drift_score": float(drift), "nu_jump_score": float(jump),
               "nu_traj": [round(x, 3) for x in nus], **pair_energy_share(Z, rep, chans), **signed_pair_asymmetry(Z, chans)}
        rows.append(row)
        print(f"[r5] {name} ({info['site']}): pooled H0' rej {row['pooled_H0p_window_rej']:.2f} alarm {row['pooled_H0p_alarm']} | "
              f"within-run K={row['within_run_K']} rej {row['within_run_H0p_window_rej'] if row['within_run_H0p_window_rej'] is not None else float('nan')} alarm {row['within_run_H0p_alarm']} | "
              f"nu shape {shape} | pair share LF/RF {row['LF/RF']:.2f} LH/RH {row['LH/RH']:.2f} | "
              f"signed LHRH force {row.get('LHRH_foot_force_signed', float('nan')):+.3f}", flush=True)
    T = pd.DataFrame(rows)
    T.drop(columns=["nu_traj"]).to_csv(res_dir / "e20_r5_anomaly_hunt.csv", index=False)
    (res_dir / "e20_r5_full.json").write_text(json.dumps(rows, indent=1, default=str))
    _plot_r5(res_dir, T)
    npool = int(T.pooled_H0p_alarm.notna().sum()); nwithin = int(T.within_run_H0p_alarm.notna().sum())
    line = (f"[e20 R5] H0' alarms: pooled-over-runs {npool}/{len(T)} vs within-single-run {nwithin}/{len(T)} — "
            f"the asymmetry is {'stationary inside a run and varies BETWEEN runs (condition change, nuisance)' if nwithin < npool else 'non-stationary even inside a run'}. "
            f"nu-shape classes: {T.nu_shape.value_counts().to_dict()}. "
            f"Pair energy share (median): LF/RF {T['LF/RF'].median():.2f}, LH/RH {T['LH/RH'].median():.2f}. "
            f"Signed LH-RH foot-force asymmetry (median over Jan sessions with the LH board): "
            f"{T[T.day=='2026-01-05']['LHRH_foot_force_signed'].median():+.3f} vs Mar (no board) "
            f"{T[T.day=='2026-03-06']['LHRH_foot_force_signed'].median():+.3f}")
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
    return T


def _plot_r5(res_dir, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    xs = np.arange(len(T))
    ax = axes[0]
    ax.bar(xs - 0.2, T.pooled_H0p_window_rej, 0.4, color="tab:red", label="H₀′ pooled over runs")
    ax.bar(xs + 0.2, T.within_run_H0p_window_rej.astype(float), 0.4, color="tab:green", label="H₀′ within one run")
    ax.axhline(0.05, color="k", ls=":", lw=0.8); ax.axhspan(0, 0.12, color="green", alpha=0.10)
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7); ax.set_ylabel("H₀′ window-reject rate")
    ax.set_title("R5 diagnosis: pooling separate straight runs breaks H₀′,\nwithin a single run it holds ⇒ between-run condition change", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    ax.bar(xs - 0.2, T["LF/RF"], 0.4, color="tab:purple", label="front pair LF/RF")
    ax.bar(xs + 0.2, T["LH/RH"], 0.4, color="tab:brown", label="hind pair LH/RH")
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("share of Π⁻ energy"); ax.set_title("which mirror PAIR carries the asymmetry\n(R⁻ ranks pairs, never sides)", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    for c, lab, col in (("LHRH_foot_force_signed", "LH−RH foot force", "tab:brown"), ("LFRF_foot_force_signed", "LF−RF foot force", "tab:purple")):
        if c in T:
            ax.plot(xs, T[c], "o-", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(-0.5, 7.5, color="tab:blue", alpha=0.07); ax.text(3.5, ax.get_ylim()[1] * 0.9, "Jan (LH IMU board fitted)", fontsize=7, ha="center")
    ax.axvspan(7.5, len(T) - 0.5, color="tab:green", alpha=0.07); ax.text(9, ax.get_ylim()[1] * 0.9, "Mar (no board)", fontsize=7, ha="center")
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("signed left−right mean (standardised)"); ax.set_title("P-LH: SIDE needs the signed mean, not the energy", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("e20 R5 — natural-asymmetry hunt and diagnosis on the own Go2 corpus", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e20_r5_diagnosis.png", dpi=115); plt.close(fig)


# ----------------------------------------------------------------------------- R2
def stage_r2(res_dir):
    """Cross-period reproducibility: nu0 per session and the H0' test with the calibration window taken from a DIFFERENT
    session (same site where possible; January vs March both ways)."""
    det = CFG["detect"]; elems = {}
    for day, name in R.all_sessions():
        Z, rep, info, chans = R.build_element(name)
        elems[name] = (Z, rep, info)
    rows = []
    for name, (Z, rep, info) in elems.items():
        c = calibrate(Z[:max(det["window"], Z.shape[0] // 3)], rep, n_boot=100, rng=np.random.default_rng(0))
        rows.append({"session": name, "site": info["site"], "day": "2026-01-05" if name[0] in "nx" else "2026-03-06",
                     "K": int(Z.shape[0]), "nu0": float(c["nu0"]), "nu0_boot_std": float(c["nu0_boot_std"])})
    N = pd.DataFrame(rows); N.to_csv(res_dir / "e20_r2_nu0.csv", index=False)
    # cross-session H0': calibration from session i, monitoring windows from session j
    cross = []
    for ci, (Zc, repc, ic) in elems.items():
        for mj, (Zm, repm, im) in elems.items():
            if ci == mj:
                continue
            win = det["window"]; Kc = min(Zc.shape[0], 60)
            nwp = min(Zm.shape[0] // win, 12)
            if Kc < win or nwp < 2:
                continue
            p = np.array([h0prime_test(Zc[:Kc], Zm[w * win:(w + 1) * win], repc, M=det["M"],
                                       rng=np.random.default_rng([11, w]))["p"] for w in range(nwp)])
            cross.append({"cal": ci, "cal_site": ic["site"], "mon": mj, "mon_site": im["site"],
                          "same_site": ic["site"] == im["site"],
                          "cal_day": "Jan" if ci[0] in "nx" else "Mar", "mon_day": "Jan" if mj[0] in "nx" else "Mar",
                          "n_windows": int(nwp), "window_rej": float(np.mean(p <= det["alpha"]))})
    C = pd.DataFrame(cross); C.to_csv(res_dir / "e20_r2_cross.csv", index=False)
    _plot_r2(res_dir, N, C)
    jan = N[N.day == "2026-01-05"]; mar = N[N.day == "2026-03-06"]
    within_site = N.groupby("site").nu0.std().to_dict()
    xj = C[(C.cal_day == "Jan") & (C.mon_day == "Mar")]["window_rej"].mean()
    xm = C[(C.cal_day == "Mar") & (C.mon_day == "Jan")]["window_rej"].mean()
    same = C[C.same_site]["window_rej"].mean(); diff = C[~C.same_site]["window_rej"].mean()
    line = (f"[e20 R2] nu0: Jan {jan.nu0.min():.1f}-{jan.nu0.max():.1f} (n={len(jan)}), Mar {mar.nu0.min():.1f}-{mar.nu0.max():.1f} (n={len(mar)}); "
            f"within-site sd {({k: round(v,2) for k,v in within_site.items()})}; between-date ratio "
            f"mean(Mar)/mean(Jan) = {mar.nu0.mean()/jan.nu0.mean():.2f}. "
            f"Cross-session H0' window-reject: Jan-cal->Mar-mon {xj:.2f}, Mar-cal->Jan-mon {xm:.2f}, "
            f"same-site {same:.2f} vs different-site {diff:.2f} (alpha={det['alpha']}).")
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
    return N, C


def _plot_r2(res_dir, N, C):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    col = {"A": "tab:orange", "B": "tab:blue", "C": "tab:green"}
    for s, g in N.groupby("site"):
        ax.errorbar(np.arange(len(g)), g.nu0, yerr=g.nu0_boot_std, fmt="o-", color=col[s], capsize=3, label=f"site {s} ({'Jan' if s!='C' else 'Mar'})")
    ax.set_xlabel("session index within site"); ax.set_ylabel("ν₀"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("ν₀ by site and date — the two-month reproducibility check", fontsize=9)
    ax = axes[1]
    piv = C.pivot_table(index="cal", columns="mon", values="window_rej")
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xlabel("monitored session"); ax.set_ylabel("calibration session")
    ax.set_title("cross-session H₀′ window-reject rate\n(diagonal blocks = same site)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("e20 R2 — cross-period / cross-site reproducibility of the asymmetry level", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e20_r2_cross_period.png", dpi=115); plt.close(fig)


# ----------------------------------------------------------------------------- R6
def stage_r6(res_dir):
    """Foot-IMU (LH leg, Jan sessions only) as an INDEPENDENT check of the phase estimator: the board's vertical-axis
    acceleration spikes at touch-down; compare those impact times with the estimator's LH stance-onset phase."""
    from geofdi.io.go2_quadric import load_foot_imu
    from geofdi.phase.estimator import estimate_phase, gait_signal_from_columns
    from scipy.signal import butter, filtfilt, find_peaks
    rows = []
    for day, name in R.all_sessions():
        sdir = R.session_dir(name); fcsv = sorted((sdir / "foot_imu").glob("*.csv")) if (sdir / "foot_imu").exists() else []
        if not fcsv:
            rows.append({"session": name, "day": day, "foot_imu": False}); continue
        df, man, rep = load_go2_quadric_session(sdir)
        fi = load_foot_imu(fcsv[0])
        mask, minfo = straight_mask_go2(df)
        idx = np.where(mask)[0]
        if not len(idx):
            rows.append({"session": name, "day": day, "foot_imu": True, "error": "no straight run"}); continue
        run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
        sub = df.iloc[run[0]:run[-1] + 1].reset_index(drop=True)
        t0a, t1a = df["t_abs"].to_numpy()[run[0]], df["t_abs"].to_numpy()[run[-1]]
        sig = gait_signal_from_columns(sub)
        theta, pinfo = estimate_phase(sub, contact_cols=[f"c_{l}" for l in LEGS], signal=sig)
        per = float(pinfo["period_s"])
        # foot-IMU impacts inside the same absolute-time window
        f = fi[(fi.t_abs >= t0a) & (fi.t_abs <= t1a)].reset_index(drop=True)
        if len(f) < 100:
            rows.append({"session": name, "day": day, "foot_imu": True, "error": f"only {len(f)} foot-IMU rows overlap"}); continue
        fs_f = 1.0 / np.median(np.diff(f.t_abs.to_numpy()))
        a = np.linalg.norm(f[["a0x", "a0y", "a0z"]].to_numpy(), axis=1)
        b, aa = butter(2, min(0.45, 8.0 / (fs_f / 2)), btype="high"); hi = np.abs(filtfilt(b, aa, a - np.nanmean(a)))
        pk, _ = find_peaks(hi, height=np.nanquantile(hi, 0.9), distance=int(0.6 * per * fs_f))
        t_imp = f.t_abs.to_numpy()[pk]
        # estimator LH stance-onset times from the contact flag of the SAME run
        c = sub["c_LH"].to_numpy() > 0.5
        onset_idx = np.where((~c[:-1]) & c[1:])[0] + 1
        t_on = sub["t_abs"].to_numpy()[onset_idx] if "t_abs" in sub else sub["t"].to_numpy()[onset_idx] + t0a
        # phase error: for each impact, distance to the nearest stance onset, in fractions of a period
        if len(t_imp) > 5 and len(t_on) > 5:
            d = np.abs(t_imp[:, None] - t_on[None, :]).min(axis=1)
            d = np.minimum(d, per - d)                                 # circular
            err = float(np.median(d) / per)
        else:
            err = float("nan")
        rows.append({"session": name, "day": day, "foot_imu": True, "run_s": round(float(t1a - t0a), 1), "period_s": round(per, 4),
                     "foot_imu_rate_hz": round(float(fs_f), 1), "n_impacts": int(len(t_imp)), "n_stance_onsets": int(len(t_on)),
                     "median_phase_error_frac": round(err, 4) if np.isfinite(err) else None})
        print(f"[r6] {name}: period {per:.3f}s, foot-IMU {fs_f:.0f} Hz, {len(t_imp)} impacts vs {len(t_on)} LH stance onsets, "
              f"median phase error {err*100:.1f} % of a period", flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e20_r6_footimu_phase.csv", index=False)
    ok = T[T.median_phase_error_frac.notna()] if "median_phase_error_frac" in T else T.iloc[:0]
    line = (f"[e20 R6] foot-IMU (LH leg, {int(T.foot_imu.sum())}/{len(T)} sessions carry it) as an independent phase check: "
            + (f"median touch-down phase error {100*ok.median_phase_error_frac.median():.1f} % of a gait period over {len(ok)} sessions "
               f"(pre-registered validation threshold 10 %). " if len(ok) else "no session produced a usable comparison. ")
            + "Single leg ⇒ used only as a validation source, never as a mirror channel.")
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
    return T


STAGES = {"r2": stage_r2, "r5": stage_r5, "r6": stage_r6}


# ----------------------------------------------------------------------------- R4
def stage_r4(res_dir):
    """Estimator value on real outdoor data. The corpus has NO joint stream, but the high-level API publishes
    `foot_position_body`, which is exactly the body-frame contact point a contact-aided InEKF needs — so the experiment
    the sprint plan made conditional on a joint stream IS possible (pi_gating `use_provided_feet`).
    Reference: the Fixposition ENU track, **quality-gated per P-RTK** (fix OK only); the gated fraction is reported."""
    from geofdi.estimate.pi_gating import build_event_library, run_gated_filter
    from geofdi.inekf.kinematics import Go2Kinematics
    ie = CFG["inekf"]; kin = Go2Kinematics(); rows = []
    common = dict(sigma_gyro=ie["sigma_gyro"], sigma_accel=ie["sigma_accel"], sigma_contact=ie["sigma_contact"],
                  sigma_kin_floor=ie["sigma_kin_floor"], alpha=ie["alpha"], use_provided_feet=True)
    for day, name in R.all_sessions():
        df, man, rep = load_go2_quadric_session(R.session_dir(name))
        mask, minfo = straight_mask_go2(df)
        idx = np.where(mask)[0]
        if not len(idx):
            rows.append({"session": name, "error": "no straight run"}); continue
        runs = [r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(r) > 2000]
        if len(runs) < 2:
            rows.append({"session": name, "error": f"only {len(runs)} runs > 2000 samples"}); continue
        cal = df.iloc[runs[0][0]:runs[0][-1] + 1].reset_index(drop=True)
        lib = build_event_library(cal, kin, **common)
        # monitoring run: the longest remaining run with a usable RTK reference
        best, best_ok = None, -1.0
        for r in runs[1:]:
            sub = df.iloc[r[0]:r[-1] + 1]
            ok = float(np.mean(sub["rtk_fix_ok"].to_numpy(bool) & np.isfinite(sub["rtk_e"].to_numpy())))
            if ok * len(sub) > best_ok:
                best, best_ok = sub.reset_index(drop=True), ok * len(sub)
        if best is None or len(best) < 1000:
            rows.append({"session": name, "error": "no monitoring run"}); continue
        ok = best["rtk_fix_ok"].to_numpy(bool) & np.isfinite(best["rtk_e"].to_numpy())
        gated_frac = float(np.mean(~ok))
        for mode in ("none", "threshold", "geofdi_hard", "geofdi_soft"):
            kw = dict(mode=mode, lib=lib, **common)
            if mode == "threshold":
                kw.update(foot_speed_thresh=ie["foot_speed_thresh"], cov_inflate=ie["cov_inflate"])
            est, info = run_gated_filter(best, kin, **kw)
            # RTK-referenced error on the fix-OK samples, after aligning the start (the filter has no global frame)
            e = best["rtk_e"].to_numpy(); n = best["rtk_n"].to_numpy()
            if ok.sum() < 200:
                continue
            i0 = int(np.argmax(ok))
            ref = np.stack([e - e[i0], n - n[i0]], 1)
            p = est[:, :2] - est[i0, :2]
            # rotate the estimate into the reference frame by the best-fit yaw over the gated samples (Umeyama, rotation only)
            A = p[ok]; B = ref[ok]
            H = A.T @ B; U, _, Vt = np.linalg.svd(H); Rm = Vt.T @ U.T
            if np.linalg.det(Rm) < 0:
                Vt[-1] *= -1; Rm = Vt.T @ U.T
            err = np.linalg.norm((Rm @ A.T).T - B, axis=1)
            path = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))
            W = info["weights"]
            rows.append({"session": name, "site": R.site_of(name), "day": day, "mode": mode, "n": int(len(best)),
                         "rtk_gated_out_frac": round(gated_frac, 3), "ref_path_m": round(path, 1),
                         "ate_rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 3), "ate_final_m": round(float(err[-1]), 3),
                         "ate_rel_pct": round(100 * float(np.sqrt(np.mean(err ** 2))) / max(path, 1e-6), 2),
                         "mean_gate_weight": round(float(np.nanmean(W)), 4), "reject_frac": round(float(1 - np.nanmean(W)), 4)})
        r0 = [x for x in rows if x.get("session") == name]
        if r0:
            print("[r4] " + name + " (" + str(r0[0].get("site")) + f", RTK dropped {gated_frac:.2f}): " +
                  "; ".join(f"{x['mode']} ATE {x['ate_rmse_m']:.2f} m ({x['ate_rel_pct']:.1f}% of {x['ref_path_m']:.0f} m), reject {x['reject_frac']:.3f}" for x in r0 if "mode" in x), flush=True)
    T = pd.DataFrame(rows); T.to_csv(res_dir / "e20_r4_estimator.csv", index=False)
    if "mode" in T and T["mode"].notna().any():
        g = T[T["mode"].notna()].groupby("mode")[["ate_rmse_m", "ate_rel_pct", "reject_frac"]].median()
        _plot_r4(res_dir, T)
        line = ("[e20 R4] contact-aided InEKF on the real Go2 corpus using foot_position_body (no joint stream needed), "
                f"RTK-referenced on the fix-OK samples (mean dropped fraction {T['rtk_gated_out_frac'].mean():.2f}; "
                f"site B {T[T.site=='B']['rtk_gated_out_frac'].mean():.2f} vs site C {T[T.site=='C']['rtk_gated_out_frac'].mean():.2f} — P-RTK). "
                + "; ".join(f"{m}: median ATE {r.ate_rmse_m:.2f} m ({r.ate_rel_pct:.1f} % of path), nominal reject {r.reject_frac:.3f}" for m, r in g.iterrows()))
    else:
        line = "[e20 R4] no session produced a usable RTK-referenced comparison — reported as skipped."
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
    return T


def _plot_r4(res_dir, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = T[T["mode"].notna()]
    modes = ["none", "threshold", "geofdi_hard", "geofdi_soft"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for i, m in enumerate(modes):
        d = T[T["mode"] == m]
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).normal(0, 0.06, len(d)), d.ate_rel_pct, s=22, alpha=0.75)
        ax.hlines(d.ate_rel_pct.median(), i - 0.25, i + 0.25, color="k", lw=2)
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=8)
    ax.set_ylabel("ATE RMSE / reference path length [%]"); ax.set_title("R4: contact-aided InEKF vs the RTK reference\n(one point per session, bar = median)", fontsize=9)
    ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    for i, m in enumerate(modes):
        d = T[T["mode"] == m]
        ax.bar(i, d.reject_frac.median(), 0.6, color=["grey", "tab:orange", "tab:blue", "tab:cyan"][i])
    ax.axhline(CFG["detect"]["alpha"], color="r", ls="--", lw=0.9, label="α")
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=8)
    ax.set_ylabel("nominal foot-measurement reject rate"); ax.set_title("gate activity on real outdoor trot", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.suptitle("e20 R4 — estimator value on the own Go2 corpus (RTK quality-gated per P-RTK)", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e20_r4_estimator.png", dpi=115); plt.close(fig)


STAGES["r4"] = stage_r4
