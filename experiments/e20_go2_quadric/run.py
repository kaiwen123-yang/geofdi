#!/usr/bin/env python3
"""e20 — GeoFDI on the QUADRIC-GINS Go2 corpus (Sprint 9 Block R). Pre-registration: docs/protocol/go2_real_preregistration.md.

  --stage r1  per-session R- H0 / H0' (11 sessions) + summary figure
  --stage r2  cross-period reproducibility: nu0 per session, Jan-calibrated H0' applied to Mar and back
  --stage r5  natural-anomaly hunt: sequential H0' e-process per session + 173247-style diagnosis of every rejection
  --stage r6  foot-IMU (LH) as an independent phase / touch-down check
  --stage r4  estimator value: contact-aided InEKF (foot_position_body + foot-force contacts) vs the RTK reference,
              with and without pi_i gating; RTK quality-gated per P-RTK
    python experiments/e20_go2_quadric/run.py --stage r1|r2|r4|r5|r6|all
Outputs -> $GEOFDI_DATA_ROOT/results/e20_go2_quadric/<run>/
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
from scipy import stats

from geofdi.detect.evalue import eprocess
from geofdi.detect.h0prime import calibrate, h0prime_test, nu
from geofdi.detect.permutation import hg_permutation_test, hg_permutation_tests, pooled_scale
from geofdi.groups.c2 import C2Rep
from geofdi.io.go2_quadric import LEGS, load_go2_quadric_session, straight_mask_go2
from geofdi.phase.estimator import estimate_phase, gait_signal_from_columns
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import isotypic_split

DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
CFG = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())


def session_dir(name: str) -> Path:
    for day, names in CFG["sessions"].items():
        if name in names:
            return DATA / "data/raw/go2" / day / name
    raise KeyError(name)


def all_sessions():
    return [(day, n) for day, names in CFG["sessions"].items() for n in names]


def site_of(name: str) -> str:
    return CFG["sites"]["".join(c for c in name if not c.isdigit())]


# ------------------------------------------------------------------ element construction
def build_element(name: str, det=None, min_run_s=None):
    """Straight-run cycles of one session -> (Z, C2Rep, info). Every straight run is phase-registered on its own (the
    phase clock must not be extrapolated across a gap) and the cycles are concatenated in time order."""
    det = det or CFG["detect"]; min_run_s = min_run_s or CFG["registration"]["min_run_s"]
    df, man, rep = load_go2_quadric_session(session_dir(name))
    mask, minfo = straight_mask_go2(df, min_run_s=min_run_s)
    idx = np.where(mask)[0]
    chans = [c["name"] for c in man["channels"] if c["in_Z"]]
    Zs, run_meta = [], []
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            sub = df.iloc[r[0]:r[-1] + 1].reset_index(drop=True)
            if float(sub["t"].iloc[-1] - sub["t"].iloc[0]) < min_run_s:
                continue
            try:
                sig = gait_signal_from_columns(sub)
                theta, pinfo = estimate_phase(sub, contact_cols=[f"c_{l}" for l in LEGS], signal=sig)
            except Exception as e:                       # a run too short/noisy for a phase fit is skipped, and counted
                run_meta.append({"t0": float(sub["t"].iloc[0]), "error": str(e)}); continue
            d2 = sub.copy(); d2["theta_hat"] = theta
            Zr, _ = register_cycles(d2, chans, N=det["N"], theta_col="theta_hat",
                                    drop_first=det["drop_first"], drop_last=det["drop_last"])
            if Zr.shape[0] and np.isfinite(Zr).all():
                Zs.append(Zr)
                run_meta.append({"t0": round(float(sub["t"].iloc[0]), 1), "t1": round(float(sub["t"].iloc[-1]), 1),
                                 "K": int(Zr.shape[0]), "period_s": round(float(pinfo["period_s"]), 4)})
    Z = np.concatenate(Zs) if Zs else np.empty((0, len(chans), det["N"]))
    info = {"session": name, "site": site_of(name), "day": rep.get("day", ""), "K": int(Z.shape[0]), "d": int(Z.shape[1]),
            "n_runs_used": len(Zs), "straight": {k: v for k, v in minfo.items() if k != "runs"},
            "rate_hz": rep["rate_hz"], "duration_s": rep["duration_s"], "fix_ok": rep["fixposition"].get("fix_ok_fraction"),
            "foot_imu": rep["foot_imu_present"], "runs": run_meta,
            "period_s_median": float(np.median([m["period_s"] for m in run_meta if "period_s" in m])) if any("period_s" in m for m in run_meta) else float("nan")}
    return Z, C2Rep(man), info, chans


def h0_h0prime(Z, rep, det, seed=0):
    """Naive H0 (whole-element + per-window flip test, e-process) and H0' (per-window vs the calibration third)."""
    K = Z.shape[0]; win, M, alpha = det["window"], det["M"], det["alpha"]
    out = {"K": int(K)}
    if K < 12:
        out["error"] = "fewer than 12 cycles"; return out
    r_all = hg_permutation_tests(Z, rep, M=M, rng=np.random.default_rng(seed))
    out["H0_whole_p"] = {k: float(v["p"]) for k, v in r_all.items()}
    nw = K // win
    pw = np.array([hg_permutation_test(Z[w * win:(w + 1) * win], rep, statistic="paired_energy", M=M,
                                       rng=np.random.default_rng([seed, w]))[0] for w in range(nw)])
    E, alarm = eprocess(pw, alpha)
    out.update(H0_window_rej=float(np.mean(pw <= alpha)), H0_n_windows=int(nw), H0_eproc_max=float(E.max()) if nw else np.nan,
               H0_alarm=alarm, H0_ks_p=float(stats.kstest(pw, "uniform").pvalue) if nw > 3 else float("nan"), H0_pw=[float(x) for x in pw])
    Kcal = max(win, K // 3); nwp = (K - Kcal) // win
    pwp = np.array([h0prime_test(Z[:Kcal], Z[Kcal + w * win:Kcal + (w + 1) * win], rep, M=M,
                                 rng=np.random.default_rng([seed, 500 + w]))["p"] for w in range(nwp)])
    Ep, alarmp = eprocess(pwp, alpha) if nwp else (np.array([]), None)
    cal = calibrate(Z[:Kcal], rep, n_boot=100, rng=np.random.default_rng(seed))
    out.update(K_cal=int(Kcal), H0p_n_windows=int(nwp), H0p_window_rej=float(np.mean(pwp <= alpha)) if nwp else np.nan,
               H0p_eproc_max=float(Ep.max()) if nwp else np.nan, H0p_alarm=alarmp,
               H0p_ks_p=float(stats.kstest(pwp, "uniform").pvalue) if nwp > 3 else float("nan"),
               nu0=float(cal["nu0"]), nu0_boot_std=float(cal["nu0_boot_std"]), H0p_pw=[float(x) for x in pwp])
    return out


def per_leg_ranking(Z, rep, chans):
    """Which leg carries the anti-symmetric energy (P-LH). Pi^- energy per leg on standardized channels; R- pairs legs,
    so the ranking is over mirror PAIRS (front LF/RF vs hind LH/RH) plus the per-leg share of the pair's channels."""
    _, Zm = isotypic_split(Z, rep)
    sc = Z.transpose(1, 0, 2).reshape(Z.shape[1], -1).std(axis=1) + 1e-12
    e = (Zm / sc[None, :, None]) ** 2
    per_leg = {}
    for l in LEGS:
        ix = [i for i, c in enumerate(chans) if c.endswith(f"_{l}") or f"_{l}_" in c]
        per_leg[l] = float(e[:, ix, :].mean())
    pair = {"front(LF/RF)": 0.5 * (per_leg["LF"] + per_leg["RF"]), "hind(LH/RH)": 0.5 * (per_leg["LH"] + per_leg["RH"])}
    return {"per_leg_antisym_energy": per_leg, "per_pair": pair,
            "top_leg": max(per_leg, key=per_leg.get), "top_pair": max(pair, key=pair.get)}


# ------------------------------------------------------------------ R1
def stage_r1(res_dir):
    det = CFG["detect"]; rows = []
    for day, name in all_sessions():
        Z, rep, info, chans = build_element(name)
        r = h0_h0prime(Z, rep, det)
        rank = per_leg_ranking(Z, rep, chans) if Z.shape[0] >= 12 else {}
        row = {**{k: v for k, v in info.items() if k not in ("runs", "straight")}, **r, **rank,
               "straight_s": info["straight"]["masked_s"], "straight_runs": info["straight"]["n_runs"], "day": day}
        rows.append(row)
        print(f"[r1] {name} ({info['site']}, {day}): K={r.get('K')} straight {info['straight']['masked_s']:.0f}s | "
              f"H0 p={r.get('H0_whole_p',{}).get('paired_energy',float('nan')):.4f} win-rej {r.get('H0_window_rej',float('nan')):.2f} alarm {r.get('H0_alarm')} | "
              f"H0' win-rej {r.get('H0p_window_rej',float('nan')):.2f} eproc {r.get('H0p_eproc_max',float('nan')):.1f} alarm {r.get('H0p_alarm')} | "
              f"nu0 {r.get('nu0',float('nan')):.2f} | top {rank.get('top_leg','-')}/{rank.get('top_pair','-')}", flush=True)
    T = pd.DataFrame(rows)
    T.drop(columns=[c for c in ("H0_pw", "H0p_pw", "per_leg_antisym_energy", "per_pair") if c in T]).to_csv(res_dir / "e20_r1_sessions.csv", index=False)
    (res_dir / "e20_r1_full.json").write_text(json.dumps(rows, indent=1, default=str))
    _plot_r1(res_dir, T, rows)
    ok_H0 = int((T["H0_whole_p"].apply(lambda d: d.get("paired_energy", 1.0)) <= det["alpha"]).sum())
    n_alarm = int(T["H0p_alarm"].notna().sum())
    line = (f"[e20 R1] {len(T)} Go2 sessions: naive H0 rejects on {ok_H0}/{len(T)} (prediction 1: >= 8/11); "
            f"H0' sequential e-process alarms on {n_alarm}/{len(T)} (prediction 2: none on a healthy session); "
            f"H0' window-reject rate median {T['H0p_window_rej'].median():.2f}; nu0 range {T['nu0'].min():.2f}-{T['nu0'].max():.2f}; "
            f"per-pair anti-symmetric energy top = " + ", ".join(f"{s}:{r['top_pair']}" for s, r in zip(T.session, rows)))
    (res_dir / "conclusions.txt").open("a").write(line + "\n"); print(line)
    return T


def _plot_r1(res_dir, T, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    alpha = CFG["detect"]["alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.arange(len(T)); col = {"A": "tab:orange", "B": "tab:blue", "C": "tab:green"}
    cs = [col[s] for s in T.site]
    ax = axes[0]
    ax.bar(xs - 0.2, T["H0_window_rej"], 0.4, color="tab:red", label="naive H₀ window-reject")
    ax.bar(xs + 0.2, T["H0p_window_rej"], 0.4, color="tab:blue", label="H₀′ window-reject")
    ax.axhline(alpha, color="k", ls=":", lw=0.8, label="α"); ax.axhspan(0, 0.12, color="green", alpha=0.10)
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R⁻ window-reject rate"); ax.set_title("Go2 (own corpus, >1 yr in service, 5 kg payload):\nnaive H₀ sees the natural asymmetry, H₀′ stays in band", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    ax = axes[1]
    ax.bar(xs, T["nu0"], color=cs); ax.errorbar(xs, T["nu0"], yerr=T["nu0_boot_std"], fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ν₀ (H₀′ calibration asymmetry level)")
    ax.set_title("ν₀ per session — colour = site (A orange / B blue / C green)\nJan (nmb,xb) vs Mar (by): the two-month reproducibility check", fontsize=8.5); ax.grid(alpha=.3, axis="y")
    ax = axes[2]
    ep = T["H0p_eproc_max"].to_numpy()
    ax.semilogy(xs, np.maximum(ep, 1e-3), "o", color="tab:blue", label="H₀′ e-process max")
    ax.semilogy(xs, np.maximum(T["H0_eproc_max"].to_numpy(), 1e-3), "s", color="tab:red", label="H₀ e-process max")
    ax.axhline(1 / alpha, color="r", ls="--", lw=0.9, label="alarm 1/α")
    ax.set_xticks(xs); ax.set_xticklabels(T.session, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("max e-process"); ax.set_title("sequential monitor: H₀ alarms (real asymmetry),\nH₀′ does not (asymmetry is stationary within a session)", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("e20 R1 — GeoFDI R⁻ on 11 own Unitree Go2 sessions (QUADRIC-GINS corpus, 3 sites, 2 dates two months apart)", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / "e20_r1_go2_h0_h0prime.png", dpi=115); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="r1", choices=["r1", "r2", "r4", "r5", "r6", "all"])
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    res_dir = DATA / "results" / CFG["experiment"] / (a.run_id or f"e20-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}")
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(CFG, sort_keys=False))
    stages = ["r1", "r2", "r5", "r6", "r4"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "r1":
            stage_r1(res_dir)
        else:
            from _stages import STAGES
            STAGES[s](res_dir)
    print(f"[e20] results -> {res_dir}")


if __name__ == "__main__":
    main()
