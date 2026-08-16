#!/usr/bin/env python3
"""B6 — Mini Cheetah flying trot in ROLLING-STYLE TIME BLOCKS instead of cycle registration (Sprint 9).

Sprint 8 found that on the Mini Cheetah's 0.25 s flying trot the cycle-level H0' is elevated: the gait is non-stationary
cycle-to-cycle and the kinematic phase estimator is stressed. B6 asks whether replacing phase registration by
fixed-duration time blocks (as the wheeled M1 rolling mode does) brings H0' back into band.

The subtlety that decides the construction: for a trot the symmetry is sigma_* = (mirror, shift by T/2), so applying the
pure mirror to a *time-aligned* block is NOT a symmetry — a block element must be phase-free. We therefore use the
exactly-equivariant phase-free summary of each block: per channel the **first two moments** (mean, std) over the block.
Under the trot symmetry the time-marginal law of a full block is invariant under the pure mirror rho_g, so
    mean -> channel sign * mean(partner),    std -> std(partner)   (a magnitude: sign +1)
holds exactly, with no phase alignment anywhere. Blocks are cut on the same straight-trot mask as Sprint 8.

    python experiments/e17_public_realdata/block_mode.py [--L 2.0]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from geofdi.detect.evalue import eprocess
from geofdi.detect.h0prime import calibrate, h0prime_test
from geofdi.detect.permutation import hg_permutation_test, hg_permutation_tests
from geofdi.groups.c2 import C2Rep
from geofdi.io.minicheetah import load_minicheetah, stance_trot_mask
from geofdi.sim.telemetry import z_channel_names

DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
TERRAINS = ["asphalt_road", "concrete_difficult_slippery", "forest", "grass", "middle_pebble", "rock_road", "sidewalk", "small_pebble"]


def block_moment_element(df, manifest, mask, L_s: float):
    """Fixed-duration blocks -> phase-free element: per channel [mean, std] over the block. Returns (Z, rep, info).
    The manifest is rebuilt with delta_theta = 0 (pure reflection, no phase shift) and each channel duplicated into a
    'mean' copy (keeps its mirror sign) and a 'std' copy (a magnitude, sign +1)."""
    chans = [c["name"] for c in manifest["channels"] if c["in_Z"]]
    by_name = {c["name"]: c for c in manifest["channels"]}
    # drop channels the dataset does not provide (Mini Cheetah has tau_est -> tau_meas but no tau_cmd), keeping mirror
    # pairs intact: a channel is kept only if it AND its partner are finite on the masked rows.
    X_all = df[chans].to_numpy()[mask]
    finite = {c: bool(np.isfinite(X_all[:, i]).all()) for i, c in enumerate(chans)}
    chans = [c for c in chans if finite.get(c, False) and finite.get(by_name[c]["partner"], False)]
    t = df["t"].to_numpy(); dt = float(np.median(np.diff(t)))
    idx = np.where(mask)[0]
    blocks = []
    if len(idx):
        for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            n_per = int(round(L_s / dt))
            for b in range(len(r) // n_per):
                sl = r[b * n_per:(b + 1) * n_per]
                X = df[chans].to_numpy()[sl]
                if not np.isfinite(X).all():
                    continue
                blocks.append(np.stack([X.mean(axis=0), X.std(axis=0)], axis=1))     # (d, 2)
    if not blocks:
        return np.empty((0, 2 * len(chans), 1)), None, {"K": 0}
    Z = np.stack(blocks)                                   # (K, d, 2)
    # flatten (channel, moment) into one channel axis with N = 1 grid point
    K, d, _ = Z.shape
    Zf = np.concatenate([Z[:, :, 0], Z[:, :, 1]], axis=1)[:, :, None]                # (K, 2d, 1)
    ch = []
    for c in chans:                                        # mean copies keep the channel's mirror sign
        src = by_name[c]
        ch.append({"name": f"mean_{c}", "group": src["group"], "leg": src.get("leg"), "joint": src.get("joint"),
                   "kind": src["kind"], "partner": f"mean_{src['partner']}", "sign": src["sign"], "in_Z": True})
    for c in chans:                                        # std copies are magnitudes
        src = by_name[c]
        ch.append({"name": f"std_{c}", "group": src["group"], "leg": src.get("leg"), "joint": src.get("joint"),
                   "kind": "scalar-magnitude", "partner": f"std_{src['partner']}", "sign": +1, "in_Z": True})
    man2 = {"channels": ch, "gait_group": {"delta_theta": 0.0}}                        # pure reflection: no phase shift
    return Zf, C2Rep(man2), {"K": int(K), "L_s": L_s, "d": int(2 * len(chans))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=float, default=2.0); ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    res = DATA / "results" / "e17_public_realdata" / (a.run_id or f"b6-{_dt.datetime.now().strftime('%Y%m%d-%H%M')}")
    res.mkdir(parents=True, exist_ok=True)
    alpha, M, win = 0.05, 512, 10
    rows = []
    for terr in TERRAINS:
        mat = DATA / "data/raw/public/minicheetah-contact" / terr / f"{terr}.mat"
        if not mat.exists():
            continue
        df, man, rep = load_minicheetah(mat)
        mask, mi = stance_trot_mask(df)
        Z, crep, info = block_moment_element(df, man, mask, a.L)
        if info["K"] < 12:
            rows.append({"terrain": terr, "K": info["K"], "error": "too few blocks"}); continue
        K = Z.shape[0]
        r_all = hg_permutation_tests(Z, crep, M=M, rng=np.random.default_rng(0))
        nw = K // win
        pw = np.array([hg_permutation_test(Z[w * win:(w + 1) * win], crep, statistic="paired_energy", M=M,
                                           rng=np.random.default_rng([1, w]))[0] for w in range(nw)])
        E, alarm = eprocess(pw, alpha)
        Kcal = max(win, K // 3); nwp = (K - Kcal) // win
        pwp = np.array([h0prime_test(Z[:Kcal], Z[Kcal + w * win:Kcal + (w + 1) * win], crep, M=M,
                                     rng=np.random.default_rng([2, w]))["p"] for w in range(nwp)])
        Ep, alarmp = eprocess(pwp, alpha) if nwp else (np.array([]), None)
        cal = calibrate(Z[:Kcal], crep, n_boot=100, rng=np.random.default_rng(0))
        rows.append({"terrain": terr, "L_s": a.L, "K": int(K), "d": info["d"], "straight_s": round(mi["masked_s"], 1),
                     "H0_p": float(r_all["paired_energy"]["p"]), "H0_window_rej": float(np.mean(pw <= alpha)),
                     "H0_alarm": alarm, "H0p_n_windows": int(nwp),
                     "H0p_window_rej": float(np.mean(pwp <= alpha)) if nwp else float("nan"),
                     "H0p_eproc_max": float(Ep.max()) if nwp else float("nan"), "H0p_alarm": alarmp,
                     "H0p_ks_p": float(stats.kstest(pwp, "uniform").pvalue) if nwp > 3 else float("nan"),
                     "nu0": float(cal["nu0"])})
        print(f"[b6] {terr}: K={K} blocks of {a.L}s | H0 p={rows[-1]['H0_p']:.4f} win-rej {rows[-1]['H0_window_rej']:.2f} | "
              f"H0' win-rej {rows[-1]['H0p_window_rej']:.2f} alarm {rows[-1]['H0p_alarm']} (n={nwp})", flush=True)
    T = pd.DataFrame(rows); T.to_csv(res / f"b6_minicheetah_blockmode_L{a.L}.csv", index=False)
    ok = T[T.H0p_window_rej.notna()]
    in_band = int((ok.H0p_window_rej <= 0.12).sum())
    line = (f"[B6] Mini Cheetah flying trot, ROLLING-STYLE {a.L}s blocks with a phase-free (mean,std) element, no cycle "
            f"registration: H0' window-reject rate {list(ok.H0p_window_rej.round(2))} on {len(ok)} terrains, "
            f"{in_band}/{len(ok)} within the H0 band (<=0.12); H0' e-process alarms on {int(ok.H0p_alarm.notna().sum())}/{len(ok)}; "
            f"naive H0 still rejects on {int((ok.H0_p<=alpha).sum())}/{len(ok)}.")
    (res / "conclusions.txt").open("a").write(line + "\n"); print(line)


if __name__ == "__main__":
    main()
