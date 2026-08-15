#!/usr/bin/env python3
"""e03 — external public simulation benchmark (Liu et al. RA-L 2025, Unitree A1 in Gazebo). Sprint 7 Block E2.
Pre-registration: docs/protocol/e03_preregistration.md (committed before this run).

  prep    load the 10 CSVs, estimated phase (HFE diagonal signal), cycles + half-cycles, per-half-cycle mirror score /
          tracking-error score / Mahalanobis features, command condition and fault label -> parquet cache
  detect  R- half-cycle e-process, R+ tracking conformal e-process, Mahalanobis conformal e-process: per-episode detection
          (inside the episode + grace), delay, localisation; false alarms on the healthy gaps; per-class summary
  gru     Liu Table I GRU regressor (57 -> 256 -> 12, MSE), leave-one-file-out x 3 seeds + eta and single->double splits;
          deployment rule (low-pass eta_hat, joint faulty iff < 0.7) and the unified conformal score 1 - min eta_hat
  report  tables + the four-class figure

    python experiments/e03_liu_a1_headtohead/run.py --stage prep|detect|gru|report|all [--run-id ID] [--quick]
Only derived statistics are written (no raw rows).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.detect.evalue import p_to_e
from geofdi.detect.monitors import conformal_pvalues
from geofdi.detect.sequential import calibration_scale, half_cycles
from geofdi.groups.c2 import C2Rep
from geofdi.io.liu_a1 import JOINTS, LEGS, CALF_INDEX, command_segments, fault_episodes, load_liu_file
from geofdi.phase.estimator import estimate_phase
from geofdi.phase.registration import register_cycles

EXP_NAME = "e03_liu_a1_headtohead"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
PAIR_OF = {"LF": "F", "RF": "F", "LH": "H", "RH": "H"}


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------------------------ prep
def prep(cfg, res_dir, quick=False):
    ddir = DATA_ROOT / "data" / cfg["data_dir"]; files = sorted(ddir.glob("*.csv"))
    if quick:
        files = files[:2]
    N = cfg["registration"]["N"]; rows = []; H_all = {}; man = None
    for f in files:
        df, man, lab = load_liu_file(f); rep = C2Rep(man); chans = [c["name"] for c in man["channels"] if c["in_Z"]]
        th, pinfo = estimate_phase(df, joint=cfg["registration"]["phase_joint"], fs=cfg["rate_hz"])
        d2 = df.copy(); d2["th"] = th
        Z, meta = register_cycles(d2, chans, N=N, theta_col="th", drop_first=cfg["registration"]["drop_first_cycles"], drop_last=cfg["registration"]["drop_last_cycles"])
        Zq, _ = register_cycles(d2, [f"q_{l}_{j}" for l in LEGS for j in JOINTS] + [f"q_des_{l}_{j}" for l in LEGS for j in JOINTS], N=N, theta_col="th",
                                drop_first=cfg["registration"]["drop_first_cycles"], drop_last=cfg["registration"]["drop_last_cycles"])
        rs = np.array(meta["row_start"]); K = Z.shape[0]
        # half-cycle bookkeeping: rows at half-cycle starts (approx: cycle start + half the cycle length)
        lens = np.diff(np.append(rs, rs[-1] + int(np.median(np.diff(rs))) if K > 1 else rs[-1] + 60))
        h_rows = np.empty(2 * K, dtype=int); h_rows[0::2] = rs; h_rows[1::2] = rs + lens // 2
        h_end = np.append(h_rows[1:], rs[-1] + lens[-1])
        H = half_cycles(Z); Hq = half_cycles(Zq)
        # scores per half-cycle j >= 1 (mirror score uses j-1)
        eps = fault_episodes(lab); segs = command_segments(df)
        cond = np.full(len(df), -1); cond_key = {}
        for s in segs:
            key = (round(s["vx"], 3), round(s["vy"], 3), round(s["wz"], 3)); cond_key.setdefault(key, len(cond_key)); cond[s["row_start"]:s["row_end"]] = cond_key[key]
        label = np.full(len(df), -1)                       # -1 healthy, else episode index
        for ei, e in enumerate(eps):
            label[e["row_start"]:e["row_end"]] = ei
        H_all[f.name] = {"H": H.astype(np.float32), "Hq": Hq.astype(np.float32), "h_rows": h_rows, "h_end": h_end, "cond": cond, "cond_key": {str(k): v for k, v in cond_key.items()},
                         "label": label, "episodes": eps, "segments": segs, "chans": chans, "phase": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in pinfo.items()},
                         "K": K, "n_rows": len(df)}
        rows.append({"file": f.name, "rows": len(df), "cycles": K, "f0_hz": pinfo["f0_hz"], "episodes": len(eps), "classes": json.dumps({c: sum(e["cls"] == c for e in eps) for c in ("single", "mirror", "same_side", "diagonal")}),
                     "conditions": json.dumps({str(k): v for k, v in cond_key.items()})})
        print(f"  [prep] {f.name}: {K} cycles, f0 {pinfo['f0_hz']:.3f} Hz, {len(eps)} episodes", flush=True)
    pd.DataFrame(rows).to_csv(res_dir / "e03_prep_files.csv", index=False)
    np.save(res_dir / "prep_cache.npy", np.array([H_all, man], dtype=object), allow_pickle=True)
    return H_all, man


# ------------------------------------------------------------------------------------ detect
def _hc_scores(H, rep, scale):
    Hs = rep.mirror_only(H); D = (H[1:] - Hs[:-1]) / scale
    per_ch = (D ** 2).mean(axis=2)                                    # (2K-1, d) per-channel contribution
    return per_ch.mean(axis=1), per_ch


def _episode_windows(labels_hc, cond_hc, h_rows, h_end, eps, grace_rows, n_rows):
    """For each episode: indices of half-cycles that START inside [onset, offset + grace); healthy gaps: half-cycles
    starting in [prev_end + grace, next_onset). Returns list of (episode dict or None, hc index array)."""
    out = []
    onsets = [e["row_start"] for e in eps]; ends = [e["row_end"] for e in eps]
    for ei, e in enumerate(eps):
        idx = np.where((h_rows >= e["row_start"]) & (h_rows < e["row_end"] + grace_rows))[0]
        out.append((e, idx))
    gaps = []
    bounds = [(0, onsets[0])] + [(ends[i] + grace_rows, onsets[i + 1]) for i in range(len(eps) - 1)] + [(ends[-1] + grace_rows, n_rows)]
    for a, b in bounds:
        idx = np.where((h_rows >= a) & (h_rows < b))[0]
        if len(idx) >= 4:
            gaps.append((a, b, idx))
    return out, gaps


def _run_eprocess(p, alpha):
    E = np.cumprod(p_to_e(p)); hits = np.where(E >= 1.0 / alpha)[0]
    return (int(hits[0]) if len(hits) else None), float(E.max()) if len(E) else 0.0


def detect(cfg, res_dir, H_all=None, man=None, quick=False):
    if H_all is None:
        H_all, man = np.load(res_dir / "prep_cache.npy", allow_pickle=True)
    rep = C2Rep(man); alpha = cfg["detect"]["alpha"]; grace = int(round(cfg["detect"]["grace_s"] * cfg["rate_hz"])); minc = cfg["detect"]["min_calibration_halfcycles"]
    chans = next(iter(H_all.values()))["chans"]
    # global scale from all healthy half-cycles of all files
    Hn = np.concatenate([v["H"][np.where(v["label"][v["h_rows"]] < 0)[0]] for v in H_all.values()])
    scale = calibration_scale(Hn, rep)
    from sklearn.covariance import LedoitWolf
    # per-file per-half-cycle scores
    per_file = {}
    for fn, v in H_all.items():
        s_m, per_ch = _hc_scores(v["H"], rep, scale)                        # index j-1 -> half-cycle j
        Hq = v["Hq"]; nq = Hq.shape[1] // 2
        s_track = ((Hq[:, :nq, :] - Hq[:, nq:, :]) ** 2).mean(axis=(1, 2))         # per half-cycle tracking error
        per_leg = np.stack([((Hq[:, 3 * li:3 * li + 3, :] - Hq[:, nq + 3 * li:nq + 3 * li + 3, :]) ** 2).mean(axis=(1, 2)) for li in range(4)], axis=1)
        feat = np.concatenate([v["H"].mean(axis=2), v["H"].std(axis=2)], axis=1)   # Mahalanobis features per half-cycle
        lab_hc = v["label"][v["h_rows"]]; cond_hc = v["cond"][v["h_rows"]]
        per_file[fn] = {"s_m": s_m, "per_ch": per_ch, "s_track": s_track, "per_leg": per_leg, "feat": feat, "lab": lab_hc, "cond": cond_hc, "cond_key": v["cond_key"]}
    # conformal calibration per (file, condition): healthy half-cycles of the OTHER files with the same command
    def cal_sets(fn, cond_val, kind):
        key = [k for k, vv in per_file[fn]["cond_key"].items() if vv == cond_val]
        key = key[0] if key else None
        pool = []
        for gn, g in per_file.items():
            if gn == fn:
                continue
            gk = [k for k, vv in g["cond_key"].items() if k == key]
            if not gk:
                continue
            cval = g["cond_key"][gk[0]]; m = (g["lab"] < 0) & (g["cond"] == cval)
            if kind == "s_m":
                m = m[1:]; pool.append(g["s_m"][m])
            elif kind == "s_track":
                pool.append(g["s_track"][m])
            else:
                pool.append(g["feat"][m])
        if pool and sum(len(x) for x in pool) >= minc:
            return np.concatenate(pool), "other files, same condition"
        pool = []
        for gn, g in per_file.items():
            if gn == fn:
                continue
            m = g["lab"] < 0
            pool.append(g["s_m"][m[1:]] if kind == "s_m" else (g["s_track"][m] if kind == "s_track" else g["feat"][m]))
        return np.concatenate(pool), "other files, all conditions"
    rows = []; gap_rows = []
    for fn, v in H_all.items():
        pf = per_file[fn]; eps = v["episodes"]
        seg_of_row = lambda r0: next((sg for sg in v["segments"] if sg["row_start"] <= r0 < sg["row_end"]), None)
        ep_w, gaps = _episode_windows(pf["lab"], pf["cond"], v["h_rows"], v["h_end"], eps, grace, v["n_rows"])
        # detectors: p-values per half-cycle (index space of half-cycles; s_m index j-1)
        for det in ("Rminus_halfcycle", "Rplus_track", "mahalanobis"):
            # per condition calibration
            p_all = np.full(len(v["h_rows"]), np.nan); src = {}
            for cval in np.unique(pf["cond"]):
                if cval < 0:
                    continue
                cal, source = cal_sets(fn, cval, "s_m" if det == "Rminus_halfcycle" else ("s_track" if det == "Rplus_track" else "feat")); src[int(cval)] = source
                idx = np.where(pf["cond"] == cval)[0]
                if det == "Rminus_halfcycle":
                    idx = idx[idx >= 1]; p_all[idx] = conformal_pvalues(cal, pf["s_m"][idx - 1])
                elif det == "Rplus_track":
                    p_all[idx] = conformal_pvalues(cal, pf["s_track"][idx])
                else:
                    lw = LedoitWolf().fit(cal); mu = cal.mean(0); P = lw.precision_
                    sc_cal = np.einsum("ki,ij,kj->k", cal - mu, P, cal - mu); sc = np.einsum("ki,ij,kj->k", pf["feat"][idx] - mu, P, pf["feat"][idx] - mu)
                    p_all[idx] = conformal_pvalues(sc_cal, sc)
            for e, idx in ep_w:
                idx = idx[np.isfinite(p_all[idx])]
                if len(idx) == 0:
                    sg = seg_of_row(e["row_start"]) or {"straight": None, "vx": np.nan, "wz": np.nan}
                    rows.append({"file": fn, "episode": eps.index(e), "cls": e["cls"], "eta": e["eta"][0], "legs": "+".join(e["legs"]), "duration_s": (e["row_end"] - e["row_start"]) / cfg["rate_hz"], "detector": det, "detected": False, "delay_s": np.nan, "n_hc": 0, "loc_pair": None, "loc_pair_correct": None, "loc_leg": None, "loc_leg_correct": None, "straight": sg["straight"], "cmd_vx": sg["vx"], "cmd_wz": sg["wz"]}); continue
                al, Emax = _run_eprocess(p_all[idx], alpha)
                det_ok = al is not None
                delay = (v["h_end"][idx[al]] - e["row_start"]) / cfg["rate_hz"] if det_ok else np.nan
                loc_pair = loc_leg = None; pc = lc = None
                if det_ok and det == "Rminus_halfcycle":
                    ch_e = pf["per_ch"][idx[:al + 1] - 1].mean(axis=0)     # per-channel contribution over the alarm segment
                    grp = {}
                    for i, n in enumerate(chans):
                        parts = n.split("_")
                        if len(parts) >= 3 and parts[-2] in LEGS:
                            grp[(PAIR_OF[parts[-2]], parts[-1])] = grp.get((PAIR_OF[parts[-2]], parts[-1]), 0.0) + ch_e[i]
                    loc_pair = max(grp.items(), key=lambda kv: kv[1])[0]; loc_pair = f"{loc_pair[0]}-{loc_pair[1]}"
                    true_pairs = {f"{PAIR_OF[l]}-KFE" for l in e["legs"]}; pc = loc_pair in true_pairs
                if det_ok and det == "Rplus_track":
                    dev = pf["per_leg"][idx[:al + 1]].mean(axis=0)
                    # relative to the file's healthy per-leg level (same condition)
                    hm = (pf["lab"] < 0) & (pf["cond"] == pf["cond"][idx[0]]); base = pf["per_leg"][hm].mean(axis=0) + 1e-12
                    loc_leg = LEGS[int(np.argmax(dev / base))]; lc = loc_leg in e["legs"]
                sg = seg_of_row(e["row_start"]) or {"straight": None, "vx": np.nan, "wz": np.nan}
                rows.append({"file": fn, "episode": eps.index(e), "cls": e["cls"], "eta": e["eta"][0], "legs": "+".join(e["legs"]), "duration_s": (e["row_end"] - e["row_start"]) / cfg["rate_hz"], "detector": det,
                             "detected": bool(det_ok), "delay_s": float(delay) if det_ok else np.nan, "n_hc": int(len(idx)), "Emax": Emax, "loc_pair": loc_pair, "loc_pair_correct": pc, "loc_leg": loc_leg, "loc_leg_correct": lc,
                             "straight": sg["straight"], "cmd_vx": sg["vx"], "cmd_wz": sg["wz"]})
            for a, b, idx in gaps:
                idx = idx[np.isfinite(p_all[idx])]
                if len(idx) < 4:
                    continue
                al, Emax = _run_eprocess(p_all[idx], alpha)
                gap_rows.append({"file": fn, "gap_start_row": int(a), "gap_len_s": (b - a) / cfg["rate_hz"], "detector": det, "false_alarm": al is not None, "n_hc": int(len(idx)), "cond": int(pf["cond"][idx[0]])})
        print(f"  [detect] {fn} done", flush=True)
    ep = pd.DataFrame(rows); ep.to_csv(res_dir / "e03_episodes.csv", index=False)
    gp = pd.DataFrame(gap_rows); gp.to_csv(res_dir / "e03_healthy_gaps.csv", index=False)
    return ep, gp


# ------------------------------------------------------------------------------------ gru
def _file_arrays(path, cfg):
    """57-input sequence at 50 Hz (rows decimated by 2) in the paper's layout + eta targets + labels/conditions."""
    X = np.loadtxt(path, delimiter=",", usecols=range(69))
    seq = np.concatenate([X[:, 0:3], X[:, 3:6], X[:, 18:30], X[:, 30:42], X[:, 42:54], X[:, 54:66], X[:, 66:69]], axis=1)   # 57
    eta = X[:, 6:18]
    dec = 100 // cfg["gru"]["fs_hz"]
    return seq[::dec].astype(np.float32), eta[::dec].astype(np.float32)


def gru_stage(cfg, res_dir, quick=False):
    import torch
    from geofdi.baselines.gru import GRURegressor, WindowSetReg, eta_lowpass_threshold, predict_eta, train_gru_regressor
    ddir = DATA_ROOT / "data" / cfg["data_dir"]; files = sorted(ddir.glob("*.csv"))
    if quick:
        files = files[:3]
    g = cfg["gru"]; dev = "cuda" if torch.cuda.is_available() else "cpu"; dec = 100 // g["fs_hz"]
    data = {f.name: _file_arrays(f, cfg) for f in files}
    mu = np.concatenate([d[0] for d in data.values()]).mean(0); sd = np.concatenate([d[0] for d in data.values()]).std(0) + 1e-6
    for k in data:
        data[k] = ((data[k][0] - mu) / sd, data[k][1])
    epochs = 3 if quick else g["epochs"]; seeds = g["seeds"][:1] if quick else g["seeds"]
    grace = int(round(cfg["detect"]["grace_s"] * g["fs_hz"]))
    splits = []
    for f in files:                                                                 # LOFO
        splits.append(("lofo", f.name, [k for k in data if k != f.name], [f.name], None))
    if not quick:
        splits.append(("eta_0.4_to_0.6", "all", list(data), list(data), (0.4, 0.6)))
        splits.append(("eta_0.6_to_0.4", "all", list(data), list(data), (0.6, 0.4)))
        singles = [k for k in data if "Single" in k]; doubles = [k for k in data if "Double" in k]
        splits.append(("single_to_double", "doubles", singles, doubles, None))
    rows = []; ep_rows = []
    for split, name, train_files, test_files, eta_split in splits:
        for seed in seeds:
            seqs, tg = [], []
            for k in train_files:
                s, e = data[k]
                if eta_split is not None:            # keep rows that are healthy or at the training eta
                    keep = np.all(np.isclose(e, 1.0) | np.isclose(e, eta_split[0]), axis=1)
                    # cut into contiguous kept segments
                    idx = np.where(keep)[0]; br = np.where(np.diff(idx) > 1)[0] + 1
                    for seg in np.split(idx, br):
                        if len(seg) > g["window"]:
                            seqs.append(s[seg]); tg.append(e[seg])
                else:
                    seqs.append(s); tg.append(e)
            ws = WindowSetReg(seqs, tg, g["window"], g["stride_train"])
            torch.manual_seed(seed); model = GRURegressor(g["input"], g["hidden"], g["layers"], 12)
            t0 = _dt.datetime.now()
            hist = train_gru_regressor(model, ws, epochs=epochs, batch=g["batch"], lr=g["lr"], device=dev, seed=seed, max_batches_per_epoch=g["max_batches_per_epoch"], log=None)
            secs = (_dt.datetime.now() - t0).total_seconds()
            # test: sliding windows stride 1 on each test file
            for k in test_files:
                s, e = data[k]; n = len(s)
                idx0 = np.arange(0, n - g["window"] + 1)
                X = np.stack([s[i:i + g["window"]] for i in idx0])
                eta_hat = predict_eta(model, X, device=dev)                       # window ends at i + window - 1
                flags = eta_lowpass_threshold(eta_hat, fc_hz=g["lowpass_hz"], fs=g["fs_hz"], thr=g["threshold"])
                t_end = idx0 + g["window"] - 1
                lab_e = e[t_end]; faulty = lab_e < 0.999                            # per window-end row: which joints faulty (truth)
                # episodes at 50 Hz
                fl = np.any(faulty, axis=1); ch = np.where(np.diff(fl.astype(int)) != 0)[0] + 1; b = np.concatenate([[0], ch, [len(fl)]])
                for a0, b0 in zip(b[:-1], b[1:]):
                    seg_f = fl[a0]
                    if eta_split is not None:
                        # test only episodes at the held-out eta (train eta episodes are 'seen')
                        if seg_f and not np.any(np.isclose(lab_e[a0][faulty[a0]], eta_split[1])):
                            continue
                    if seg_f:
                        true_j = np.where(faulty[a0])[0]; end = min(b0 + grace, len(fl))
                        hit = np.where(np.any(flags[a0:end], axis=1))[0]
                        det = len(hit) > 0; delay = hit[0] / g["fs_hz"] if det else np.nan
                        loc = int(np.argmin(eta_hat[a0 + hit[0]])) if det else None
                        legs = [CALF_INDEX.get(int(j), f"j{j}") for j in true_j]
                        cls = "single" if len(legs) == 1 else {("LF", "RF"): "mirror", ("LH", "RH"): "mirror", ("LF", "LH"): "same_side", ("RF", "RH"): "same_side", ("LF", "RH"): "diagonal", ("LH", "RF"): "diagonal"}.get(tuple(sorted(legs)), "other")
                        ep_rows.append({"split": split, "fold": name, "seed": seed, "file": k, "cls": cls, "eta": float(lab_e[a0][true_j[0]]), "legs": "+".join(legs), "detected": det, "delay_s": delay,
                                        "loc_joint": loc, "loc_correct": (loc in true_j.tolist()) if det else None, "duration_s": (b0 - a0) / g["fs_hz"]})
                    else:
                        a1 = a0 + grace if a0 > 0 else a0
                        if b0 - a1 > 2 * g["fs_hz"]:
                            fa = bool(np.any(flags[a1:b0]))
                            ep_rows.append({"split": split, "fold": name, "seed": seed, "file": k, "cls": "healthy_gap", "eta": 1.0, "legs": "", "detected": fa, "delay_s": np.nan, "loc_joint": None, "loc_correct": None, "duration_s": (b0 - a1) / g["fs_hz"]})
            rows.append({"split": split, "fold": name, "seed": seed, "train_files": len(train_files), "n_windows": len(ws), "final_train_mse": hist[-1]["loss"], "train_seconds": secs, "device": dev})
            print(f"  [gru] {split} {name} seed {seed}: train MSE {hist[-1]['loss']:.4f} in {secs:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(res_dir / "e03_gru_training.csv", index=False)
    pd.DataFrame(ep_rows).to_csv(res_dir / "e03_gru_episodes.csv", index=False)


# ------------------------------------------------------------------------------------ report
def report(cfg, res_dir):
    ep = pd.read_csv(res_dir / "e03_episodes.csv"); gp = pd.read_csv(res_dir / "e03_healthy_gaps.csv")
    summ = ep.groupby(["cls", "detector"]).agg(n=("detected", "size"), det_rate=("detected", "mean"), delay_median_s=("delay_s", "median"), delay_q90_s=("delay_s", lambda x: x.quantile(0.9)),
                                                loc_pair_acc=("loc_pair_correct", lambda x: pd.Series(x).dropna().astype(float).mean() if x.notna().any() else np.nan),
                                                loc_leg_acc=("loc_leg_correct", lambda x: pd.Series(x).dropna().astype(float).mean() if x.notna().any() else np.nan)).reset_index()
    by_eta = ep.groupby(["cls", "eta", "detector"]).agg(n=("detected", "size"), det_rate=("detected", "mean"), delay_median_s=("delay_s", "median")).reset_index()
    by_str = ep.groupby(["cls", "straight", "detector"]).agg(n=("detected", "size"), det_rate=("detected", "mean"), delay_median_s=("delay_s", "median")).reset_index()
    by_str.to_csv(res_dir / "e03_summary_by_class_straight.csv", index=False)
    fa = gp.groupby("detector").agg(n_gaps=("false_alarm", "size"), fa_rate_per_gap=("false_alarm", "mean"), gap_len_s=("gap_len_s", "mean")).reset_index()
    summ.to_csv(res_dir / "e03_summary_by_class.csv", index=False); by_eta.to_csv(res_dir / "e03_summary_by_class_eta.csv", index=False); fa.to_csv(res_dir / "e03_false_alarms.csv", index=False)
    gru_line = ""
    if (res_dir / "e03_gru_episodes.csv").exists():
        ge = pd.read_csv(res_dir / "e03_gru_episodes.csv")
        gs = ge[ge.cls != "healthy_gap"].groupby(["split", "cls"]).agg(n=("detected", "size"), det_rate=("detected", "mean"), delay_median_s=("delay_s", "median"), loc_acc=("loc_correct", lambda x: pd.Series(x).dropna().astype(float).mean() if x.notna().any() else np.nan)).reset_index()
        gfa = ge[ge.cls == "healthy_gap"].groupby("split").agg(n=("detected", "size"), fa_rate_per_gap=("detected", "mean")).reset_index()
        gsp = ge[ge.cls != "healthy_gap"].groupby(["split", "cls", "seed"]).detected.mean().reset_index().groupby(["split", "cls"]).detected.agg(["mean", "std"]).reset_index()
        gs.to_csv(res_dir / "e03_gru_summary.csv", index=False); gfa.to_csv(res_dir / "e03_gru_false_alarms.csv", index=False); gsp.to_csv(res_dir / "e03_gru_seed_spread.csv", index=False)
        gru_line = " | GRU (Table I, rule eta<0.7): " + "; ".join(f"{r.split}/{r.cls}: det {r.det_rate:.2f} (n={r.n}) delay {r.delay_median_s:.2f}s loc {r.loc_acc:.2f}" for r in gs.itertuples()) + " | GRU FA per gap: " + "; ".join(f"{r.split} {r.fa_rate_per_gap:.3f}" for r in gfa.itertuples())
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    classes = ["single", "mirror", "same_side", "diagonal"]; dets = ["Rminus_halfcycle", "Rplus_track", "mahalanobis"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(len(classes)); w = 0.25
    for i, d in enumerate(dets):
        v = [float(summ[(summ.cls == c) & (summ.detector == d)].det_rate.iloc[0]) if len(summ[(summ.cls == c) & (summ.detector == d)]) else np.nan for c in classes]
        axes[0].bar(x + (i - 1) * w, v, width=w, label=d)
    for i, d in enumerate(dets):
        f_ = fa[fa.detector == d]
        if len(f_):
            axes[0].axhline(float(f_.fa_rate_per_gap.iloc[0]), color=f"C{i}", ls=":", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(classes); axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("episode detection rate"); axes[0].legend(fontsize=7); axes[0].set_title("e03 — detection per fault class (dotted: false-alarm rate per 4 s healthy gap)", fontsize=8)
    for i, d in enumerate(dets):
        v = [float(summ[(summ.cls == c) & (summ.detector == d)].delay_median_s.iloc[0]) if len(summ[(summ.cls == c) & (summ.detector == d)]) else np.nan for c in classes]
        axes[1].bar(x + (i - 1) * w, v, width=w, label=d)
    axes[1].set_xticks(x); axes[1].set_xticklabels(classes); axes[1].set_ylabel("median delay [s]"); axes[1].set_title("median detection delay (episodes are 1-2 s)", fontsize=8); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(res_dir / "e03_four_classes.png", dpi=140); plt.close(fig)
    stl = by_str[by_str.detector == "Rminus_halfcycle"]
    line = "[e03] R- by class x straight/turning: " + "; ".join(f"{r.cls}/{'straight' if r.straight else 'turning'}: det {r.det_rate:.2f} (n={r.n})" for r in stl.itertuples()) + " || per class (n, det rate, median delay s): " + "; ".join(f"{r.cls}/{r.detector}: n={r.n} det {r.det_rate:.2f} delay {r.delay_median_s:.2f}" + (f" pair-loc {r.loc_pair_acc:.2f}" if np.isfinite(r.loc_pair_acc) else "") + (f" leg-loc {r.loc_leg_acc:.2f}" if np.isfinite(r.loc_leg_acc) else "") for r in summ.itertuples()) \
        + " | false alarms per healthy gap: " + "; ".join(f"{r.detector} {r.fa_rate_per_gap:.3f} (n={r.n_gaps}, {r.gap_len_s:.1f} s)" for r in fa.itertuples()) + gru_line
    _conclude(res_dir, line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["prep", "detect", "gru", "report", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml")); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True); (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["prep", "detect", "gru", "report"] if a.stage == "all" else [a.stage]
    print(f"[{EXP_NAME}] run_id={a.run_id} stages={stages} quick={a.quick}", flush=True)
    H_all = man = None
    for s in stages:
        t0 = _dt.datetime.now()
        if s == "prep":
            H_all, man = prep(cfg, res_dir, quick=a.quick)
        elif s == "detect":
            detect(cfg, res_dir, H_all, man, quick=a.quick)
        elif s == "gru":
            gru_stage(cfg, res_dir, quick=a.quick)
        else:
            report(cfg, res_dir)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E03 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
