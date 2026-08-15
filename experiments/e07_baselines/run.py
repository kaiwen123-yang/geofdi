#!/usr/bin/env python3
"""e07 — baselines under the unified FAR protocol: {R- e-CUSUM, rplus_resid, rplus_track, GRU (fault-trained),
GRU-noInertia (unseen-type test), AE (nominal), Mahalanobis (nominal)} on the e04a fault grid + nuisance rows.

    python experiments/e07_baselines/run.py [--run-id ID] [--quick]

Stages inside: (1) training data (nominal + seen-magnitude faults) -> GRU x2, AE, Mahalanobis fit (GPU);
(2) test grid + nuisances -> per-run scores of all detectors -> unified alarms -> tables and figures.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.baselines.autoencoder import WindowAE, ae_scores, train_ae
from geofdi.baselines.gru import GRUClassifier, WindowSet, predict_windows, train_gru
from geofdi.baselines.mahalanobis import MahalanobisGate, cycle_features
from geofdi.baselines.protocol import alarm_from_scores, cycle_scores_from_windows
from geofdi.detect.monitors import MirrorMonitor, calibrate_ecusum_threshold, ecusum, tracking_scores
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import binom_ci, pmap
from geofdi.sim.telemetry import JOINTS, LEGS, z_channel_names

EXP_NAME = "e07_baselines"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]


def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        s[k] = {**s.get("controller", {}), **v} if k == "controller" else v
    return s


def _conclude(res_dir, line):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------------------------ worker
def _run(sim_cfg, K_cal, K_post, N, drop_first, ocfg, det, seed, want_seq=True, want_rminus=True):
    """One rollout -> everything the detectors need: Z cycles-based scores computed here (R- window p-values, residual
    scores, tracking scores, Mahalanobis features), plus the standardized raw sequence for the GRU/AE (scored in the parent)."""
    cfg = SimConfig(**sim_cfg); period = float(cfg.controller.get("period_s", 0.5))
    cfg.duration_s = (K_cal + K_post + drop_first + 2) * period
    df, man = rollout(cfg)
    chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=drop_first)
    Zq, _ = register_cycles(df, QREF, N=N, drop_first=drop_first)
    K = min(K_cal + K_post, Z.shape[0]); Z = Z[:K]; Zq = Zq[:K]
    out = {"K": K, "row_start": meta["row_start"][:K]}
    # our channels
    dyn = Go2Dynamics(ocfg["backend"], armature=ocfg["armature"], damping=ocfg["damping"], frictionloss=ocfg["frictionloss"])
    r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=ocfg["cutoff_hz"], torque=ocfg["torque"])[:, 6:]
    Zr, _ = registered_residuals(df, r, N=N, drop_first=drop_first)
    out["score_resid"] = residual_scores(Zr[:K]); out["score_track"] = tracking_scores(Z, chans, Zq)
    if want_rminus:
        rep = C2Rep(man); mm = MirrorMonitor(rep, window=det["window_rminus"], M=det["M"], statistic="paired_energy", alpha=det["alpha"])
        out["p_rminus"] = mm.window_pvalues(Z, seed=seed)
    out["feat_maha"] = cycle_features(Z)
    if want_seq:
        X = df[chans].to_numpy().astype(np.float32); out["seq"] = X
        # cycle index per row (-1 outside registered cycles)
        cyc = np.full(len(df), -1, dtype=np.int32); rs = list(meta["row_start"][:K]) + [len(df)]
        for k in range(K):
            cyc[rs[k]:rs[k + 1]] = k
        out["row_cycle"] = cyc
        out["chans"] = chans
    return out


def _windows(seq, win, stride):
    idx = np.arange(0, len(seq) - win + 1, stride)
    return np.stack([seq[a:a + win] for a in idx]), idx + win - 1          # windows and their end rows


# ------------------------------------------------------------------------------------ main pipeline
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    quick = args.quick
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[{EXP_NAME}] run_id={args.run_id} quick={quick}", flush=True)
    P = cfg["protocol"]; K_cal, K_post = P["K_cal"], P["K_post"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]
    oc = cfg["observer"]; det = cfg["detect"]; alpha = det["alpha"]; win = cfg["window"]["win"]
    t_on = (K_cal + df0) * 0.5
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_all0 = _dt.datetime.now()

    # ---------------- (1) training data
    tr = cfg["train"]; n_nom = 4 if quick else tr["nominal_reps"]; n_f = 1 if quick else tr["fault_reps"]
    train_args = []; train_meta = []
    sid = tr["seed_base"]
    for r in range(n_nom):
        train_args.append((_sim(cfg, sid), K_cal, K_post, N, df0, oc, det, sid, True, False)); train_meta.append(("nominal", None, None, sid)); sid += 1
    for ftype, mags in tr["seen"].items():
        for m in mags:
            for joint in P["joints"]:
                for r in range(n_f):
                    fault = dict(type=ftype, leg=P["leg"], joint=joint, magnitude=float(m), t_onset=t_on)
                    train_args.append((_sim(cfg, sid, faults=[fault]), K_cal, K_post, N, df0, oc, det, sid, True, False)); train_meta.append((ftype, m, joint, sid)); sid += 1
    print(f"  [train] {len(train_args)} rollouts", flush=True)
    tres = pmap(_run, train_args, cfg["workers"])
    chans = tres[0]["chans"]; d = len(chans)
    nom_seqs = [o["seq"] for o, mt in zip(tres, train_meta) if mt[0] == "nominal"]
    mu = np.concatenate(nom_seqs).mean(0); sd = np.concatenate(nom_seqs).std(0) + 1e-6
    std = lambda X: (X - mu) / sd
    # labels per row: 0 before onset row, 1 after (fault runs); nominal all 0
    def row_labels(o, mt):
        lab = np.zeros(len(o["seq"]), dtype=np.float32)
        if mt[0] != "nominal":
            onset_row = o["row_start"][K_cal] if len(o["row_start"]) > K_cal else len(o["seq"])
            lab[onset_row:] = 1.0
        return lab
    seqs = [std(o["seq"]) for o in tres]; labs = [row_labels(o, mt) for o, mt in zip(tres, train_meta)]
    # GRU (all seen types) and GRU-holdout (without the held-out type)
    g = cfg["gru"]; models = {}
    for name, excl in (("gru", None), ("gru_noinertia", tr["unseen_type_holdout"])):
        keep = [i for i, mt in enumerate(train_meta) if excl is None or mt[0] != excl]
        ws = WindowSet([seqs[i] for i in keep], [labs[i] for i in keep], win, cfg["window"]["stride_train"])
        model = GRUClassifier(d, g["hidden"], g["layers"])
        t0 = _dt.datetime.now()
        hist = train_gru(model, ws, epochs=(2 if quick else g["epochs"]), batch=g["batch"], lr=g["lr"], device=dev, log=lambda s: print(s, flush=True))
        models[name] = model; print(f"  [train] {name}: {len(ws)} windows, {(_dt.datetime.now() - t0).total_seconds():.0f}s, final loss {hist[-1]['loss']:.4f}", flush=True)
    # AE on nominal windows
    a = cfg["ae"]; Xn = np.concatenate([_windows(s, win, cfg["window"]["stride_train"])[0] for s, mt in zip(seqs, train_meta) if mt[0] == "nominal"])
    ae = WindowAE(win, d, a["latent"], a["hidden"]); t0 = _dt.datetime.now()
    train_ae(ae, Xn, epochs=(3 if quick else a["epochs"]), batch=a["batch"], lr=a["lr"], device=dev, log=lambda s: print(s, flush=True))
    print(f"  [train] AE: {len(Xn)} nominal windows, {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    # Mahalanobis on nominal cycle features (all cycles of the nominal training runs)
    F = np.concatenate([o["feat_maha"] for o, mt in zip(tres, train_meta) if mt[0] == "nominal"]); maha = MahalanobisGate().fit(F)
    torch.save({"gru": models["gru"].state_dict(), "gru_noinertia": models["gru_noinertia"].state_dict(), "ae": ae.state_dict(), "mu": mu, "sd": sd, "chans": chans,
                "maha_mu": maha.mu, "maha_P": maha.P, "config": cfg}, DATA_ROOT / "models" / f"e07_baselines_{args.run_id}.pt")
    del tres, seqs, Xn

    # ---------------- (2) test grid + nuisances
    T = cfg["test"]; R = 2 if quick else T["R"]; Rn = 3 if quick else T["nuisance_R"]
    cells = [(ft, m, j) for ft, mags in T["grid"].items() for m in (mags[:1] if quick else mags) for j in P["joints"]]
    if quick:
        cells = cells[:3]
    jobs = []; meta = []; sid = T["seed_base"]
    for ci, (ft, m, j) in enumerate(cells):
        for r in range(R):
            fault = dict(type=ft, leg=P["leg"], joint=j, magnitude=float(m), t_onset=t_on)
            jobs.append((_sim(cfg, sid, faults=[fault]), K_cal, K_post, N, df0, oc, det, sid + 500000, True, True)); meta.append((ft, m, j, r, "fault")); sid += 1
    for name, spec in T["nuisances"].items():
        for r in range(Rn):
            nu = [] if spec.get("type", "none") == "none" else [dict(spec)]        # 'none' = plain nominal run (true FAR row)
            jobs.append((_sim(cfg, sid, nuisance=nu), K_cal, K_post, N, df0, oc, det, sid + 500000, True, True)); meta.append((name, spec.get("magnitude", 0.0), "-", r, "nuisance")); sid += 1
    print(f"  [test] {len(jobs)} runs ({len(cells)} cells x {R} + nuisances)", flush=True)
    rows = []; nom_p_pool = []; per_run = []
    chunk = 44
    for c0 in range(0, len(jobs), chunk):
        res = pmap(_run, jobs[c0:c0 + chunk], cfg["workers"])
        for o, mt in zip(res, meta[c0:c0 + chunk]):
            K = o["K"]; X = std(o["seq"]); W, ends = _windows(X, win, cfg["window"]["stride_test"]); wc = o["row_cycle"][ends]
            sc = {"rplus_resid": o["score_resid"], "rplus_track": o["score_track"], "mahalanobis": maha.score(o["feat_maha"]),
                  "ae": cycle_scores_from_windows(ae_scores(ae, W, dev), wc, K)}
            for name, model in models.items():
                sc[name] = cycle_scores_from_windows(predict_windows(model, W, dev), wc, K)
            rec = {"meta": mt, "K": K, "scores": sc, "p_rminus": o["p_rminus"]}
            nom_p_pool.append(o["p_rminus"][:K_cal // det["window_rminus"]]); per_run.append(rec)
        print(f"  [test] {min(c0 + chunk, len(jobs))}/{len(jobs)} runs scored", flush=True)
    h = calibrate_ecusum_threshold(nom_p_pool, det["ecusum_horizon_windows"], alpha, det["ecusum_boot"], np.random.default_rng(7))
    for rec in per_run:
        ft, m, j, r, kind = rec["meta"]; K = rec["K"]
        S, al = ecusum(rec["p_rminus"], h, start=K_cal // det["window_rminus"])
        d_rm = None if al is None else (al - K_cal // det["window_rminus"] + 1) * det["window_rminus"]
        rows.append({"kind": kind, "fault": ft, "magnitude": m, "joint": j, "rep": r, "detector": "Rminus_ecusum", "delay": d_rm, "delay_naive": np.nan, "naive_rate": np.nan})
        for name, s in rec["scores"].items():
            a_ = alarm_from_scores(s, K_cal, alpha)
            rows.append({"kind": kind, "fault": ft, "magnitude": m, "joint": j, "rep": r, "detector": name, "delay": a_["delay_eproc"], "delay_naive": a_["delay_naive"], "naive_rate": a_["naive_rate"]})
    runs = pd.DataFrame(rows); runs.to_csv(res_dir / "e07_runs.csv", index=False)
    # tables
    def agg(g):
        dl = g["delay"].astype(float)
        return pd.Series({"det100": float(np.mean(dl <= K_post)), "det20": float(np.mean(dl <= 20)), "delay_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan,
                          "delay_q90": float(np.nanquantile(dl, 0.9)) if np.isfinite(dl).any() else np.nan, "naive_rate": float(np.nanmean(g["naive_rate"])), "n": len(g)})
    tab = runs[runs.kind == "fault"].groupby(["fault", "magnitude", "joint", "detector"]).apply(agg).reset_index(); tab.to_csv(res_dir / "e07_table.csv", index=False)
    nuis = runs[runs.kind == "nuisance"].groupby(["fault", "detector"]).apply(lambda g: pd.Series({"alarm_fraction": float(np.mean(g["delay"].astype(float) <= K_post)), "naive_rate": float(np.nanmean(g["naive_rate"])), "n": len(g)})).reset_index()
    nuis.to_csv(res_dir / "e07_nuisance.csv", index=False)
    piv = tab.pivot_table(index=["fault", "magnitude", "joint"], columns="detector", values="det100"); piv.to_csv(res_dir / "e07_det100_pivot.csv")
    pivd = tab.pivot_table(index=["fault", "magnitude", "joint"], columns="detector", values="delay_median"); pivd.to_csv(res_dir / "e07_delay_pivot.csv")
    # GRU generalisation: seen vs unseen magnitude, unseen type
    seen = {ft: set(float(x) for x in mags) for ft, mags in tr["seen"].items()}
    tab["seen_magnitude"] = [float(m) in seen.get(ft, set()) for ft, m in zip(tab.fault, tab.magnitude)]
    gen = tab[tab.detector.isin(["gru", "gru_noinertia"])].groupby(["detector", "fault", "seen_magnitude"]).apply(lambda g: pd.Series({"det100": g.det100.mean(), "delay_median": g.delay_median.median(), "cells": len(g)})).reset_index()
    gen.to_csv(res_dir / "e07_gru_generalisation.csv", index=False)
    # figure: det100 heat per detector and delay-vs-magnitude curves
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dets = ["Rminus_ecusum", "rplus_resid", "rplus_track", "gru", "gru_noinertia", "ae", "mahalanobis"]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(piv) + 1.6)); M_ = piv.reindex(columns=[c for c in dets if c in piv.columns]).to_numpy()
    im = ax.imshow(M_, aspect="auto", vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(M_.shape[1])); ax.set_xticklabels([c for c in dets if c in piv.columns], rotation=30, fontsize=7)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels([f"{a} {b} {c}" for a, b, c in piv.index], fontsize=6); plt.colorbar(im, ax=ax, label="det100")
    for i in range(M_.shape[0]):
        for jx in range(M_.shape[1]):
            ax.text(jx, i, f"{M_[i, jx]:.2f}", ha="center", va="center", fontsize=5, color="w" if M_[i, jx] < 0.6 else "k")
    ax.set_title("e07 — detection rate within 100 cycles at unified FAR 0.05 (e-process on conformal p)", fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / "e07_det100_heat.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.4)); x = np.arange(len(nuis.detector.unique()))
    for i, nm in enumerate(nuis.fault.unique()):
        sub = nuis[nuis.fault == nm].set_index("detector").reindex(dets)
        ax.bar(x + 0.35 * i - 0.17, sub.alarm_fraction, width=0.33, label=nm)
    ax.axhline(alpha, color="k", ls="--"); ax.set_xticks(x); ax.set_xticklabels(dets, rotation=30, fontsize=7); ax.set_ylabel("false-alarm fraction (100 cycles)"); ax.legend(fontsize=7)
    ax.set_title("e07 — nuisance rows: who alarms without a fault", fontsize=8); fig.tight_layout(); fig.savefig(res_dir / "e07_nuisance.png", dpi=150); plt.close(fig)
    _conclude(res_dir, f"[e07] tables written; e-CUSUM h={h:.3f}; det100 by detector (mean over grid): "
              + ", ".join(f"{dname}: {tab[tab.detector == dname].det100.mean():.2f} (median delay {tab[tab.detector == dname].delay_median.median():.1f})" for dname in dets if (tab.detector == dname).any())
              + " | nuisance alarm fractions: " + "; ".join(f"{r.fault}/{r.detector}: {r.alarm_fraction:.2f}" for r in nuis.itertuples())
              + " | GRU generalisation: " + "; ".join(f"{r.detector}/{r.fault}/{'seen' if r.seen_magnitude else 'unseen'}: det100 {r.det100:.2f}" for r in gen.itertuples()))
    print(f"E07 ALL DONE in {(_dt.datetime.now() - t_all0).total_seconds():.0f}s", flush=True)


if __name__ == "__main__":
    main()
