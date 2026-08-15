#!/usr/bin/env python3
"""Generate the NOMINAL DeLaN dataset (go2_urdf_sym trot, several speeds / terrains, S1 noise) and train the four
per-leg DeLaN nets on the GPU. Weights + history + beta_hat go to $GEOFDI_DATA_ROOT/models/delan/<tag>/.

    python scripts/train_delan.py --gen                       # dataset -> $GEOFDI_DATA_ROOT/data/processed/sim/delan/nominal_v1.npz
    python scripts/train_delan.py --tag full                   # train on all training rollouts (~500k samples / leg)
    python scripts/train_delan.py --tag n50k --n-train 50000   # degraded variants (e06 iii): fewer samples
    python scripts/train_delan.py --tag noise0.5 --label-noise 0.5   # ... or label noise (N m, added to the targets)

Only nominal rollouts are ever used (no faults, no nuisances): the DeLaN model is a nominal model.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from geofdi.dynamics.delan import DeLaNQuadruped, beta_hat, contact_torques_all, leg_arrays, train_leg   # noqa: E402
from geofdi.dynamics.pin_model import Go2Dynamics                                                # noqa: E402
from geofdi.sim.env import SimConfig, rollout                                                    # noqa: E402
from geofdi.sim.pipeline import pmap                                                             # noqa: E402
from geofdi.sim.telemetry import LEGS                                                            # noqa: E402

DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
DATASET = DATA_ROOT / "data" / "processed" / "sim" / "delan" / "nominal_v1.npz"
MODELS = DATA_ROOT / "models" / "delan"
CFG = REPO / "experiments" / "e06_n3_isolability" / "delan_config.yaml"


def _rollout_arrays(sim_cfg, seed):
    cfg = SimConfig(**sim_cfg); cfg.seed = int(seed)
    df, man = rollout(cfg)
    dyn = Go2Dynamics("pin", armature=0.01, damping=0.01, frictionloss=0.2)
    out = {}
    jt_all = contact_torques_all(df, dyn)
    for li, leg in enumerate(LEGS):
        q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt)
        jt = jt_all[:, 3 * li:3 * li + 3]
        out[leg] = {"q": q, "dq": dq, "ddq": ddq, "a": a, "y": tau + jt, "tau": tau, "jt": jt}   # M ddq + h = tau + J^T f
    n = len(df); out["rollout_id"] = np.full(n, seed); out["t"] = df["t"].to_numpy(); out["theta"] = df["theta"].to_numpy()
    return out


def generate(cfg, workers):
    conds = []
    for sp in cfg["data"]["speeds"]:
        for terr in cfg["data"]["terrains"]:
            conds.append((sp, terr))
    args = []; sid = cfg["data"]["seed_base"]
    for rep in range(cfg["data"]["rollouts_per_condition"]):
        for sp, terr in conds:
            s = dict(cfg["sim"]); s["speed"] = float(sp); s["duration_s"] = float(cfg["data"]["duration_s"])
            s.update({"terrain": "flat"} if terr == "flat" else {"terrain": "slope", "slope_deg": float(terr.split(":")[1]), "slope_axis": terr.split(":")[0]})
            args.append((s, sid)); sid += 1
    print(f"[delan-data] {len(args)} rollouts x {cfg['data']['duration_s']} s", flush=True)
    t0 = time.time(); res = pmap(_rollout_arrays, args, workers)
    ds = {}
    for leg in LEGS:
        for k in ("q", "dq", "ddq", "a", "y", "tau", "jt"):
            ds[f"{leg}_{k}"] = np.concatenate([r[leg][k] for r in res]).astype(np.float32)
    ds["rollout_id"] = np.concatenate([r["rollout_id"] for r in res]); ds["t"] = np.concatenate([r["t"] for r in res]); ds["theta"] = np.concatenate([r["theta"] for r in res])
    ds["conditions"] = np.array([json.dumps({"speed": a[0]["speed"], "terrain": a[0]["terrain"], "slope_deg": a[0].get("slope_deg", 0), "slope_axis": a[0].get("slope_axis", ""), "seed": a[1]}) for a in args])
    DATASET.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(DATASET, **ds)
    print(f"[delan-data] wrote {DATASET} ({ds['rollout_id'].shape[0]} samples per leg) in {time.time() - t0:.0f}s", flush=True)


def train(cfg, tag, n_train, label_noise, epochs, seed, device):
    import torch
    ds = np.load(DATASET, allow_pickle=True); rid = ds["rollout_id"]; ids = np.unique(rid)
    rng = np.random.default_rng(seed)
    val_ids = ids[rng.choice(len(ids), size=max(1, int(round(cfg["train"]["val_fraction"] * len(ids)))), replace=False)]
    tr_mask = ~np.isin(rid, val_ids); va_mask = np.isin(rid, val_ids)
    tr_idx = np.where(tr_mask)[0]
    if n_train is not None and n_train < len(tr_idx):
        tr_idx = rng.choice(tr_idx, size=n_train, replace=False)
    out_dir = MODELS / tag; out_dir.mkdir(parents=True, exist_ok=True)
    quad = DeLaNQuadruped.build(hidden=cfg["model"]["hidden"], depth=cfg["model"]["depth"], eps=cfg["model"]["eps"],
                                damping=cfg["model"]["damping"], frictionloss=cfg["model"]["frictionloss"], device=device)
    report = {"tag": tag, "n_train_per_leg": int(len(tr_idx)), "n_val_per_leg": int(va_mask.sum()), "label_noise": label_noise, "epochs": epochs, "seed": seed,
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cuda": torch.version.cuda,
              "torch": torch.__version__, "started": _dt.datetime.now().isoformat(timespec="seconds"), "legs": {}}
    t0 = time.time()
    for leg in LEGS:
        d = {"train": {k: ds[f"{leg}_{k}"][tr_idx] for k in ("q", "dq", "ddq", "a", "y")}, "val": {k: ds[f"{leg}_{k}"][va_mask] for k in ("q", "dq", "ddq", "a", "y")}}
        if label_noise > 0:
            d["train"]["y"] = d["train"]["y"] + rng.normal(0, label_noise, d["train"]["y"].shape).astype(np.float32)
        print(f"  [{tag}] leg {leg}: train {len(tr_idx)} val {int(va_mask.sum())}", flush=True)
        t1 = time.time()
        hist, res = train_leg(quad.nets[leg], d, epochs=epochs, batch=cfg["train"]["batch"], lr=cfg["train"]["lr"], device=device,
                              weight_decay=cfg["train"]["weight_decay"], seed=seed, log=lambda m: print(m, flush=True))
        bh = beta_hat(d["val"]["q"], res, n_bins=cfg["train"]["beta_bins"], quantile=cfg["train"]["beta_quantile"])
        report["legs"][leg] = {"history": hist, "final_val_mse": hist[-1]["val_mse"], "final_val_rmse_per_joint": hist[-1]["val_rmse_per_joint"],
                               "beta_hat": bh, "train_seconds": time.time() - t1}
    quad.meta.update({"tag": tag, "n_train_per_leg": int(len(tr_idx)), "label_noise": label_noise, "epochs": epochs, "seed": seed})
    quad.save(out_dir)
    report["total_seconds"] = time.time() - t0
    report["beta_hat_global_q95_mean_over_legs"] = float(np.mean([report["legs"][l]["beta_hat"]["global_q"] for l in LEGS]))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[delan] {tag}: done in {report['total_seconds']:.0f}s; beta_hat(q95) per leg " + str({l: round(report['legs'][l]['beta_hat']['global_q'], 3) for l in LEGS}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true"); ap.add_argument("--tag", default="full"); ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--label-noise", type=float, default=0.0); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=22); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    if args.gen:
        generate(cfg, args.workers); return
    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train(cfg, args.tag, args.n_train, args.label_noise, args.epochs or cfg["train"]["epochs"], args.seed, device)


if __name__ == "__main__":
    main()
