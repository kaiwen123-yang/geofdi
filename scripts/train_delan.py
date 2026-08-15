#!/usr/bin/env python3
"""Generate the NOMINAL DeLaN dataset (go2_urdf_sym trot, several speeds / terrains, S1 noise) and train the four
per-leg DeLaN nets on the GPU. Weights + history + beta_hat go to $GEOFDI_DATA_ROOT/models/delan/<tag>/.

    python scripts/train_delan.py --gen                       # dataset -> $GEOFDI_DATA_ROOT/data/processed/sim/delan/nominal_v1.npz
    python scripts/train_delan.py --tag full                   # train on all training rollouts (~500k samples / leg)
    python scripts/train_delan.py --tag n50k --n-train 50000   # degraded variants (e06 iii): fewer samples
    python scripts/train_delan.py --tag noise0.5 --label-noise 0.5   # ... or label noise (N m, added to the targets)
    python scripts/train_delan.py --equivariant --tag equiv_full      # Sprint 6: mirror weight sharing (2 templates F/H)
    python scripts/train_delan.py --equivariant --n-templates 1 --tag equiv1_full   # ablation: one template for all legs
    python scripts/train_delan.py --defect --tag full                 # delta_f^(0.95) of an existing model on the val split
    python scripts/train_delan.py --gen-weld / --weld --tag weld_plain_v1 [--equivariant]   # welded-trunk dataset / models

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
from geofdi.dynamics.delan_equiv import EquivariantDeLaN, equivariance_defect, load_delan, train_equivariant   # noqa: E402
from geofdi.dynamics.pin_model import Go2Dynamics                                                # noqa: E402
from geofdi.sim.env import SimConfig, rollout                                                    # noqa: E402
from geofdi.sim.pipeline import pmap                                                             # noqa: E402
from geofdi.sim.telemetry import LEGS                                                            # noqa: E402

DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
DATASET = DATA_ROOT / "data" / "processed" / "sim" / "delan" / "nominal_v1.npz"
DATASET_WELD = DATA_ROOT / "data" / "processed" / "sim" / "delan" / "weld_v1.npz"
MODELS = DATA_ROOT / "models" / "delan"
MODELS_WELD = DATA_ROOT / "models" / "delan_weld"
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


def _weld_rollout_arrays(sim_cfg, seed):
    """Welded trunk (leg = fixed-base 3-dof arm): contact term is zero, y = tau_cmd."""
    cfg = SimConfig(**sim_cfg); cfg.seed = int(seed); cfg.weld_base = True
    df, man = rollout(cfg)
    out = {}
    for leg in LEGS:
        q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt)
        out[leg] = {"q": q, "dq": dq, "ddq": ddq, "a": a, "y": tau, "tau": tau, "jt": np.zeros_like(tau)}
    n = len(df); out["rollout_id"] = np.full(n, seed); out["t"] = df["t"].to_numpy(); out["theta"] = df["theta"].to_numpy()
    return out


def generate_weld(cfg, workers, rollouts=40, duration_s=25.0, seed_base=53000):
    """The e06 stage_weld nominal training set (same seeds), stored once so that plain and equivariant weld models train
    on the identical split."""
    args = []
    for i in range(rollouts):
        s = dict(cfg["sim"]); s["duration_s"] = float(duration_s); args.append((s, seed_base + i))
    print(f"[delan-weld-data] {len(args)} welded rollouts x {duration_s} s", flush=True)
    t0 = time.time(); res = pmap(_weld_rollout_arrays, args, workers)
    ds = {}
    for leg in LEGS:
        for k in ("q", "dq", "ddq", "a", "y", "tau", "jt"):
            ds[f"{leg}_{k}"] = np.concatenate([r[leg][k] for r in res]).astype(np.float32)
    ds["rollout_id"] = np.concatenate([r["rollout_id"] for r in res]); ds["t"] = np.concatenate([r["t"] for r in res]); ds["theta"] = np.concatenate([r["theta"] for r in res])
    ds["conditions"] = np.array([json.dumps({"weld_base": True, "seed": a[1]}) for a in args])
    DATASET_WELD.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(DATASET_WELD, **ds)
    print(f"[delan-weld-data] wrote {DATASET_WELD} ({ds['rollout_id'].shape[0]} samples per leg) in {time.time() - t0:.0f}s", flush=True)


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


def _split(ds, cfg, seed, n_train):
    """Rollout-level train/val split (identical for plain and equivariant models of the same seed) and the optional
    per-leg training subsample (the same row indices for every leg)."""
    rid = ds["rollout_id"]; ids = np.unique(rid)
    rng = np.random.default_rng(seed)
    val_ids = ids[rng.choice(len(ids), size=max(1, int(round(cfg["train"]["val_fraction"] * len(ids)))), replace=False)]
    tr_mask = ~np.isin(rid, val_ids); va_mask = np.isin(rid, val_ids)
    tr_idx = np.where(tr_mask)[0]
    if n_train is not None and n_train < len(tr_idx):
        tr_idx = rng.choice(tr_idx, size=n_train, replace=False)
    return tr_idx, va_mask, rng


def train(cfg, tag, n_train, label_noise, epochs, seed, device, equivariant=False, n_templates=2, weld=False):
    import torch
    ds = np.load(DATASET_WELD if weld else DATASET, allow_pickle=True)
    tr_idx, va_mask, rng = _split(ds, cfg, seed, n_train)
    out_dir = (MODELS_WELD if weld else MODELS) / tag; out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(hidden=cfg["model"]["hidden"], depth=cfg["model"]["depth"], eps=cfg["model"]["eps"], damping=cfg["model"]["damping"],
              frictionloss=cfg["model"]["frictionloss"], device=device)
    quad = EquivariantDeLaN.build(n_templates=n_templates, **kw) if equivariant else DeLaNQuadruped.build(**kw)
    report = {"tag": tag, "equivariant": bool(equivariant), "n_templates": (n_templates if equivariant else 4), "weld": bool(weld),
              "n_train_per_leg": int(len(tr_idx)), "n_val_per_leg": int(va_mask.sum()), "label_noise": label_noise, "epochs": epochs, "seed": seed,
              "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cuda": torch.version.cuda,
              "torch": torch.__version__, "started": _dt.datetime.now().isoformat(timespec="seconds"), "legs": {}}
    t0 = time.time()
    if equivariant:
        data = {}
        for leg in LEGS:
            d = {"train": {k: ds[f"{leg}_{k}"][tr_idx] for k in ("q", "dq", "ddq", "a", "y")}, "val": {k: ds[f"{leg}_{k}"][va_mask] for k in ("q", "dq", "ddq", "a", "y")}}
            if label_noise > 0:
                d["train"]["y"] = d["train"]["y"] + rng.normal(0, label_noise, d["train"]["y"].shape).astype(np.float32)
            data[leg] = d
        rep = train_equivariant(quad, data, epochs=epochs, batch=cfg["train"]["batch"], lr=cfg["train"]["lr"], device=device,
                                weight_decay=cfg["train"]["weight_decay"], seed=seed, log=lambda m: print(m, flush=True),
                                n_bins=cfg["train"]["beta_bins"], quantile=cfg["train"]["beta_quantile"])
        report["templates"] = rep["templates"]; report["legs"] = rep["legs"]
        for leg in LEGS:
            report["legs"][leg]["train_seconds"] = None
        quad.meta.update({"tag": tag, "n_train_per_leg": int(len(tr_idx)), "label_noise": label_noise, "epochs": epochs, "seed": seed, "weld": bool(weld)})
        quad.save(out_dir)
        report["total_seconds"] = time.time() - t0
        report["beta_hat_global_q95_mean_over_legs"] = float(np.mean([report["legs"][l]["beta_hat"]["global_q"] for l in LEGS]))
        report["defect"] = _defect_report(quad, ds, va_mask)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))
        print(f"[delan] {tag} (equivariant, {n_templates} templates): done in {report['total_seconds']:.0f}s; val rmse per leg "
              + str({l: np.round(report['legs'][l]['final_val_rmse_per_joint'], 3).tolist() for l in LEGS})
              + f"; delta_f q95 {report['defect']['q95']:.2e}", flush=True)
        return
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
    quad.meta.update({"tag": tag, "n_train_per_leg": int(len(tr_idx)), "label_noise": label_noise, "epochs": epochs, "seed": seed, "weld": bool(weld)})
    quad.save(out_dir)
    report["total_seconds"] = time.time() - t0
    report["beta_hat_global_q95_mean_over_legs"] = float(np.mean([report["legs"][l]["beta_hat"]["global_q"] for l in LEGS]))
    report["defect"] = _defect_report(quad, ds, va_mask)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[delan] {tag}: done in {report['total_seconds']:.0f}s; beta_hat(q95) per leg " + str({l: round(report['legs'][l]['beta_hat']['global_q'], 3) for l in LEGS})
          + f"; delta_f q95 {report['defect']['q95']:.3f}", flush=True)


def _defect_report(model, ds, va_mask, max_samples: int = 50000, seed: int = 0) -> dict:
    """delta_f quantiles on nominal VALIDATION samples: for each mirror pair the samples of the first leg are the inputs z
    of f_leg and (S q, S dq, S ddq, E a) the inputs of f_partner (Part 2, Definition def:defect, quantile version)."""
    rng = np.random.default_rng(seed); idx = np.where(va_mask)[0]
    if len(idx) > max_samples:
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    out = {"n_samples": int(len(idx)), "pairs": {}}
    arrs = []
    for leg, partner in (("LF", "RF"), ("LH", "RH")):
        d = equivariance_defect(model, ds[f"{leg}_q"][idx], ds[f"{leg}_dq"][idx], ds[f"{leg}_ddq"][idx], ds[f"{leg}_a"][idx], pairs=[(leg, partner)])
        out["pairs"][f"{leg}-{partner}"] = {"q95": d["q95"], "q50": d["q50"], "max": d["max"], "rms": d["rms"]}
        arrs.append(d[f"{leg}-{partner}"])
    d_all = np.concatenate(arrs)                      # pooled over both pairs (same sample count per pair)
    out.update({"q95": float(np.quantile(d_all, 0.95)), "q50": float(np.quantile(d_all, 0.5)), "max": float(d_all.max()), "rms": float(np.sqrt(np.mean(d_all ** 2)))})
    return out


def defect_only(tag, seed, weld=False, cfg=None):
    """Compute delta_f for an existing model (plain or equivariant) on the validation split of its dataset."""
    ds = np.load(DATASET_WELD if weld else DATASET, allow_pickle=True)
    _, va_mask, _ = _split(ds, cfg, seed, None)
    mdir = (MODELS_WELD if weld else MODELS) / tag
    model = load_delan(mdir, device="cpu")
    rep = _defect_report(model, ds, va_mask)
    p = mdir / "report.json"; r = json.loads(p.read_text()) if p.exists() else {"tag": tag}
    r["defect"] = rep; p.write_text(json.dumps(r, indent=1))
    print(f"[delan] {tag}: delta_f q95 {rep['q95']:.4f} q50 {rep['q50']:.4f} max {rep['max']:.3f} rms {rep['rms']:.4f} (n={rep['n_samples']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true"); ap.add_argument("--tag", default="full"); ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--label-noise", type=float, default=0.0); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=22); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--equivariant", action="store_true", help="mirror weight sharing (Sprint 6)")
    ap.add_argument("--n-templates", type=int, default=2, choices=[1, 2])
    ap.add_argument("--defect", action="store_true", help="only compute delta_f of an existing model tag")
    ap.add_argument("--gen-weld", action="store_true"); ap.add_argument("--weld", action="store_true", help="use the welded-trunk dataset / model dir")
    ap.add_argument("--weld-rollouts", type=int, default=40); ap.add_argument("--weld-duration", type=float, default=25.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    if args.gen:
        generate(cfg, args.workers); return
    if args.gen_weld:
        generate_weld(cfg, args.workers, rollouts=args.weld_rollouts, duration_s=args.weld_duration); return
    if args.defect:
        defect_only(args.tag, args.seed, weld=args.weld, cfg=cfg); return
    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train(cfg, args.tag, args.n_train, args.label_noise, args.epochs or cfg["train"]["epochs"], args.seed, device,
          equivariant=args.equivariant, n_templates=args.n_templates, weld=args.weld)


if __name__ == "__main__":
    main()
