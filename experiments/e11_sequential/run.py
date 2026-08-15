#!/usr/bin/env python3
"""e11 — sequential-layer unification (Sprint 7 Block S). Go2 go2_urdf_sym.

  arl : the three aggregators (e-process on decimated half-cycles, e-CUSUM, conformal-CUSUM) on the R- half-cycle mirror
        score at two ARL0 targets {1/alpha, 5/alpha}; measured ARL0 on nominal streams and the detection delay on
        mid-magnitude faults -> ARL0-vs-delay trade-off curve.
  compl : two-channel complementarity timeline (paper figure): R- (half-cycle e-process on the equivariant residual) and
        R+ (residual magnitude conformal e-process) alarm times for single-leg / bilateral-mirror / lateral payload /
        symmetric drift.

    python experiments/e11_sequential/run.py --stage arl|compl|all [--run-id ID] [--quick] [--workers N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import copy
import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from geofdi.detect.evalue import p_to_e
from geofdi.detect.monitors import conformal_pvalues, eprocess_alarm
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.detect.sequential import ConformalCusum, ECusum, EProcess, calibrate_threshold, calibration_scale, half_cycles, mirror_scores
from geofdi.dynamics.delan import contact_torques_all, leg_arrays
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import residual_manifest
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import pmap
from geofdi.sim.telemetry import JOINTS, LEGS, z_channel_names

EXP_NAME = "e11_sequential"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
_MODELS = {}
_RES_REP = C2Rep(residual_manifest(include_base=False))


def _sim(cfg, seed, **over):
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        s[k] = v
    return s


def _conclude(res_dir, line):
    print(line, flush=True); (res_dir / "conclusions.txt").open("a").write(line + "\n")


def _load_eq(tag):
    import torch
    from geofdi.dynamics.delan_equiv import load_delan
    if tag not in _MODELS:
        _MODELS[tag] = load_delan(DATA_ROOT / "models" / "delan" / tag, device="cuda" if torch.cuda.is_available() else "cpu")
    return _MODELS[tag]


def _worker(sim_cfg, fault, K_cal, K_post, N, df0, oc, need_arrays=False):
    cfg = SimConfig(**dict(sim_cfg, faults=[dict(fault, t_onset=(K_cal + df0) * 0.5)] if fault else [])); cfg.duration_s = (K_cal + K_post + df0 + 2) * 0.5
    df, man = rollout(cfg); chans = z_channel_names(man); rep = C2Rep(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=df0); Z = Z[:K_cal + K_post]
    H = half_cycles(Z); scale = calibration_scale(H[:2 * K_cal], rep); s = mirror_scores(H, rep, scale)
    out = {"s_cal": s[:2 * K_cal - 1].astype(np.float32), "s_mon": s[2 * K_cal - 1:].astype(np.float32), "K": int(Z.shape[0])}
    if need_arrays:
        dyn = Go2Dynamics(oc["backend"], armature=oc["armature"], damping=oc["damping"], frictionloss=oc["frictionloss"])
        jt = contact_torques_all(df, dyn); legs = {}
        for li, leg in enumerate(LEGS):
            q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt); legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32), "y": (tau + jt[:, 3 * li:3 * li + 3]).astype(np.float32)}
        r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=oc["cutoff_hz"], torque=oc["torque"])[:, 6:]
        Zr_an, _ = registered_residuals(df, r, N=N, drop_first=df0)
        out["arrays"] = {"legs": legs, "theta": df["theta"].to_numpy(), "t": df["t"].to_numpy(), "N": N, "drop_first": df0}
        out["Zr_an"] = Zr_an[:out["K"]].astype(np.float32)
        out["Z"] = Z.astype(np.float32); out["man"] = man; out["chans"] = chans
        Zc, _ = register_cycles(df, [f"qref_{l}_{j}" for l in LEGS for j in JOINTS], N=N, drop_first=df0); out["Zq"] = Zc[:out["K"]].astype(np.float32)
    return out


# ------------------------------------------------------------------------------------ ARL / delay
def stage_arl(cfg, res_dir, quick=False):
    ac = cfg["arl"]; R = 8 if quick else ac["R"]; K_cal, K_post = ac["K_cal"], ac["K_post"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]; alpha = 0.05
    if quick:
        K_cal = 100
    nom = pmap(_worker, [(_sim(cfg, ac["seed_base"] + 900000 + r), None, K_cal, 400 if not quick else 100, N, df0, oc) for r in range(max(12, R))], cfg["workers"])
    n_thr = len(nom) // 2
    rows = []; trade = []
    for arl0 in ([20] if quick else ac["arl0_targets"]):
        far = 1.0 / arl0                                                   # target false-alarm-within-horizon over a horizon ~ arl0
        horizon_half = 2 * arl0
        dets = {}
        # decimated e-process: no threshold, but we report its ARL0 empirically; e-CUSUM / conformal-CUSUM: calibrate h for the target
        ec = ECusum(alpha); ec.calibrate(np.concatenate([r["s_cal"] for r in nom[:n_thr]])); ec.h = calibrate_threshold(ec, [r["s_mon"] for r in nom[:n_thr]], horizon=horizon_half, far=far, n_boot=800, rng=np.random.default_rng(1))
        cc = ConformalCusum(alpha); cc.calibrate(np.concatenate([r["s_cal"] for r in nom[:n_thr]])); cc.h = calibrate_threshold(cc, [r["s_mon"] for r in nom[:n_thr]], horizon=horizon_half, far=far, n_boot=800, rng=np.random.default_rng(2))
        dets = {"eprocess_decim": EProcess(1.0 / arl0), "ecusum": ec, "conformal_cusum": cc}
        # nominal ARL0 (censored at K_post_nom) on held-out nominal runs
        for name, d in dets.items():
            arls = []
            for r in nom[n_thr:]:
                d.calibrate(r["s_cal"][::2] if name == "eprocess_decim" else r["s_cal"])
                s = r["s_mon"][::2] if name == "eprocess_decim" else r["s_mon"]; S, al = d.run(s)
                step = 1.0 if name == "eprocess_decim" else 0.5
                arls.append((al + 1) * step if al is not None else len(s) * step)
            arl_meas = float(np.mean(arls)); alarm_frac = float(np.mean([a < (len(nom[n_thr]["s_mon"]) * (1.0 if name == "eprocess_decim" else 0.5)) for a in arls]))
            rows.append({"arl0_target": arl0, "detector": name, "kind": "nominal", "arl0_measured_cycles": arl_meas, "alarm_frac": alarm_frac, "h": getattr(d, "h", np.nan)})
        # fault delays
        for fspec in (ac["faults"][:1] if quick else ac["faults"]):
            fres = pmap(_worker, [(_sim(cfg, ac["seed_base"] + 1000 + r), {k: v for k, v in fspec.items() if k not in ("name",)}, K_cal, K_post, N, df0, oc) for r in range(R)], cfg["workers"])
            for name, d in dets.items():
                dl = []
                for r in fres:
                    d.calibrate(r["s_cal"][::2] if name == "eprocess_decim" else r["s_cal"]); s = r["s_mon"][::2] if name == "eprocess_decim" else r["s_mon"]; S, al = d.run(s)
                    step = 1.0 if name == "eprocess_decim" else 0.5; dl.append(np.nan if al is None else (al + 1) * step)
                dl = np.array(dl)
                rows.append({"arl0_target": arl0, "detector": name, "kind": fspec["name"], "delay_median_cycles": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan, "det_rate": float(np.mean(~np.isnan(dl)))})
                trade.append({"arl0_target": arl0, "arl0_measured": float([x for x in rows if x["detector"] == name and x["kind"] == "nominal" and x["arl0_target"] == arl0][0]["arl0_measured_cycles"]),
                              "detector": name, "fault": fspec["name"], "delay_median_cycles": float(np.nanmedian(dl)) if np.isfinite(dl).any() else np.nan})
        print(f"  [arl] target {arl0} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e11_arl.csv", index=False)
    tr = pd.DataFrame(trade); tr.to_csv(res_dir / "e11_tradeoff.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, mk in (("eprocess_decim", "o"), ("ecusum", "s"), ("conformal_cusum", "^")):
        for fault, c in zip(tr.fault.unique(), ("C0", "C1", "C2", "C3")):
            sub = tr[(tr.detector == name) & (tr.fault == fault)].sort_values("arl0_measured")
            ax.plot(sub.arl0_measured, sub.delay_median_cycles, mk + "-", color=c, label=f"{name} / {fault}" if name == "eprocess_decim" else None, ms=6, alpha=0.8)
    ax.set_xlabel("measured nominal ARL0 [cycles]"); ax.set_ylabel("median detection delay [cycles]"); ax.set_title("e11 — ARL0 vs delay trade-off (R⁻ half-cycle sequential detectors)", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(res_dir / "e11_tradeoff.png", dpi=140); plt.close(fig)
    nom_rows = tab[tab.kind == "nominal"]
    _conclude(res_dir, "[e11-arl] measured nominal ARL0 (cycles) / target: " + "; ".join(f"{r.detector}@{r.arl0_target}: {r.arl0_measured_cycles:.0f} (alarm frac {r.alarm_frac:.2f})" for r in nom_rows.itertuples())
              + " | delay: " + "; ".join(f"{r.detector}@{r.arl0_target}/{r.kind}: {r.delay_median_cycles:.1f}" for r in tab[tab.kind != "nominal"].itertuples()))
    return tab


# ------------------------------------------------------------------------------------ complementarity
def stage_compl(cfg, res_dir, quick=False):
    cc = cfg["complementarity"]; R = 6 if quick else cc["R"]; K_cal, K_post = cc["K_cal"], (40 if quick else cc["K_post"]); N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]; alpha = 0.05
    eq = cfg["delan"]["equivariant"]; _load_eq(eq); rows = []; timelines = {}
    onset_s = (K_cal + df0) * 0.5                            # fault/nuisance onset time (s), K_cal cycles of 0.5 s
    for ci, (cond, over) in enumerate(cc["conditions"].items()):
        fault = over["faults"][0] if "faults" in over else None
        # bake any nuisance (with its onset) into the sim config so the worker is a top-level picklable callable
        nuis = [dict(f, t_onset=onset_s) for f in over["nuisance"]] if "nuisance" in over else None
        args = [(_sim(cfg, cc["seed_base"] + 1000 * ci + r, **({"nuisance": nuis} if nuis else {})),
                 fault, K_cal, K_post, N, df0, oc, True) for r in range(R)]
        res = pmap(_worker, args, cfg["workers"])
        rm_alarm = []; rp_alarm = []; rm_tl = []; rp_tl = []
        for r_i, o in enumerate(res):
            # R- half-cycle e-process (decimated) on the equivariant residual
            arr = o["arrays"]; T = len(arr["theta"]); rr = np.zeros((T, 12), dtype=np.float32); dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]})
            for li, leg in enumerate(LEGS):
                L = arr["legs"][leg]; rr[:, 3 * li:3 * li + 3] = L["y"] - _MODELS[eq].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
            Zr, _ = registered_residuals(dfr, rr, N=N, drop_first=df0); Zr = Zr[:o["K"]]
            H = half_cycles(Zr); scale = calibration_scale(H[:2 * K_cal], _RES_REP); s = mirror_scores(H, _RES_REP, scale)
            p_hc = conformal_pvalues(s[:2 * K_cal - 1:2], s[2 * K_cal - 1::2])       # decimated
            E, al = eprocess_alarm(p_hc, alpha, start=0); rm_alarm.append(np.nan if al is None else al + 1); rm_tl.append(np.log(np.maximum(1e-3, p_hc)))
            # R+ residual magnitude conformal e-process (per cycle)
            sc = residual_scores(Zr); pr = conformal_pvalues(sc[:K_cal], sc[K_cal:]); E, al = eprocess_alarm(pr, alpha, start=0); rp_alarm.append(np.nan if al is None else al + 1); rp_tl.append(np.log(np.maximum(1e-3, pr)))
        rows.append({"condition": cond, "R": R, "Rminus_alarm_frac": float(np.mean(~np.isnan(rm_alarm))), "Rminus_delay_median": float(np.nanmedian(rm_alarm)) if np.isfinite(rm_alarm).any() else np.nan,
                     "Rplus_alarm_frac": float(np.mean(~np.isnan(rp_alarm))), "Rplus_delay_median": float(np.nanmedian(rp_alarm)) if np.isfinite(rp_alarm).any() else np.nan})
        L = min(min(len(x) for x in rm_tl), min(len(x) for x in rp_tl))
        timelines[cond] = (np.mean([x[:L] for x in rm_tl], axis=0), np.mean([x[:L] for x in rp_tl], axis=0))
        print(f"  [compl] {cond} done", flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e11_complementarity.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    conds = list(cc["conditions"].keys()); fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.4), sharey=True)
    for ax, cond in zip(axes, conds):
        rm, rp = timelines[cond]
        ax.plot(np.arange(len(rm)) - K_cal // 2, rm, "C0-", label="R⁻ (½-cycle, Π⁻r)"); ax.plot(np.arange(len(rp)) - K_cal, rp, "C2-", label="R⁺ (residual magnitude)")
        ax.axvline(0, color="k", ls="--", lw=1); ax.axhline(np.log(alpha), color="0.5", ls=":", lw=1); ax.set_title(cond, fontsize=8); ax.set_xlabel("cycles after onset")
    axes[0].set_ylabel("mean log p"); axes[0].legend(fontsize=6)
    fig.suptitle("e11 — two-channel complementarity: R⁻ sees the antisymmetric (single/lateral), R⁺ the magnitude (bilateral/drift)", fontsize=9); fig.tight_layout()
    fig.savefig(res_dir / "e11_complementarity.png", dpi=140); plt.close(fig)
    _conclude(res_dir, "[e11-compl] R-/R+ alarm fraction (delay): " + "; ".join(f"{r.condition}: R- {r.Rminus_alarm_frac:.2f} ({r.Rminus_delay_median}) / R+ {r.Rplus_alarm_frac:.2f} ({r.Rplus_delay_median})" for r in tab.itertuples()))
    return tab


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["arl", "compl", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml")); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    if a.workers:
        cfg["workers"] = a.workers
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True); (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for s in (["arl", "compl"] if a.stage == "all" else [a.stage]):
        t0 = _dt.datetime.now(); {"arl": stage_arl, "compl": stage_compl}[s](cfg, res_dir, quick=a.quick); print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E11 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
