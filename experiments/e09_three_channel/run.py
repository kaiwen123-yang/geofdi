#!/usr/bin/env python3
"""e09 — three-channel isolation + the LH-KFE friction diagnosis/fix (Sprint 7 Block I).
Pre-registration: docs/protocol/e09_preregistration.md (committed before this run).

  diagnose : reproduce the e13c analytic-row LH-KFE friction inversion and the fix (whole-leg energy score vs joint-row
             |mean shift|) -> diagnosis.md.
  iso      : 9 classes x R runs; reading vector = R- (equivariant-DeLaN residual) + joint residual rows (fixed left/right
             rule) + base momentum-residual rows; confusion for analytic-row and equivariant-row sources side by side;
             per-class three-row readout figure; contact-wrench sensitivity (+-10/20 %).

    python experiments/e09_three_channel/run.py --stage diagnose|iso|all [--run-id ID] [--quick]
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

from geofdi.detect.monitors import MirrorMonitor, calibrate_ecusum_threshold, ecusum
from geofdi.detect.rplus import registered_residuals, residual_scores
from geofdi.dynamics.delan import contact_torques_all, leg_arrays
from geofdi.dynamics.momentum_observer import run_observer
from geofdi.dynamics.pin_model import Go2Dynamics
from geofdi.groups.c2 import C2Rep
from geofdi.isolation.three_channel import decide, readout, resolve_left_right
from geofdi.phase.registration import register_cycles
from geofdi.residuals.mirror_pairs import BASE_COLS, RES_COLS, residual_manifest
from geofdi.sim.env import SimConfig, rollout
from geofdi.sim.pipeline import pmap
from geofdi.sim.telemetry import JOINTS, LEGS, z_channel_names

EXP_NAME = "e09_three_channel"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ["GEOFDI_DATA_ROOT"])
QREF = [f"qref_{l}_{j}" for l in LEGS for j in JOINTS]
_MODELS = {}
_RES_REP = C2Rep(residual_manifest(include_base=False))
_RES_REP_BASE = C2Rep(residual_manifest(include_base=True))


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


# ------------------------------------------------------------------------------------ diagnosis
def _diag_one(sim_cfg, fault, K_cal, K_post, N, df0, oc):
    cfg = SimConfig(**dict(sim_cfg, faults=[fault])); cfg.duration_s = (K_cal + K_post + df0 + 2) * 0.5
    df, man = rollout(cfg); dyn = Go2Dynamics(oc["backend"], armature=oc["armature"], damping=oc["damping"], frictionloss=oc["frictionloss"])
    r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=oc["cutoff_hz"], torque=oc["torque"])[:, 6:]
    Zr, _ = registered_residuals(df, r, N=N, drop_first=df0); Zr = Zr[:K_cal + K_post]
    sl = residual_scores(Zr, per_leg=True); dev_energy = (sl[K_cal:].mean(0) - sl[:K_cal].mean(0)) / (sl[:K_cal].std(0) + 1e-12)
    shift = (Zr[K_cal:].mean(axis=(0, 2)) - Zr[:K_cal].mean(axis=(0, 2)))
    return dev_energy, shift


def stage_diagnose(cfg, res_dir, quick=False):
    R = 6 if quick else 16; K_cal, K_post = cfg["iso"]["K_cal"], cfg["iso"]["K_post"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]; t_on = (K_cal + df0) * 0.5
    lines = ["# e09 diagnosis — the e13c analytic-row LH-KFE friction left/right inversion", ""]
    faults = {"friction_LH-KFE": dict(type="friction_scale", leg="LH", joint="KFE", magnitude=1.0, t_onset=t_on),
              "gain_LH-KFE": dict(type="actuator_gain", leg="LH", joint="KFE", magnitude=-0.2, t_onset=t_on)}
    rows = []
    for name, fault in faults.items():
        res = pmap(_diag_one, [(_sim(cfg, cfg["iso"]["seed_base"] + 300000 + r), fault, K_cal, K_post, N, df0, oc) for r in range(R)], cfg["workers"])
        dev = np.array([r[0] for r in res]); shift = np.array([r[1] for r in res])
        # old rule: max(signed energy deviation) over the hind pair; new rule: max |joint-row shift|
        old_pick = ["LH" if d[2] > d[3] else "RH" for d in dev]; kfe_abs = np.abs(shift[:, [2, 5, 8, 11]])
        new_pick = ["LH" if a[2] > a[3] else "RH" for a in kfe_abs]
        old_lh = old_pick.count("LH"); new_lh = new_pick.count("LH")
        rows.append({"fault": name, "R": R, "old_rule_LH_count": old_lh, "new_rule_LH_count": new_lh, "dev_energy_mean_LFRFLHRH": dev.mean(0).round(3).tolist(), "kfe_row_absshift_mean_LFRFLHRH": kfe_abs.mean(0).round(4).tolist()})
        lines.append(f"- **{name}** (truth LH): old energy-deviation rule picks LH {old_lh}/{R}, new |KFE-row shift| rule picks LH {new_lh}/{R}; "
                     f"per-leg energy deviation mean {dev.mean(0).round(3).tolist()} (friction damps the leg -> its energy DROPS -> max picks the wrong leg); |KFE row shift| mean {kfe_abs.mean(0).round(4).tolist()}")
    pd.DataFrame(rows).to_csv(res_dir / "e09_diagnosis.csv", index=False)
    lines += ["", "**Root cause**: `residual_scores(per_leg=True)` is a residual ENERGY score; a friction increase damps the",
              "faulty leg, so its energy DECREASES (deviation goes negative), and `max(signed deviation)` selects a non-faulty",
              "leg. **Fix**: resolve left/right by the |mean shift| of the pair's joint residual row (monotone in the fault for",
              "gain, bias and friction). This is what `isolation.three_channel.resolve_left_right` does."]
    (res_dir / "diagnosis.md").write_text("\n".join(lines) + "\n")
    _conclude(res_dir, "[e09-diagnose] " + " | ".join(f"{r['fault']}: old rule LH {r['old_rule_LH_count']}/{r['R']}, fixed rule LH {r['new_rule_LH_count']}/{r['R']}" for r in rows))
    return rows


# ------------------------------------------------------------------------------------ isolation
def _iso_one(sim_cfg, over, K_cal, K_post, N, df0, oc, contact_scales, need_arrays):
    cfg = SimConfig(**dict(sim_cfg, **{k: v for k, v in over.items()})); cfg.duration_s = (K_cal + K_post + df0 + 2) * 0.5
    df, man = rollout(cfg); chans = z_channel_names(man)
    Z, meta = register_cycles(df, chans, N=N, drop_first=df0); Z = Z[:K_cal + K_post]
    dyn = Go2Dynamics(oc["backend"], armature=oc["armature"], damping=oc["damping"], frictionloss=oc["frictionloss"])
    out = {"K": int(Z.shape[0]), "Z": Z.astype(np.float32), "man": man, "chans": chans}
    # analytic residual (joint + base) at several contact-wrench scales
    for cs in contact_scales:
        r = run_observer(df, dyn, dt=cfg.ctrl_dt, cutoff_hz=oc["cutoff_hz"], torque=oc["torque"], contact_scale=1.0 + cs)
        Zr, _ = registered_residuals(df, r[:, 6:], N=N, drop_first=df0)
        dfb = df[["t", "theta"]].copy()
        for i, c in enumerate(BASE_COLS):
            dfb[c] = r[:, i]
        Zb, _ = register_cycles(dfb, BASE_COLS, N=N, drop_first=df0)
        out[f"Zr_an_{cs}"] = Zr[:out["K"]].astype(np.float32); out[f"Zb_{cs}"] = Zb[:out["K"]].astype(np.float32)
    if need_arrays:
        jt = contact_torques_all(df, dyn); legs = {}
        for li, leg in enumerate(LEGS):
            q, dq, ddq, a, tau = leg_arrays(df, leg, dt=cfg.ctrl_dt); legs[leg] = {"q": q.astype(np.float32), "dq": dq.astype(np.float32), "ddq": ddq.astype(np.float32), "a": a.astype(np.float32), "y": (tau + jt[:, 3 * li:3 * li + 3]).astype(np.float32)}
        out["arrays"] = {"legs": legs, "theta": df["theta"].to_numpy(), "t": df["t"].to_numpy(), "N": N, "drop_first": df0}
    return out


def _eq_residual(out, tag):
    arr = out.get("arrays")
    if arr is None:
        return None
    T = len(arr["theta"]); r = np.zeros((T, 12), dtype=np.float32); dfr = pd.DataFrame({"theta": arr["theta"], "t": arr["t"]})
    for li, leg in enumerate(LEGS):
        L = arr["legs"][leg]; r[:, 3 * li:3 * li + 3] = L["y"] - _MODELS[tag].predict(leg, L["q"], L["dq"], L["ddq"], L["a"])
    Zr, _ = registered_residuals(dfr, r, N=arr["N"], drop_first=arr["drop_first"])
    return Zr[:out["K"]].astype(np.float32)


def stage_iso(cfg, res_dir, quick=False):
    iso = cfg["iso"]; R = 6 if quick else iso["R"]; K_cal, K_post = iso["K_cal"], iso["K_post"]; N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; oc = cfg["observer"]; alpha = cfg["detect"]["alpha"]; t_on = (K_cal + df0) * 0.5
    eq = cfg["delan"]["equivariant"]; _load_eq(eq); cs_list = [0.0] if quick else iso["contact_sensitivity"]
    classes = iso["classes"] if not quick else iso["classes"][:4] + iso["classes"][-1:]
    # nominal e-CUSUM thresholds per source (equivariant residual, analytic residual)
    conf_rows = []; sens_rows = []; readout_store = {}
    nominal_pw = {"eq": [], "an": []}
    for ci, cl in enumerate(classes):
        over = {}
        for key in ("faults", "nuisance"):
            if key in cl:
                over[key] = [dict(f, t_onset=t_on) for f in cl[key]]
        args = [(_sim(cfg, iso["seed_base"] + 1000 * ci + r), over, K_cal, K_post, N, df0, oc, cs_list, True) for r in range(R)]
        outs = pmap(_iso_one, args, cfg["workers"])
        for r_i, o in enumerate(outs):
            Zr_eq = _eq_residual(o, eq)
            rep_raw = C2Rep(o["man"])
            # R- alarm on the equivariant residual (5-cycle windows, e-process); nominal thresholds pooled below
            for src, Zr, rep in (("eq", Zr_eq, _RES_REP), ("an", o["Zr_an_0.0"], _RES_REP)):
                mm = MirrorMonitor(rep, window=cfg["detect"]["window_rminus"], M=cfg["detect"]["M"], statistic="paired_energy", alpha=alpha)
                pw = mm.window_pvalues(Zr, seed=iso["seed_base"] + 90000 + 1000 * ci + r_i)
                if cl["name"] == "nominal":
                    nominal_pw[src].append(pw[:K_cal // 5])
                o[f"pw_{src}"] = pw
            o["Zr_eq"] = Zr_eq; o.pop("arrays", None)
        readout_store[cl["name"]] = (cl, outs)
        print(f"  [iso] {cl['name']} done", flush=True)
    h = {src: calibrate_ecusum_threshold(nominal_pw[src], K_post // 5, far=alpha, n_boot=800, rng=np.random.default_rng(4)) for src in nominal_pw}
    w0 = K_cal // 5
    # decisions
    for name, (cl, outs) in readout_store.items():
        for r_i, o in enumerate(outs):
            for src, Zr_key in (("analytic_rows", "Zr_an_0.0"), ("equiv_rows", "Zr_eq")):
                Zr = o[Zr_key]; Zb = o["Zb_0.0"]
                S, al = ecusum(o["pw_eq" if src == "equiv_rows" else "pw_an"], h["eq" if src == "equiv_rows" else "an"], start=w0); alarmed = al is not None
                rd = readout(o["Z"], Zr, Zb, C2Rep(o["man"]), _RES_REP, K_cal, o["chans"], RES_COLS, use_residual_for_rminus=True)
                label, conf, why = decide(rd, alarmed, base_z_thresh=iso["base_z_thresh"], share_thresh=iso["share_thresh"])
                conf_rows.append({"class": cl["name"], "truth": cl["truth"], "rep": r_i, "source": src, "label": label, "correct": _match(label, cl["truth"]), "alarmed": alarmed,
                                  "max_leg_share": rd["max_leg_share"], "base_fz_z": rd["base_fz_z"], "base_mx_z": rd["base_mx_z"], "resolved": f"{rd['resolved_leg']}-{rd['resolved_joint']}", "why": why})
            # contact-wrench sensitivity (analytic rows / base rows): re-decide at each scale
            for cs in cs_list:
                if f"Zb_{cs}" not in o:
                    continue
                rd = readout(o["Z"], o[f"Zr_an_{cs}"], o[f"Zb_{cs}"], C2Rep(o["man"]), _RES_REP, K_cal, o["chans"], RES_COLS, use_residual_for_rminus=True)
                S, al = ecusum(o["pw_an"], h["an"], start=w0)
                label, conf, why = decide(rd, al is not None, base_z_thresh=iso["base_z_thresh"], share_thresh=iso["share_thresh"])
                sens_rows.append({"class": cl["name"], "truth": cl["truth"], "rep": r_i, "contact_scale_err": cs, "label": label, "correct": _match(label, cl["truth"]), "base_fz_z": rd["base_fz_z"], "base_mx_z": rd["base_mx_z"], "max_leg_share": rd["max_leg_share"]})
    conf = pd.DataFrame(conf_rows); conf.to_csv(res_dir / "e09_confusion.csv", index=False)
    sens = pd.DataFrame(sens_rows); sens.to_csv(res_dir / "e09_contact_sensitivity.csv", index=False)
    acc = conf.groupby("source").correct.mean().reset_index(); acc.to_csv(res_dir / "e09_accuracy.csv", index=False)
    for src in conf.source.unique():
        cm = pd.crosstab(conf[conf.source == src].truth, conf[conf.source == src].label); cm.to_csv(res_dir / f"e09_confusion_{src}.csv")
    _plot(res_dir, conf, sens)
    def cls_acc(src, cls):
        s = conf[(conf.source == src) & (conf["class"] == cls)]; return float(s.correct.mean()) if len(s) else np.nan
    ss = sens.groupby("contact_scale_err").correct.mean()
    _conclude(res_dir, "[e09-iso] accuracy: " + "; ".join(f"{r.source} {r.correct:.2f}" for r in acc.itertuples())
              + f" | single_friction_LH-KFE (the diagnosis class): analytic {cls_acc('analytic_rows','single_friction_LH-KFE'):.2f}, equiv {cls_acc('equiv_rows','single_friction_LH-KFE'):.2f}"
              + " | contact-wrench sensitivity (accuracy vs +-error): " + ", ".join(f"{k:+.0%} {v:.2f}" for k, v in ss.items()))
    return conf


def _match(label, truth):
    if label == truth:
        return True
    # single_leg:X matches single_leg:X; inertia truth accepts the KFE class
    return label.split(":")[0] == truth.split(":")[0] and label.split(":")[-1] == truth.split(":")[-1]


def _plot(res_dir, conf, sens):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    srcs = list(conf.source.unique()); fig, axes = plt.subplots(1, len(srcs) + 1, figsize=(5.2 * (len(srcs) + 1), 4.2))
    for ax, src in zip(axes, srcs):
        cm = pd.crosstab(conf[conf.source == src].truth, conf[conf.source == src].label)
        im = ax.imshow(cm.to_numpy(), cmap="Blues", aspect="auto"); ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns, rotation=70, fontsize=6, ha="right"); ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index, fontsize=6)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm.iloc[i, j]:
                    ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center", fontsize=6, color="w" if cm.iloc[i, j] > cm.to_numpy().max() / 2 else "k")
        ax.set_title(f"{src}: acc {conf[conf.source==src].correct.mean():.2f}", fontsize=9); ax.set_xlabel("label"); ax.set_ylabel("truth")
    ss = sens.groupby(["contact_scale_err"]).correct.mean().reset_index(); axes[-1].plot(ss.contact_scale_err * 100, ss.correct, "o-"); axes[-1].set_xlabel("contact-wrench error [%]"); axes[-1].set_ylabel("isolation accuracy"); axes[-1].set_ylim(0, 1.05); axes[-1].grid(alpha=0.3); axes[-1].set_title("contact-wrench sensitivity", fontsize=9)
    fig.suptitle("e09 — three-channel isolation confusion (analytic vs equivariant residual rows) + contact sensitivity", fontsize=10); fig.tight_layout()
    fig.savefig(res_dir / "e09_confusion.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["diagnose", "iso", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml")); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    if a.workers:
        cfg["workers"] = a.workers
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True); (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["diagnose", "iso"] if a.stage == "all" else [a.stage]
    print(f"[{EXP_NAME}] run_id={a.run_id} stages={stages}", flush=True)
    for s in stages:
        t0 = _dt.datetime.now(); {"diagnose": stage_diagnose, "iso": stage_iso}[s](cfg, res_dir, quick=a.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    print("E09 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
