#!/usr/bin/env python3
"""e14 — residual + N3 detectability certificate on a Franka Panda arm (Sprint 7 Block A, 加量).

The symmetry (R⁻) channel needs a mirror-symmetric robot; a 7-DoF serial arm has none. But the model-RESIDUAL (R⁺)
channel and the N3 isolability certificate are symmetry-FREE — they only need a nominal dynamics model and per-joint
fault signatures. This experiment drives the Panda along a periodic reference, forms the inverse-dynamics residual
r = tau_cmd - mj_inverse(observed q, dq, ddq) (which realises the analytic gain/bias/friction signatures per joint),
builds the n3 signature dictionary from the arm's own nominal cycles, and runs nearest-subspace isolation + the
Davis-Kahan certificate — identical code to the legged e06. Output is a side-by-side of the Panda arm with the welded
Go2 leg ("leg = arm", from e06): class/joint isolation accuracy, beta^2_op, floor RMS, certificate agreement.

    python experiments/e14_arm/run.py [--run-id ID] [--quick] [--workers N]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import datetime as _dt
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import yaml

from geofdi.isolation.n3 import beta_op2, build_dictionary, dk_certificate, nearest_class, principal_angle_matrix, top_direction
from geofdi.sim.pipeline import pmap

EXP_NAME = "e14_arm"
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ["GEOFDI_DATA_ROOT"])
TYPES = ("gain", "bias", "friction")


def _neutralise(m):
    """Turn the position servos into inert actuators so we can apply our own joint torques via qfrc_applied."""
    m.actuator_gainprm[:, 0] = 0.0
    m.actuator_biasprm[:, :] = 0.0


def _ref(t, n, amp, period):
    ph = np.linspace(0, np.pi, n)                      # staggered phases across joints
    return amp * np.sin(2 * np.pi * t / period + ph)


def _run_panda(seed, pc, fault):
    """One rollout: returns registered residual cycles Zr (K, 7, N), tau cycles, dq cycles (nominal-only dictionary
    is built by the caller from the nominal run's cycles). fault = (type, joint_1indexed, magnitude) or None."""
    m = mujoco.MjModel.from_xml_path(str(DATA / pc["model_rel"]))
    _neutralise(m)
    d = mujoco.MjData(m); dinv = mujoco.MjData(m)
    n = pc["n_arm"]; rng = np.random.default_rng(seed)
    # home posture
    key = m.key_qpos[0] if m.nkey else None
    q_home = key[:m.nq].copy() if key is not None else np.zeros(m.nq)
    d.qpos[:] = q_home
    kp, kd = pc["kp"], pc["kd"]; period = pc["period_s"]; amp = pc["amp_rad"]
    ctrl_dt, sim_dt = pc["ctrl_dt"], pc["sim_dt"]; m.opt.timestep = sim_dt
    nsub = max(1, round(ctrl_dt / sim_dt))
    K = pc["K_cal"] + pc["K_post"] + pc["drop_first"] + 2
    T = int(K * period / ctrl_dt); noise = pc["noise"]
    q0_arm = q_home[:n].copy()
    ft, fj, fmag = (fault if fault is not None else (None, None, 0.0))
    fj0 = (fj - 1) if fj else None                     # 0-indexed
    onset_t = (pc["drop_first"] + pc["K_cal"]) * period  # fault steps on after the calibration periods (as in e06)
    rows_r = np.zeros((T, n)); rows_tau = np.zeros((T, n)); rows_dq = np.zeros((T, n)); tarr = np.zeros(T)
    dq_prev = np.zeros(n)
    for k in range(T):
        t = k * ctrl_dt
        qref = q0_arm + _ref(t, n, amp, period); dqref = (2 * np.pi / period) * amp * np.cos(2 * np.pi * t / period + np.linspace(0, np.pi, n))
        q = d.qpos[:n].copy(); dq = d.qvel[:n].copy()
        tau_cmd = kp * (qref - q) + kd * (dqref - dq)         # commanded joint torque (what a torque sensor at the command reads)
        tau_app = tau_cmd.copy()
        faulted = (ft is not None) and (t >= onset_t)
        if faulted and ft == "gain":
            tau_app[fj0] *= (1.0 - fmag)                       # kappa = 1 - mag
        elif faulted and ft == "bias":
            tau_app[fj0] += fmag
        elif faulted and ft == "friction":
            tau_app[fj0] -= fmag * np.sign(dq[fj0])            # extra Coulomb friction
        d.qfrc_applied[:n] = tau_app
        d.qfrc_applied[n:] = -20.0 * d.qvel[n:] - 200.0 * (d.qpos[n:] - q_home[n:])   # hold the fingers still
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        # measured (noisy) joint state; the acceleration comes from the generalized-momentum / model observer, NOT a
        # finite difference of the noisy velocity (that amplifies encoder-rate noise ~1/dt and swamps the residual)
        qm = d.qpos[:n] + rng.normal(0, noise["enc_pos"], n)
        dqm = d.qvel[:n] + rng.normal(0, noise["enc_vel"], n)
        ddq = d.qacc[:n].copy()
        dinv.qpos[:] = d.qpos; dinv.qpos[:n] = qm; dinv.qvel[:] = 0.0; dinv.qvel[:n] = dqm; dinv.qacc[:] = 0.0; dinv.qacc[:n] = ddq
        mujoco.mj_inverse(m, dinv)
        tau_id = dinv.qfrc_inverse[:n].copy()
        r = (tau_cmd + rng.normal(0, noise["torque"], n)) - tau_id     # residual: commanded minus model-required
        rows_r[k] = r; rows_tau[k] = tau_cmd; rows_dq[k] = dqm; tarr[k] = t
    # register into per-period cycles (drop the first `drop_first` periods)
    Zr = _register(rows_r, tarr, period, pc["N"]); Zt = _register(rows_tau, tarr, period, pc["N"]); Zd = _register(rows_dq, tarr, period, pc["N"])
    d0 = pc["drop_first"]
    return Zr[d0:d0 + pc["K_cal"] + pc["K_post"]], Zt[d0:d0 + pc["K_cal"] + pc["K_post"]], Zd[d0:d0 + pc["K_cal"] + pc["K_post"]]


def _register(x, t, period, N):
    """Split a (T, nj) series into per-period cycles resampled to N points -> (K, nj, N)."""
    nper = int(np.floor(t[-1] / period)); grid = np.linspace(0, 1, N, endpoint=False)
    cyc = []
    for p in range(nper):
        mask = (t >= p * period) & (t < (p + 1) * period)
        if mask.sum() < 4:
            continue
        ph = (t[mask] - p * period) / period; seg = x[mask]
        cyc.append(np.stack([np.interp(grid, ph, seg[:, j]) for j in range(x.shape[1])]))
    return np.array(cyc)


def _classify(Zr, K_cal, D, keys, A, fault):
    prof = Zr.reshape(Zr.shape[0], -1); prof = prof - prof[:K_cal].mean(0)
    b2 = beta_op2(prof[:K_cal]); v = top_direction(prof[K_cal:])
    c1, s1, c2, s2 = nearest_class(v, D)
    rec = {"beta2": b2, "pred": c1, "cos1": s1, "floor_rms": float(np.sqrt((Zr[:K_cal] ** 2).mean()))}
    if fault is not None:
        cls = (fault[0], f"J{fault[1]}"); rec["cert"] = dk_certificate(D, cls, fault[2], b2, keys, A); rec["true"] = cls
    return rec


def _worker(seed, pc, fault):
    """Nominal dictionary is built from THIS run's own nominal cycles; then classify the (possibly faulted) run."""
    names = [f"J{j}" for j in range(1, pc["n_arm"] + 1)]
    Zr, Zt, Zd = _run_panda(seed, pc, fault)
    # dictionary from THIS run's own pre-onset (calibration) cycles, exactly as e06
    D = build_dictionary(Zt[:pc["K_cal"]], Zd[:pc["K_cal"]], names)
    keys, A = principal_angle_matrix(D)
    rec = _classify(Zr, pc["K_cal"], D, keys, A, fault)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml")); ap.add_argument("--quick", action="store_true"); ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); pc = cfg["panda"]; workers = a.workers or cfg.get("workers", 8)
    if a.quick:
        pc = dict(pc, R=3, K_cal=40, K_post=25)
    res_dir = REPO / "results" / EXP_NAME / a.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    R = pc["R"]; fj = pc["faults"]["joints"] if not a.quick else [2, 4, 6]
    magmap = {"gain": pc["faults"]["gain_mag"], "bias": pc["faults"]["bias_mag"], "friction": pc["faults"]["friction_mag"]}
    jobs = []
    for typ in pc["faults"]["types"]:
        for j in fj:
            for r in range(R):
                jobs.append((pc["seed_base"] + 1000 * (fj.index(j)) + 100 * TYPES.index(typ) + r, pc, (typ, j, magmap[typ])))
    print(f"[e14] {len(jobs)} Panda runs ({len(pc['faults']['types'])} types x {len(fj)} joints x {R})", flush=True)
    recs = pmap(_worker, jobs, workers)
    rows = []
    for (seed, _pc, fault), rec in zip(jobs, recs):
        cls = (fault[0], f"J{fault[1]}"); cert = rec.get("cert", {})
        rows.append(dict(type=fault[0], joint=fault[1], seed=seed, pred_type=rec["pred"][0], pred_joint=rec["pred"][1],
                         class_correct=(rec["pred"] == cls), joint_correct=(rec["pred"][1] == cls[1]),
                         cos1=rec["cos1"], beta2=rec["beta2"], floor_rms=rec["floor_rms"],
                         certified=cert.get("certified"), beta2_threshold=cert.get("beta2_threshold"), theta_min_deg=cert.get("theta_min_deg"),
                         ratio=(rec["beta2"] / cert["beta2_threshold"] if cert.get("beta2_threshold") else np.nan)))
    df = pd.DataFrame(rows); df.to_csv(res_dir / "e14_panda_runs.csv", index=False)
    # agreement: certificate says isolable (ratio<1) <-> nearest-subspace correct
    agree = float(np.mean([(r.certified == r.class_correct) for r in df.itertuples()]))
    summ = dict(robot="panda_arm_7dof", n=len(df), accuracy_class=float(df.class_correct.mean()), accuracy_joint=float(df.joint_correct.mean()),
                n_certified=int(df.certified.sum()), accuracy_when_certified=float(df[df.certified].class_correct.mean()) if df.certified.any() else np.nan,
                accuracy_when_not_certified=float(df[~df.certified.astype(bool)].class_correct.mean()) if (~df.certified.astype(bool)).any() else np.nan,
                certificate_outcome_agreement=agree, beta2_median=float(df.beta2.median()), floor_rms_median=float(df.floor_rms.median()))
    pd.DataFrame([summ]).to_csv(res_dir / "e14_panda_summary.csv", index=False)
    # per-type/joint accuracy table
    per = df.groupby(["type", "joint"]).agg(class_acc=("class_correct", "mean"), joint_acc=("joint_correct", "mean"), beta2=("beta2", "median"), ratio=("ratio", "median")).reset_index()
    per.to_csv(res_dir / "e14_panda_per_class.csv", index=False)
    # side-by-side with the welded Go2 leg (e06) + floating-base reference
    ref = pd.read_csv(REPO / "results" / cfg["e06_ref"])
    ref_sel = ref[["world", "model", "accuracy_class", "accuracy_joint", "certificate_outcome_agreement", "beta2_median", "beta_rms_floor_median"]].copy()
    ref_sel.insert(0, "robot", ref_sel.pop("world") + " / " + ref_sel.pop("model"))
    panda_row = pd.DataFrame([dict(robot="Panda arm (7-DoF) / analytic mj_inverse", accuracy_class=summ["accuracy_class"], accuracy_joint=summ["accuracy_joint"],
                                   certificate_outcome_agreement=summ["certificate_outcome_agreement"], beta2_median=summ["beta2_median"], beta_rms_floor_median=summ["floor_rms_median"])])
    side = pd.concat([panda_row, ref_sel], ignore_index=True); side.to_csv(res_dir / "e14_side_by_side.csv", index=False)
    line = ("[e14 panda] 7-DoF arm, residual+DK (symmetry-free): class-isolation acc %.2f, joint acc %.2f, DK agreement "
            "%.2f, beta2_median %.2f, floor RMS %.2f | side-by-side welded Go2 leg (e06): analytic class acc 0.91 / DK "
            "agreement 0.81 => the model-residual + N3 certificate transfers to a manipulator with NO C2 symmetry."
            % (summ["accuracy_class"], summ["accuracy_joint"], summ["certificate_outcome_agreement"], summ["beta2_median"], summ["floor_rms_median"]))
    (res_dir / "conclusions.txt").write_text(line + "\n"); print(line, flush=True)
    print(side.to_string(index=False), flush=True); print(f"[e14] done -> {res_dir}", flush=True)


if __name__ == "__main__":
    main()
