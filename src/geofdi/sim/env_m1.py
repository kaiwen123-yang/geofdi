"""Headless MuJoCo wheeled-M1 environment (Sprint 7 Block W1): rolling-mode rollouts emitting the M1 telemetry schema.

    cfg = SimConfigM1(model="m1_wheeled_sym", speed=1.0, duration_s=60, seed=3, faults=[...])
    df, manifest = rollout_m1(cfg)

Same discipline as sim/env.py (Go2): all randomness from numpy.random.default_rng(cfg.seed); MuJoCo deterministic;
GeoFDI joint order LF, RF, LH, RH x ABAD, HIP, KNEE, WHEEL (MJCF order permuted through telemetry_m1.MJCF_TO_GEOFDI);
noise injected in the body frame; wheel-ground contact flag/wrench per leg (floor contacts of the wheel geoms and the
leg bodies). Faults/nuisances (FaultBankM1): actuator_gain / actuator_bias (per joint mask), wheel_friction (wheel
geom sliding friction x (1+m)), friction_scale (joint damping/frictionloss x (1+m)), payload_symmetric /
payload_asymmetric (base mass, lateral CoM offset), drift_symmetric (OU factor on all torques and joint damping) — all
with the FaultSpec schedule semantics of sim/faults.py.

Rolling data elements: the phase column is replaced by `blk` = index of the fixed-duration block (block length set by
the registration, not here); the environment only records the commanded speed and a warm-up marker (blk = -1 while
t < warmup_s or the ramp is active).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources

import mujoco
import numpy as np
import pandas as pd

from .controller import TrotController, TrotParams
from .controller_wheeled import RollingController, RollingParams
from .env import NoiseConfig, _body_state, _exp_so3, _foot_contact_forces
from .faults import FaultSpec
from .telemetry_m1 import (GEOFDI_TO_MJCF, JOINTS, LEGS, MJCF_LEG, MJCF_TO_GEOFDI, MODEL_JOINTS, WHEEL_GEOMS, all_columns,
                           build_manifest)

MODELS_M1 = {"m1_wheeled": "assets/m1/scene_m1_wheeled.xml", "m1_wheeled_sym": "assets/m1/scene_m1_wheeled_sym.xml"}


def scene_path_m1(model: str) -> str:
    if model not in MODELS_M1:
        raise KeyError(f"unknown M1 model '{model}'; known: {sorted(MODELS_M1)}")
    return str(resources.files("geofdi.sim").joinpath(MODELS_M1[model]))


@dataclass
class SimConfigM1:
    model: str = "m1_wheeled_sym"
    mode: str = "rolling"              # 'rolling' (Σ = G) | 'stepping' (equivariant PD trot on the leg joints, wheels position-held; Σ = (g_s, 1/2))
    speed: float = 1.0                 # m/s
    duration_s: float = 30.0
    warmup_s: float = 2.0              # blk = -1 before this (also covers the speed ramp)
    ctrl_dt: float = 0.005
    sim_dt: float = 0.0025
    seed: int = 0
    controller: dict = field(default_factory=dict)      # RollingParams overrides
    noise: dict = field(default_factory=dict)           # NoiseConfig overrides
    faults: list = field(default_factory=list)
    nuisance: list = field(default_factory=list)
    imu_mode: str = "sampled"
    phase_offset: float = 0.0          # stepping mode: controller clock offset (turns); the mirror-sim test uses 0.5
    init_qpos: list | None = None
    init_qvel: list | None = None
    temp_tau_s: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)


class FaultBankM1:
    """M1 injectors on the 16-joint GeoFDI vector; schedules from FaultSpec (type names extended)."""
    TYPES = ("actuator_gain", "actuator_bias", "wheel_friction", "friction_scale", "payload_symmetric", "payload_asymmetric",
             "drift_symmetric", "encoder_bias")

    def __init__(self, specs, model: mujoco.MjModel, ctrl_dt: float, rng: np.random.Generator):
        self.specs = []
        for s in (specs or []):
            d = dict(s) if not isinstance(s, FaultSpec) else asdict(s)
            t = d.get("type")
            if t not in self.TYPES:
                raise ValueError(f"unknown M1 injector type {t!r}")
            self.specs.append(_SpecM1(**d))
        self.model = model; self.dt = ctrl_dt; self.rng = rng; self._ou = {}; self._g_sym = 0.0
        self._dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])[MJCF_TO_GEOFDI]   # GeoFDI order
        self._dof_damping0 = model.dof_damping.copy(); self._dof_frictionloss0 = model.dof_frictionloss.copy()
        self._body_mass0 = model.body_mass.copy(); self._body_ipos0 = model.body_ipos.copy(); self._body_inertia0 = model.body_inertia.copy()
        self._geom_friction0 = model.geom_friction.copy()
        self._wheel_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, WHEEL_GEOMS[l]) for l in LEGS]
        self._base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.has_model_faults = any(s.type in ("wheel_friction", "friction_scale", "payload_symmetric", "payload_asymmetric", "drift_symmetric") for s in self.specs)

    def advance(self, t):
        self._g_sym = 0.0
        for s in self.specs:
            if s.type == "drift_symmetric":
                tau = float(s.params.get("tau_s", 20.0)); a = np.exp(-self.dt / tau)
                x = self._ou.get(id(s), 0.0); x = a * x + s.magnitude * np.sqrt(1 - a * a) * self.rng.standard_normal(); self._ou[id(s)] = x
                if s.s(t) > 0:
                    self._g_sym += x * s.s(t)

    def torque(self, tau_cmd, t):
        tau = tau_cmd.copy()
        for s in self.specs:
            sv = s.s(t)
            if not sv:
                continue
            if s.type == "actuator_gain":
                m = s.mask(); tau[m] = tau[m] * (1.0 + s.magnitude * sv)
            elif s.type == "actuator_bias":
                m = s.mask(); tau[m] = tau[m] + s.magnitude * sv
        if self._g_sym != 0.0:
            tau = tau * (1.0 + self._g_sym)
        return tau

    def measure(self, q, t):
        qm = q.copy()
        for s in self.specs:
            if s.type == "encoder_bias" and s.s(t):
                qm[s.mask()] += s.magnitude * s.s(t)
        return qm

    def model_update(self, t):
        if not self.has_model_faults:
            return
        m = self.model
        m.dof_damping[:] = self._dof_damping0; m.dof_frictionloss[:] = self._dof_frictionloss0
        m.body_mass[:] = self._body_mass0; m.body_ipos[:] = self._body_ipos0; m.body_inertia[:] = self._body_inertia0
        m.geom_friction[:] = self._geom_friction0
        for s in self.specs:
            sv = s.s(t)
            if not sv:
                continue
            if s.type == "wheel_friction":
                for li, leg in enumerate(LEGS):
                    if s.leg in (None, leg):
                        g = self._wheel_geom[li]; m.geom_friction[g, 0] = self._geom_friction0[g, 0] * (1 + s.magnitude * sv)
            elif s.type == "friction_scale":
                dof = self._dof[s.mask()]
                m.dof_damping[dof] = self._dof_damping0[dof] * (1 + s.magnitude * sv); m.dof_frictionloss[dof] = self._dof_frictionloss0[dof] * (1 + s.magnitude * sv)
            elif s.type in ("payload_symmetric", "payload_asymmetric"):
                b = self._base; m0 = self._body_mass0[b]; madd = s.magnitude * sv
                m.body_mass[b] = m0 + madd
                if s.type == "payload_asymmetric":
                    off = float(s.params.get("offset_y", 0.05))
                    m.body_ipos[b] = (m0 * self._body_ipos0[b] + madd * (self._body_ipos0[b] + np.array([0, off, 0]))) / (m0 + madd)
        if self._g_sym != 0.0:
            m.dof_damping[self._dof] = self._dof_damping0[self._dof] * (1 + self._g_sym); m.dof_frictionloss[self._dof] = self._dof_frictionloss0[self._dof] * (1 + self._g_sym)


@dataclass
class _SpecM1:
    type: str
    t_onset: float = 0.0
    leg: str | None = None
    joint: str | None = None
    magnitude: float = 0.0
    schedule: str = "step"
    ramp_s: float = 0.0
    params: dict = field(default_factory=dict)

    def s(self, t):
        if self.schedule == "step":
            return 1.0 if t >= self.t_onset else 0.0
        if self.schedule == "ramp":
            return 0.0 if t < self.t_onset else float(min(1.0, (t - self.t_onset) / max(self.ramp_s, 1e-9)))
        dur = float(self.params.get("duration_s", 1.0)); return 1.0 if self.t_onset <= t < self.t_onset + dur else 0.0

    def mask(self):
        m = np.zeros(16, dtype=bool)
        for i, leg in enumerate(LEGS):
            for j, jn in enumerate(JOINTS):
                if (self.leg in (None, leg)) and (self.joint in (None, jn)):
                    m[4 * i + j] = True
        return m


def load_model_m1(cfg: SimConfigM1) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(scene_path_m1(cfg.model))


def rollout_m1(cfg: SimConfigM1, model: mujoco.MjModel | None = None, return_state: bool = False):
    if cfg.mode not in ("rolling", "stepping"):
        raise NotImplementedError("M1 modes: rolling | stepping")
    stepping = cfg.mode == "stepping"
    m = model if model is not None else load_model_m1(cfg)
    m.opt.timestep = cfg.sim_dt
    nsub = round(cfg.ctrl_dt / cfg.sim_dt)
    if abs(nsub * cfg.sim_dt - cfg.ctrl_dt) > 1e-12:
        raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
    rng = np.random.default_rng(cfg.seed)
    noise = NoiseConfig(**cfg.noise)
    if stepping:
        # equivariant PD trot on the 12 leg joints (same uniform-axis convention as the Go2 template), wheels position-held
        # W1 stepping smoke: (0, 0.8, -1.5) stance, lift 0.30/0.10, kp 200, kd 6, period 0.5 s is the first variant that stays
        # up for 30 s (z 0.45-0.46, roll +-1 deg, pitch -5 deg, front duty 0.5, hind duty 0.8 - the M1 is rear-heavy).
        sp = dict(period_s=0.5, q_stand=(0.0, 0.8, -1.5), kp=(200.0, 200.0, 200.0), kd=(6.0, 6.0, 6.0), lift_kfe=0.30, lift_hfe=0.10,
                  tau_max=(40.0, 60.0, 60.0), leg_len=0.35, stab_k_wz=0.2)
        sp.update({k: v for k, v in cfg.controller.items() if k in TrotParams.__dataclass_fields__}); sp["speed"] = cfg.speed
        tparams = TrotParams(**sp); ctrl_step = TrotController(tparams)
        kv_lock = float(cfg.controller.get("kv_lock", 2.0)); kp_lock = float(cfg.controller.get("kp_lock", 20.0))
        T = tparams.period_s; n_per = round(T / cfg.ctrl_dt)
        if abs(n_per * cfg.ctrl_dt - T) > 1e-12:
            raise ValueError("period_s must be an integer multiple of ctrl_dt")
        k0 = round(cfg.phase_offset * n_per)
        if abs(k0 - cfg.phase_offset * n_per) > 1e-9:
            raise ValueError("phase_offset must be a multiple of ctrl_dt/period_s")
    ctrl = RollingController(RollingParams(**{**{k: v for k, v in cfg.controller.items() if k in RollingParams.__dataclass_fields__}, "speed": cfg.speed}))
    d = mujoco.MjData(m)
    qadr_m = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    vadr_m = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    qadr = qadr_m[MJCF_TO_GEOFDI]; vadr = vadr_m[MJCF_TO_GEOFDI]            # GeoFDI-ordered addresses
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if cfg.init_qpos is not None:
        d.qpos[:] = np.asarray(cfg.init_qpos, dtype=float); d.qvel[:] = np.asarray(cfg.init_qvel, dtype=float) if cfg.init_qvel is not None else 0.0
    else:
        d.qpos[qadr] += rng.normal(0.0, noise.init_joint_std, 16) * np.tile([1, 1, 1, 0], 4)     # no wheel-angle perturbation
        d.qvel[vadr] += rng.normal(0.0, noise.init_vel_std, 16)
        d.qvel[3:6] += rng.normal(0.0, noise.init_body_rate_std, 3)
    mujoco.mj_forward(m, d)
    faults = FaultBankM1(list(cfg.faults) + list(cfg.nuisance), m, cfg.ctrl_dt, rng)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, WHEEL_GEOMS[l]) for l in LEGS]
    leg_bodies = {}
    for li, leg in enumerate(LEGS):
        for part in ("ABAD", "HIP", "KNEE", "FOOT"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{MJCF_LEG[leg]}_{part}_LINK")
            if bid >= 0:
                leg_bodies[bid] = li
    sid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in ("imu_acc", "imu_gyro", "base_quat", "base_linvel")}
    sadr = {n: (m.sensor_adr[i], m.sensor_dim[i]) for n, i in sid.items()}
    n_steps = round(cfg.duration_s / cfg.ctrl_dt)
    rows = np.empty((n_steps, len(all_columns())), dtype=float)
    temp = np.zeros(4); a_temp = np.exp(-cfg.ctrl_dt / cfg.temp_tau_s); gyr_prev = np.zeros(3)
    ramp_s = float(cfg.controller.get("ramp_s", RollingParams().ramp_s))
    warm = max(cfg.warmup_s, ramp_s)
    leg_idx = np.array([4 * li + j for li in range(4) for j in range(3)]); wheel_idx = np.arange(3, 16, 4)
    q_wheel_lock = d.qpos[qadr][wheel_idx].copy()
    for k in range(n_steps):
        t = d.time
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        faults.advance(t); faults.model_update(t)
        q_meas = faults.measure(q, t) + rng.normal(0.0, noise.encoder_pos_std, 16)
        dq_meas = dq + rng.normal(0.0, noise.encoder_vel_std, 16)
        body = _body_state(d, sadr, noise, rng, gyr_prev)
        if stepping:
            theta = ((k + k0) % n_per) / n_per
            tau12, qref12, _ = ctrl_step.torque(q_meas[leg_idx], dq_meas[leg_idx], theta, t, body=body)
            tau_cmd = np.zeros(16); tau_cmd[leg_idx] = tau12
            tau_cmd[wheel_idx] = np.clip(kp_lock * (q_wheel_lock - q_meas[wheel_idx]) - kv_lock * dq_meas[wheel_idx], -20, 20)
        else:
            theta = np.nan
            tau_cmd, ref = ctrl.torque(q_meas, dq_meas, t, body=body)
        tau_app = faults.torque(tau_cmd, t) + rng.normal(0.0, noise.actuator_std, 16)
        d.ctrl[:] = tau_app[GEOFDI_TO_MJCF]                                     # actuators are in MJCF order
        if cfg.imu_mode == "integrating":
            q_prev = d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4].copy(); v_prev = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3].copy()
        acc_sum = np.zeros(3); gyr_sum = np.zeros(3); fw_sum = np.zeros((4, 3)); tw_sum = np.zeros((4, 3)); cpw_sum = np.zeros((4, 3)); fzw_sum = np.zeros(4); fz_s = np.zeros(4)
        for _ in range(nsub):
            mujoco.mj_step(m, d)
            if cfg.imu_mode == "averaged":
                acc_sum += d.sensordata[sadr["imu_acc"][0]:sadr["imu_acc"][0] + 3]; gyr_sum += d.sensordata[sadr["imu_gyro"][0]:sadr["imu_gyro"][0] + 3]
            fz_s, fw_s, tw_s, cpw_s, fsum_s = _foot_contact_forces(m, d, floor, wheels, leg_bodies)
            fw_sum += fw_s; tw_sum += tw_s; cpw_sum += cpw_s; fzw_sum += fsum_s
        tau_meas = tau_app + rng.normal(0.0, noise.torque_meas_std, 16)
        if cfg.imu_mode == "sampled":
            acc0 = d.sensordata[sadr["imu_acc"][0]:sadr["imu_acc"][0] + 3].copy(); gyr0 = d.sensordata[sadr["imu_gyro"][0]:sadr["imu_gyro"][0] + 3].copy()
        elif cfg.imu_mode == "averaged":
            acc0 = acc_sum / nsub; gyr0 = gyr_sum / nsub
        else:
            mujoco.mj_forward(m, d)
            q_new = d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4]; v_new = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
            R_prev = np.zeros(9); mujoco.mju_quat2Mat(R_prev, q_prev); R_prev = R_prev.reshape(3, 3)
            R_new = np.zeros(9); mujoco.mju_quat2Mat(R_new, q_new); R_new = R_new.reshape(3, 3)
            dR = R_prev.T @ R_new; ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            gyr0 = (axis / (2 * np.sin(ang)) * ang / cfg.ctrl_dt) if ang > 1e-12 else np.zeros(3)
            R_mid = R_prev @ _exp_so3(0.5 * gyr0 * cfg.ctrl_dt); acc0 = R_mid.T @ ((v_new - v_prev) / cfg.ctrl_dt - m.opt.gravity)
        acc = acc0 + rng.normal(0.0, noise.imu_acc_std, 3); gyr = gyr0 + rng.normal(0.0, noise.imu_gyro_std, 3); gyr_prev = gyr
        fw = fw_sum / nsub; tw0 = tw_sum / nsub; cpw = np.zeros((4, 3)); tw = np.zeros((4, 3))
        for j, g in enumerate(wheels):
            cpw[j] = cpw_sum[j] / fzw_sum[j] if fzw_sum[j] > 0 else d.geom_xpos[g] - np.array([0.0, 0.0, m.geom_size[g][0]])
            tw[j] = tw0[j] - np.cross(cpw[j], fw[j])
        c = (fz_s > 0).astype(float)
        temp = a_temp * temp + (1 - a_temp) * (tau_app.reshape(4, 4) ** 2).mean(axis=1)
        base_lin = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
        blk = -1.0 if (t < warm or stepping) else float(np.floor((t - warm) / 1.0))   # provisional 1 s block index (registration re-blocks)
        rows[k, :] = np.concatenate([[d.time, blk, theta], q_meas, dq_meas, tau_cmd, tau_meas, acc, gyr, c, temp, [0.0 if stepping else ctrl._speed(t)],
                                     d.xpos[base_id], d.xquat[base_id], base_lin, fw.ravel(), cpw.ravel(), tw.ravel()])
    df = pd.DataFrame(rows, columns=all_columns())
    manifest = build_manifest(sim_meta={"model": cfg.model, "mode": cfg.mode, "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt, "speed": cfg.speed,
                                        "seed": cfg.seed, "warmup_s": warm, "imu_mode": cfg.imu_mode,
                                        "record_time": "end of control step (state at t+ctrl_dt; tau_cmd from t)"})
    if return_state:
        return df, manifest, (d.qpos.copy(), d.qvel.copy())
    return df, manifest


def keyframe_state_m1(model: str = "m1_wheeled_sym"):
    m = mujoco.MjModel.from_xml_path(scene_path_m1(model)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    return d.qpos.copy(), d.qvel.copy()
