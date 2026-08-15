"""Headless MuJoCo Go2 environment: config-driven rollouts that emit M1-schema telemetry.

    cfg = SimConfig(duration_s=30, seed=3, controller={"speed": 0.0}, noise={...}, faults=[...], nuisance=[...])
    df, manifest = rollout(cfg)

No rendering anywhere (pure mj_step); MUJOCO_GL is irrelevant. Determinism: all randomness comes from
`numpy.random.default_rng(cfg.seed)` (initial-state perturbation, actuator/measurement noise, drift
processes); MuJoCo itself is deterministic for a given input sequence, so identical configs reproduce
bit-identically on the same machine/library build.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources

import mujoco
import numpy as np
import pandas as pd

from .controller import TrotController, TrotParams
from .faults import FaultBank
from .telemetry import all_columns, build_manifest

MODEL_JOINTS = [f"{l}_{j}_joint" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]  # == LF,RF,LH,RH x HAA,HFE,KFE
FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def scene_path(terrain: str = "flat") -> str:
    if terrain not in ("flat", "slope"):
        raise NotImplementedError("terrains: flat | slope (rough is reserved for later sprints)")
    return str(resources.files("geofdi.sim").joinpath("assets/unitree_go2/scene_flat.xml"))


def apply_terrain(m: mujoco.MjModel, terrain: str, slope_deg: float, slope_axis: str) -> None:
    """'slope' tilts gravity instead of the ground plane (exactly equivalent for a robot on the plane, keeps the
    keyframe valid): slope_axis 'lateral' = roll tilt about x (ground higher on the robot's left for
    slope_deg > 0; mirror-breaking, the A3 out-and-back case), 'sagittal' = pitch tilt about y (uphill along +x
    for slope_deg > 0; mirror-symmetric)."""
    g = 9.81
    if terrain == "flat" or slope_deg == 0.0:
        m.opt.gravity[:] = (0.0, 0.0, -g)
        return
    a = np.deg2rad(slope_deg)
    if slope_axis == "lateral":
        m.opt.gravity[:] = (0.0, g * np.sin(a), -g * np.cos(a))
    elif slope_axis == "sagittal":
        m.opt.gravity[:] = (-g * np.sin(a), 0.0, -g * np.cos(a))
    else:
        raise ValueError(slope_axis)


@dataclass
class NoiseConfig:
    """S1 baseline: measurement-noise dominated (quiet actuators on flat ground, realistic sensors). With actuator
    process noise 5x larger (0.1 N m) the per-cycle flip test is mildly anti-conservative (size ~0.075-0.09 at
    alpha 0.05, R=120): the within-cycle roll construction is exact only for reversible fluctuation dynamics,
    and dynamics-driven fluctuations are not — see the rp003 MANIFEST."""
    encoder_pos_std: float = 2e-3      # rad
    encoder_vel_std: float = 3e-2      # rad/s
    torque_meas_std: float = 0.20      # N m (measured motor torque)
    actuator_std: float = 0.02         # N m process noise on the applied torque (iid per step)
    imu_acc_std: float = 0.10          # m/s^2, body frame
    imu_gyro_std: float = 1e-2         # rad/s, body frame
    init_joint_std: float = 0.02       # rad, initial joint-position perturbation
    init_vel_std: float = 0.05         # rad/s, initial joint-velocity perturbation
    init_body_rate_std: float = 0.02   # rad/s, initial body angular velocity


@dataclass
class SimConfig:
    gait: str = "trot"
    speed: float = 0.0
    terrain: str = "flat"
    slope_deg: float = 0.0            # terrain == 'slope': tilt angle
    slope_axis: str = "lateral"       # 'lateral' (mirror-breaking) | 'sagittal'
    duration_s: float = 30.0
    ctrl_dt: float = 0.005            # 200 Hz control / telemetry (M1 rate)
    sim_dt: float = 0.0025            # 400 Hz physics (2 substeps)
    seed: int = 0
    phase_offset: float = 0.0         # controller clock offset (turns); the mirror-sim test uses 0.5
    controller: dict = field(default_factory=dict)   # TrotParams overrides (speed is taken from cfg.speed)
    noise: dict = field(default_factory=dict)        # NoiseConfig overrides
    faults: list = field(default_factory=list)       # FaultSpec dicts
    nuisance: list = field(default_factory=list)     # FaultSpec dicts (nuisance types)
    init_qpos: list | None = None     # explicit initial state (overrides keyframe + perturbation)
    init_qvel: list | None = None
    temp_tau_s: float = 20.0          # temperature-surrogate time constant

    def to_dict(self) -> dict:
        return asdict(self)


def rollout(cfg: SimConfig, model: mujoco.MjModel | None = None, return_state: bool = False):
    if cfg.gait != "trot":
        raise NotImplementedError("only the trot is implemented")
    m = model if model is not None else mujoco.MjModel.from_xml_path(scene_path(cfg.terrain))
    m.opt.timestep = cfg.sim_dt
    apply_terrain(m, cfg.terrain, cfg.slope_deg, cfg.slope_axis)
    nsub = round(cfg.ctrl_dt / cfg.sim_dt)
    if abs(nsub * cfg.sim_dt - cfg.ctrl_dt) > 1e-12:
        raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
    rng = np.random.default_rng(cfg.seed)
    noise = NoiseConfig(**cfg.noise)
    cparams = TrotParams(**{**cfg.controller, "speed": cfg.speed})
    ctrl = TrotController(cparams)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    if cfg.init_qpos is not None:
        d.qpos[:] = np.asarray(cfg.init_qpos, dtype=float)
        d.qvel[:] = np.asarray(cfg.init_qvel, dtype=float) if cfg.init_qvel is not None else 0.0
    else:
        d.qpos[qadr] += rng.normal(0.0, noise.init_joint_std, 12)
        d.qvel[vadr] += rng.normal(0.0, noise.init_vel_std, 12)
        d.qvel[3:6] += rng.normal(0.0, noise.init_body_rate_std, 3)
    mujoco.mj_forward(m, d)
    faults = FaultBank(list(cfg.faults) + list(cfg.nuisance), m, cfg.ctrl_dt, rng)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
    sid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in ("imu_acc", "imu_gyro", "base_quat", "base_linvel")}
    sadr = {n: (m.sensor_adr[i], m.sensor_dim[i]) for n, i in sid.items()}
    T = cparams.period_s
    n_per = round(T / cfg.ctrl_dt)                             # control steps per gait period (integer phase clock)
    if abs(n_per * cfg.ctrl_dt - T) > 1e-12:
        raise ValueError("period_s must be an integer multiple of ctrl_dt (integer phase clock)")
    k0 = round(cfg.phase_offset * n_per)
    if abs(k0 - cfg.phase_offset * n_per) > 1e-9:
        raise ValueError("phase_offset must be a multiple of ctrl_dt/period_s")
    n_steps = round(cfg.duration_s / cfg.ctrl_dt)
    rows = np.empty((n_steps, len(all_columns())), dtype=float)
    temp = np.zeros(4)
    a_temp = np.exp(-cfg.ctrl_dt / cfg.temp_tau_s)
    gyr_prev = np.zeros(3)
    for k in range(n_steps):
        t = d.time
        theta = ((k + k0) % n_per) / n_per                     # exact rational phase; no float ambiguity at transitions
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        faults.advance(t)
        faults.model_update(t)
        q_meas = faults.measure(q, t) + rng.normal(0.0, noise.encoder_pos_std, 12)
        dq_meas = dq + rng.normal(0.0, noise.encoder_vel_std, 12)
        body = _body_state(d, sadr, noise, rng, gyr_prev)
        tau_cmd, q_ref, _ = ctrl.torque(q_meas, dq_meas, theta, t, setpoint_offset=faults.setpoint_offset(t), body=body)
        tau_app = faults.torque(tau_cmd, t) + rng.normal(0.0, noise.actuator_std, 12)
        d.ctrl[:] = tau_app
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        tau_meas = tau_app + rng.normal(0.0, noise.torque_meas_std, 12)
        acc = d.sensordata[sadr["imu_acc"][0]:sadr["imu_acc"][0] + 3] + rng.normal(0.0, noise.imu_acc_std, 3)
        gyr = d.sensordata[sadr["imu_gyro"][0]:sadr["imu_gyro"][0] + 3] + rng.normal(0.0, noise.imu_gyro_std, 3)
        gyr_prev = gyr
        c = np.zeros(4)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            for j, f in enumerate(feet):
                if (g1 == floor and g2 == f) or (g2 == floor and g1 == f):
                    c[j] = 1.0
        temp = a_temp * temp + (1 - a_temp) * (tau_app.reshape(4, 3) ** 2).mean(axis=1)
        base_lin = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
        rows[k, :] = np.concatenate([[d.time, theta], q_meas, dq_meas, tau_cmd, tau_meas, acc, gyr, c, temp,
                                     d.qpos[0:3], d.qpos[3:7], base_lin, q_ref])
    df = pd.DataFrame(rows, columns=all_columns())
    manifest = build_manifest(sim_meta={"model": "unitree_go2 (menagerie da76818e, symmetrized, mesh-free)",
                                        "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt, "period_s": T,
                                        "gait": cfg.gait, "speed": cfg.speed, "terrain": cfg.terrain,
                                        "slope_deg": cfg.slope_deg, "slope_axis": cfg.slope_axis, "seed": cfg.seed,
                                        "phase_offset": cfg.phase_offset,
                                        "record_time": "end of control step (state at t+ctrl_dt; tau_cmd from t)"})
    if return_state:
        return df, manifest, (d.qpos.copy(), d.qvel.copy())
    return df, manifest


def _body_state(d, sadr, noise, rng, gyr_meas):
    """Body-frame feedback quantities for the stabilizer: roll (from the framequat sensor), roll/yaw rate (last
    measured gyro sample), lateral velocity (framelinvel rotated into the body frame). Mirror-antisymmetric."""
    q = d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4]
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    vw = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    vb = R.T @ vw
    return {"roll": float(roll), "w_x": float(gyr_meas[0]), "w_z": float(gyr_meas[2]), "v_y": float(vb[1])}


def keyframe_state(model: mujoco.MjModel | None = None):
    """(qpos, qvel) of the symmetric 'home' keyframe — a symmetric initial condition."""
    m = model if model is not None else mujoco.MjModel.from_xml_path(scene_path())
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return d.qpos.copy(), d.qvel.copy()
