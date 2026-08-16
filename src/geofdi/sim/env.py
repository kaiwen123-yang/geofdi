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


MODELS = {
    # S0 world (mujoco_menagerie go2, symmetrized; S1-S3 reproduction baseline)
    "go2_menagerie_sym": "assets/unitree_go2/scene_flat.xml",
    # go2_description URDF worlds (Sprint 4 Block G): original (base ixy/iyz kept) and mirror-symmetrized
    "go2_urdf": "assets/go2_urdf/mjcf/scene_go2_urdf.xml",
    "go2_urdf_sym": "assets/go2_urdf/mjcf/scene_go2_urdf_sym.xml",
}


def scene_path(terrain: str = "flat", model: str = "go2_menagerie_sym") -> str:
    if terrain not in ("flat", "slope"):
        raise NotImplementedError("terrains: flat | slope (rough is reserved for later sprints)")
    if model not in MODELS:
        raise KeyError(f"unknown sim model '{model}'; known: {sorted(MODELS)}")
    return str(resources.files("geofdi.sim").joinpath(MODELS[model]))


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
    model: str = "go2_menagerie_sym"  # sim world, see MODELS (S1-S3 used the menagerie world; D005: go2_urdf_sym default for new work)
    weld_base: bool = False           # weld the trunk to the world (leg = fixed-base manipulator; Block L1 'leg = arm')
    joint_damping: float | None = None  # override the 12 leg-joint dampings at load time (URDF world: 0.01 from the xacro;
                                      # menagerie world: 2.0 — the S1 value; used for the Block G damping diagnostic)
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
    foot_contact: str = "menagerie"  # 'menagerie' (soft feet, solimp .015 1 .022: tangential creep 1-3 cm/s under load) |
                                      # 'stiff' (solref 0.004 1, solimp .9 .95 .001: rubber foot on hard floor; estimator worlds)
    contact_force_thresh: float = 0.0 # contact indicator c = 1 iff the foot's normal contact force > this [N]
                                      # (0 = geometric contact, S1/S2 default; estimators use a settled-foot threshold)
    imu_mode: str = "sampled"         # 'sampled' (last substep) | 'averaged' (mean of substep readings) | 'integrating'
                                      # (exact delta-v/delta-theta over the control step, as an integrating IMU reports;
                                      # sampled accelerometers alias the contact impulses at 200 Hz: +0.014 m/s^2 z bias)
    init_qpos: list | None = None     # explicit initial state (overrides keyframe + perturbation)
    init_qvel: list | None = None
    temp_tau_s: float = 20.0          # temperature-surrogate time constant

    def to_dict(self) -> dict:
        return asdict(self)


def _foot_contact_forces(m: mujoco.MjModel, d: mujoco.MjData, floor: int, feet: list, leg_bodies: dict | None = None):
    """Per-leg floor-contact bookkeeping for one physics substep.
    Returns fz (4,) = normal force on the FOOT geom (contact flag semantics), and the per-leg contact WRENCH from all
    contacts between the floor and the leg's bodies (foot sphere + calf-lower cylinders, one rigid body): fw (4,3) world
    force, tw (4,3) world torque about the world origin (transported to the reference point later), cpw (4,3) normal-
    force-weighted contact-point sum, fsum (4,) normal-force sum for the weighting."""
    fz = np.zeros(4); fw = np.zeros((4, 3)); tw = np.zeros((4, 3)); cpw = np.zeros((4, 3)); fsum = np.zeros(4); f6 = np.zeros(6)
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if g1 != floor and g2 != floor:
            continue
        g = g2 if g1 == floor else g1
        leg = None
        for j, f in enumerate(feet):
            if g == f:
                leg = j; break
        if leg is None and leg_bodies is not None:
            leg = leg_bodies.get(int(m.geom_bodyid[g]))
        if leg is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        Fr = d.contact[i].frame.reshape(3, 3).T                         # contact frame -> world (frame rows = axes)
        Fw = Fr @ f6[:3]; Tw = Fr @ f6[3:]                              # force and torque (condim 6: torsional/rolling friction)
        if g2 == floor:                                                 # the normal points geom1 -> geom2 and the returned
            Fw = -Fw; Tw = -Tw                                          # wrench acts on geom2: flip if the robot geom is geom1
        pos = d.contact[i].pos
        if g == feet[leg]:
            fz[leg] += abs(f6[0])
        fw[leg] += Fw; tw[leg] += Tw + np.cross(pos, Fw); cpw[leg] += abs(f6[0]) * pos; fsum[leg] += abs(f6[0])
    return fz, fw, tw, cpw, fsum


def rollout(cfg: SimConfig, model: mujoco.MjModel | None = None, return_state: bool = False):
    if cfg.gait != "trot":
        raise NotImplementedError("only the trot is implemented")
    m = model if model is not None else load_model(cfg)
    m.opt.timestep = cfg.sim_dt
    if cfg.joint_damping is not None:
        for n in MODEL_JOINTS:
            m.dof_damping[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = cfg.joint_damping
    apply_terrain(m, cfg.terrain, cfg.slope_deg, cfg.slope_axis)
    if cfg.foot_contact == "stiff":
        for gname in FOOT_GEOMS:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            m.geom_solref[gid] = (0.004, 1.0); m.geom_solimp[gid] = (0.9, 0.95, 0.001, 0.5, 2.0)
    elif cfg.foot_contact != "menagerie":
        raise ValueError(cfg.foot_contact)
    nsub = round(cfg.ctrl_dt / cfg.sim_dt)
    if abs(nsub * cfg.sim_dt - cfg.ctrl_dt) > 1e-12:
        raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
    rng = np.random.default_rng(cfg.seed)
    noise = NoiseConfig(**cfg.noise)
    cparams = TrotParams(**{**cfg.controller, "speed": cfg.speed})
    ctrl = TrotController(cparams)
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in MODEL_JOINTS])
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    if cfg.weld_base:
        d.qpos[qadr] = np.tile([0.0, 0.9, -1.8], 4)               # 'home' leg posture, trunk welded at load_model height
    else:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    if cfg.init_qpos is not None:
        d.qpos[:] = np.asarray(cfg.init_qpos, dtype=float)
        d.qvel[:] = np.asarray(cfg.init_qvel, dtype=float) if cfg.init_qvel is not None else 0.0
    else:
        d.qpos[qadr] += rng.normal(0.0, noise.init_joint_std, 12)
        d.qvel[vadr] += rng.normal(0.0, noise.init_vel_std, 12)
        if not cfg.weld_base:
            d.qvel[3:6] += rng.normal(0.0, noise.init_body_rate_std, 3)
    mujoco.mj_forward(m, d)
    faults = FaultBank(list(cfg.faults) + list(cfg.nuisance), m, cfg.ctrl_dt, rng)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
    leg_bodies = {}                       # body id -> leg index, for the leg's hip/thigh/calf bodies (contact wrench bookkeeping)
    for j, leg in enumerate(("FL", "FR", "RL", "RR")):
        for part in ("hip", "thigh", "calf"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{part}")
            if bid >= 0:
                leg_bodies[bid] = j
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
        q_meas = faults.measure(q, t) + rng.normal(0.0, 1.0, 12) * faults.encoder_noise_std(noise.encoder_pos_std, t)
        dq_meas = dq + rng.normal(0.0, noise.encoder_vel_std, 12)
        body = _body_state(d, sadr, noise, rng, gyr_prev)
        tau_cmd, q_ref, _ = ctrl.torque(q_meas, dq_meas, theta, t, setpoint_offset=faults.setpoint_offset(t), body=body)
        tau_app = faults.torque(tau_cmd, t) + rng.normal(0.0, noise.actuator_std, 12)
        d.ctrl[:] = tau_app
        if cfg.imu_mode == "integrating":
            q_prev = d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4].copy()
            v_prev = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3].copy()
        acc_sum = np.zeros(3); gyr_sum = np.zeros(3); fw_sum = np.zeros((4, 3)); tw_sum = np.zeros((4, 3)); cpw_sum = np.zeros((4, 3)); fzw_sum = np.zeros(4)
        fz_s = np.zeros(4)
        for _ in range(nsub):
            mujoco.mj_step(m, d)
            if cfg.imu_mode == "averaged":
                acc_sum += d.sensordata[sadr["imu_acc"][0]:sadr["imu_acc"][0] + 3]
                gyr_sum += d.sensordata[sadr["imu_gyro"][0]:sadr["imu_gyro"][0] + 3]
            fz_s, fw_s, tw_s, cpw_s, fsum_s = _foot_contact_forces(m, d, floor, feet, leg_bodies)   # per-substep wrenches (averaged below)
            fw_sum += fw_s; tw_sum += tw_s; cpw_sum += cpw_s; fzw_sum += fsum_s
        tau_meas = tau_app + rng.normal(0.0, 1.0, 12) * faults.torque_meas_noise_std(noise.torque_meas_std, t)
        if cfg.imu_mode == "sampled":
            acc0 = d.sensordata[sadr["imu_acc"][0]:sadr["imu_acc"][0] + 3].copy()
            gyr0 = d.sensordata[sadr["imu_gyro"][0]:sadr["imu_gyro"][0] + 3].copy()
        elif cfg.imu_mode == "averaged":
            acc0 = acc_sum / nsub; gyr0 = gyr_sum / nsub
        else:                                   # integrating IMU: exact mean specific force / angular rate over the step
            mujoco.mj_forward(m, d)             # sensors at the post-step state (velocity/quaternion of the imu site)
            q_new = d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4]
            v_new = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
            R_prev = np.zeros(9); mujoco.mju_quat2Mat(R_prev, q_prev); R_prev = R_prev.reshape(3, 3)
            R_new = np.zeros(9); mujoco.mju_quat2Mat(R_new, q_new); R_new = R_new.reshape(3, 3)
            dR = R_prev.T @ R_new
            ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            gyr0 = (axis / (2 * np.sin(ang)) * ang / cfg.ctrl_dt) if ang > 1e-12 else np.zeros(3)
            R_mid = R_prev @ _exp_so3(0.5 * gyr0 * cfg.ctrl_dt)
            acc0 = R_mid.T @ ((v_new - v_prev) / cfg.ctrl_dt - m.opt.gravity)
        if cfg.weld_base:                       # static base: MuJoCo's accelerometer on a jointless body reads 0; report the
            Rw = np.zeros(9); mujoco.mju_quat2Mat(Rw, d.sensordata[sadr["base_quat"][0]:sadr["base_quat"][0] + 4]); Rw = Rw.reshape(3, 3)
            acc0 = Rw.T @ (-m.opt.gravity)       # physical specific force (+g in the body frame at rest)
        acc = acc0 + rng.normal(0.0, noise.imu_acc_std, 3)
        gyr = gyr0 + rng.normal(0.0, noise.imu_gyro_std, 3)
        gyr_prev = gyr
        c = np.zeros(4)
        fz = fz_s                                                                    # end-of-step normal force -> contact flag (S1/S2 semantics)
        fw = fw_sum / nsub; tw0 = tw_sum / nsub                                      # mean world contact wrench over the control step (torque about the world origin)
        cpw = np.zeros((4, 3)); tw = np.zeros((4, 3))
        for j, f in enumerate(feet):
            cpw[j] = cpw_sum[j] / fzw_sum[j] if fzw_sum[j] > 0 else d.geom_xpos[f] - np.array([0.0, 0.0, m.geom_size[f][0]])
            tw[j] = tw0[j] - np.cross(cpw[j], fw[j])                                 # torque about the leg's reference contact point
        c[:] = (fz > cfg.contact_force_thresh).astype(float) if cfg.contact_force_thresh > 0 else (fz > 0).astype(float)
        temp = a_temp * temp + (1 - a_temp) * (tau_app.reshape(4, 3) ** 2).mean(axis=1)
        base_lin = d.sensordata[sadr["base_linvel"][0]:sadr["base_linvel"][0] + 3]
        foot_w = np.concatenate([d.geom_xpos[g] for g in feet])
        rows[k, :] = np.concatenate([[d.time, theta], q_meas, dq_meas, tau_cmd, tau_meas, acc, gyr, c, temp,
                                     d.xpos[base_id], d.xquat[base_id], base_lin, foot_w, q_ref, fw.ravel(), cpw.ravel(), tw.ravel()])
    df = pd.DataFrame(rows, columns=all_columns())
    manifest = build_manifest(sim_meta={"model": cfg.model, "weld_base": cfg.weld_base, "joint_damping": cfg.joint_damping,
                                        "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt, "period_s": T,
                                        "gait": cfg.gait, "speed": cfg.speed, "terrain": cfg.terrain,
                                        "slope_deg": cfg.slope_deg, "slope_axis": cfg.slope_axis, "seed": cfg.seed,
                                        "phase_offset": cfg.phase_offset,
                                        "imu_mode": cfg.imu_mode, "foot_contact": cfg.foot_contact,
                                        "contact_force_thresh": cfg.contact_force_thresh,
                                        "record_time": "end of control step (state at t+ctrl_dt; tau_cmd from t)"})
    if return_state:
        return df, manifest, (d.qpos.copy(), d.qvel.copy())
    return df, manifest


def _exp_so3(phi):
    a = np.linalg.norm(phi)
    K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    if a < 1e-10:
        return np.eye(3) + K
    return np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a**2 * K @ K


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


def load_model(cfg: "SimConfig") -> mujoco.MjModel:
    """MjModel for cfg.model / cfg.terrain; weld_base replaces the trunk's free joint by a weld to the world (the
    robot hangs at the keyframe height with the legs free: each leg is a fixed-base 3-dof manipulator)."""
    path = scene_path(cfg.terrain, cfg.model)
    if not cfg.weld_base:
        return mujoco.MjModel.from_xml_path(path)
    spec = mujoco.MjSpec.from_file(path)
    spec.compiler.fusestatic = False              # keep 'base' as a (static) body: telemetry/IMU/observer refer to it by name
    base = spec.body("base")
    for j in list(base.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    base.pos = np.array([0.0, 0.0, 0.45])
    for k in list(spec.keys):
        spec.delete(k)                                # keyframes refer to the old qpos layout
    return spec.compile()


def keyframe_state(model: mujoco.MjModel | None = None, sim_model: str = "go2_menagerie_sym"):
    """(qpos, qvel) of the symmetric 'home' keyframe — a symmetric initial condition."""
    m = model if model is not None else mujoco.MjModel.from_xml_path(scene_path(model=sim_model))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return d.qpos.copy(), d.qvel.copy()
