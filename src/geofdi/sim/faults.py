"""Fault and nuisance injectors for the Go2 simulation (uniform interface).

    FaultSpec(type, t_onset, leg, joint, magnitude, schedule, ramp_s, params)

Faults (act on the leg/joint given, or on all if None):
  actuator_gain   applied torque *= 1 + magnitude*s(t)              (magnitude -0.4 == 60 % retention)
  actuator_bias   applied torque += magnitude*s(t)   [N m]
  deadzone        applied torque := 0 where |tau| < magnitude*s(t) [N m]
  delay           applied torque delayed by magnitude [s] (rounded to control steps; step schedule only)
  friction_scale  joint damping & frictionloss *= 1 + magnitude*s(t)
  inertia_add     calf mass += magnitude*s(t) [kg] (inertia scaled with the mass)
  encoder_bias    measured q += magnitude*s(t) [rad]
  foot_friction   foot geom sliding friction *= 1 + magnitude*s(t)
Nuisances (symmetric or symmetric-in-law; not faults):
  payload_symmetric   base mass += magnitude [kg] at the base COM
  payload_asymmetric  base mass += magnitude [kg] with COM shifted by params.offset_y [m] (default 0.05)
  drift_symmetric     Ornstein-Uhlenbeck factor g_t (std = magnitude, time constant params.tau_s, default 20 s):
                      all applied torques *= 1+g_t and joint damping/frictionloss *= 1+g_t (temperature-like)
  drift_lateral       OU offset a_t [rad] (std = magnitude, params.tau_s) added to ALL HAA setpoints: a common
                      lateral lean that wanders; zero-mean and mirror-symmetric IN LAW, antisymmetric in each
                      realization, autocorrelated across cycles — the correlation stressor for e01b.
s(t): schedule 'step' -> 1[t>=t_onset]; 'ramp' -> clip((t-t_onset)/ramp_s, 0, 1); 'pulse' -> 1 on
[t_onset, t_onset+params.duration_s). Model-parameter injectors write into the MjModel every control step
(cheap) so ramps work; the pristine values are cached at construction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
FAULT_TYPES = ("actuator_gain", "actuator_bias", "deadzone", "delay", "friction_scale", "inertia_add",
               "encoder_bias", "foot_friction")
NUISANCE_TYPES = ("payload_symmetric", "payload_asymmetric", "drift_symmetric", "drift_lateral")


@dataclass
class FaultSpec:
    type: str
    t_onset: float = 0.0
    leg: str | None = None
    joint: str | None = None
    magnitude: float = 0.0
    schedule: str = "step"
    ramp_s: float = 0.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in FAULT_TYPES + NUISANCE_TYPES:
            raise ValueError(f"unknown injector type {self.type!r}")
        if self.schedule not in ("step", "ramp", "pulse"):
            raise ValueError(f"unknown schedule {self.schedule!r}")

    def s(self, t: float) -> float:
        if self.schedule == "step":
            return 1.0 if t >= self.t_onset else 0.0
        if self.schedule == "ramp":
            if t < self.t_onset:
                return 0.0
            return float(min(1.0, (t - self.t_onset) / max(self.ramp_s, 1e-9)))
        dur = float(self.params.get("duration_s", 1.0))
        return 1.0 if self.t_onset <= t < self.t_onset + dur else 0.0

    def mask(self) -> np.ndarray:
        """Boolean (12,) mask over LF,RF,LH,RH x HAA,HFE,KFE."""
        m = np.zeros(12, dtype=bool)
        for i, leg in enumerate(LEGS):
            for j, jn in enumerate(JOINTS):
                if (self.leg in (None, leg)) and (self.joint in (None, jn)):
                    m[3 * i + j] = True
        return m


class FaultBank:
    """Composes injectors; call order per control step: setpoint_offset -> torque -> measure; model_update()."""

    def __init__(self, specs, model, ctrl_dt: float, rng: np.random.Generator):
        self.specs = [s if isinstance(s, FaultSpec) else FaultSpec(**s) for s in (specs or [])]
        self.model = model
        self.dt = ctrl_dt
        self.rng = rng
        self._delay_buf = {}
        self._ou = {}                       # id(spec) -> current OU state
        # cache pristine model values
        self._dof_damping0 = model.dof_damping.copy()
        self._dof_frictionloss0 = model.dof_frictionloss.copy()
        self._body_mass0 = model.body_mass.copy()
        self._body_inertia0 = model.body_inertia.copy()
        self._body_ipos0 = model.body_ipos.copy()
        self._geom_friction0 = model.geom_friction.copy()
        import mujoco
        self._mj = mujoco
        self._calf_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
        self._foot_geom = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL", "FR", "RL", "RR")]
        self._base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        jn = [f"{l}_{j}_joint" for l in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")]
        self._dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn])
        self.has_model_faults = any(s.type in ("friction_scale", "inertia_add", "foot_friction", "payload_symmetric",
                                               "payload_asymmetric", "drift_symmetric") for s in self.specs)

    # ---- stochastic drift states --------------------------------------------------------------
    def _ou_step(self, spec: FaultSpec) -> float:
        tau = float(spec.params.get("tau_s", 20.0))
        sig = float(spec.magnitude)
        x = self._ou.get(id(spec), 0.0)
        a = np.exp(-self.dt / tau)
        x = a * x + sig * np.sqrt(1 - a * a) * self.rng.standard_normal()
        self._ou[id(spec)] = x
        return x

    def advance(self, t: float) -> None:
        """Advance stochastic drift states once per control step (call before torque())."""
        self._g_sym = 0.0
        self._a_lat = 0.0
        for s in self.specs:
            if s.type == "drift_symmetric":
                x = self._ou_step(s)
                if s.s(t) > 0:
                    self._g_sym += x * s.s(t)
            elif s.type == "drift_lateral":
                x = self._ou_step(s)
                if s.s(t) > 0:
                    self._a_lat += x * s.s(t)

    # ---- per-step hooks ---------------------------------------------------------------------------
    def setpoint_offset(self, t: float) -> np.ndarray:
        off = np.zeros(12)
        if self._a_lat != 0.0:
            off[0::3] = self._a_lat           # all HAA
        return off

    def torque(self, tau_cmd: np.ndarray, t: float) -> np.ndarray:
        """Actuator chain: delay (on the command) -> gain -> bias -> deadzone -> symmetric drift factor."""
        tau = tau_cmd.copy()
        for s in self.specs:                      # 1) delays act on the command signal
            if s.type == "delay":
                n = round(s.magnitude / self.dt)
                if n > 0:
                    buf = self._delay_buf.setdefault(id(s), deque([tau_cmd.copy()] * n, maxlen=n))
                    delayed = buf[0].copy()
                    buf.append(tau_cmd.copy())
                    if s.s(t):
                        m = s.mask(); tau[m] = delayed[m]
        for s in self.specs:                      # 2) multiplicative / additive / deadzone faults
            sv = s.s(t)
            if not sv:
                continue
            if s.type == "actuator_gain":
                m = s.mask(); tau[m] = tau[m] * (1.0 + s.magnitude * sv)
            elif s.type == "actuator_bias":
                m = s.mask(); tau[m] = tau[m] + s.magnitude * sv
            elif s.type == "deadzone":
                m = s.mask(); dz = s.magnitude * sv
                tau[m] = np.where(np.abs(tau[m]) < dz, 0.0, tau[m])
        if self._g_sym != 0.0:                    # 3) temperature-like symmetric drift on all motors
            tau = tau * (1.0 + self._g_sym)
        return tau

    def measure(self, q: np.ndarray, t: float) -> np.ndarray:
        q_meas = q.copy()
        for s in self.specs:
            if s.type == "encoder_bias":
                sv = s.s(t)
                if sv:
                    q_meas[s.mask()] += s.magnitude * sv
        return q_meas

    def model_update(self, t: float) -> None:
        if not self.has_model_faults:
            return
        m = self.model
        m.dof_damping[:] = self._dof_damping0
        m.dof_frictionloss[:] = self._dof_frictionloss0
        m.body_mass[:] = self._body_mass0
        m.body_inertia[:] = self._body_inertia0
        m.body_ipos[:] = self._body_ipos0
        m.geom_friction[:] = self._geom_friction0
        for s in self.specs:
            sv = s.s(t)
            if s.type == "friction_scale" and sv:
                dof = self._dof[s.mask()]
                m.dof_damping[dof] = self._dof_damping0[dof] * (1 + s.magnitude * sv)
                m.dof_frictionloss[dof] = self._dof_frictionloss0[dof] * (1 + s.magnitude * sv)
            elif s.type == "inertia_add" and sv:
                for i, leg in enumerate(LEGS):
                    if s.leg in (None, leg):
                        b = self._calf_body[i]
                        m0 = self._body_mass0[b]
                        m.body_mass[b] = m0 + s.magnitude * sv
                        m.body_inertia[b] = self._body_inertia0[b] * (m.body_mass[b] / m0)
            elif s.type == "foot_friction" and sv:
                for i, leg in enumerate(LEGS):
                    if s.leg in (None, leg):
                        g = self._foot_geom[i]
                        m.geom_friction[g, 0] = self._geom_friction0[g, 0] * (1 + s.magnitude * sv)
            elif s.type in ("payload_symmetric", "payload_asymmetric") and sv:
                b = self._base_body
                m0 = self._body_mass0[b]; madd = s.magnitude * sv
                m.body_mass[b] = m0 + madd
                if s.type == "payload_asymmetric":
                    off = float(s.params.get("offset_y", 0.05))
                    m.body_ipos[b] = (m0 * self._body_ipos0[b] + madd * (self._body_ipos0[b] + np.array([0, off, 0]))) / (m0 + madd)
            elif s.type == "drift_symmetric" and self._g_sym != 0.0:
                m.dof_damping[self._dof] = self._dof_damping0[self._dof] * (1 + self._g_sym)
                m.dof_frictionloss[self._dof] = self._dof_frictionloss0[self._dof] * (1 + self._g_sym)
