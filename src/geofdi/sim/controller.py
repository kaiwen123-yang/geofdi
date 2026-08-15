"""Open-loop, phase-driven PD trot controller for the Go2 that is Σ-equivariant *by construction*.

One joint-space reference template T(θ) ∈ R^3 = (HAA, HFE, KFE) is defined for a LEFT leg (uniform-axis
joint convention: abduction axis x, thigh/calf axis y for all legs). The four legs are generated from it
by the gait symmetry group Σ = {(e,0), (g_s, ½)} of the trot (theory Part 0, Section "Spatio-temporal
gait symmetry"):

    LF(θ) = T(θ)            LH(θ) = T(θ + ½)
    RF(θ) = S·T(θ + ½)      RH(θ) = S·T(θ)          S = diag(-1, +1, +1)

so RF is the mirror of LF half a period later (A2 holds exactly: no second set of hand-written
parameters exists), and diagonal legs (LF,RH), (RF,LH) are in phase — a trot. Torques are
τ = kp (q_ref − q) + kd (dq_ref − dq), clipped to the motor limits.

`AsymmetrySpec` entries break the construction on purpose (e01c): a per-leg/joint gain on kp/kd or a
setpoint bias, active from `t_start` (later entries for the same leg/joint override earlier ones).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEGS = ("LF", "RF", "LH", "RH")          # GeoFDI / M1 leg order
JOINTS = ("HAA", "HFE", "KFE")
MIRROR_LEG = {"LF": "RF", "RF": "LF", "LH": "RH", "RH": "LH"}
S_MIRROR = np.array([-1.0, 1.0, 1.0])    # uniform-axis convention: HAA flips, HFE/KFE keep sign
LEG_INDEX = {l: i for i, l in enumerate(LEGS)}
JOINT_INDEX = {j: i for i, j in enumerate(JOINTS)}


@dataclass
class AsymmetrySpec:
    leg: str = "LF"
    joint: str = "HFE"
    kp_gain: float = 1.0
    kd_gain: float = 1.0
    setpoint_bias: float = 0.0
    t_start: float = 0.0
    t_end: float = float("inf")


@dataclass
class TrotParams:
    period_s: float = 0.5
    duty: float = 0.5                      # stance fraction of the template leg
    q_stand: tuple = (0.0, 0.9, -1.8)      # HAA, HFE, KFE nominal (Go2 'home' keyframe)
    lift_kfe: float = 0.45                 # extra knee flexion at mid-swing (rad)  -> foot lift
    lift_hfe: float = 0.20                 # thigh forward at mid-swing (rad)
    leg_len: float = 0.30                  # effective leg length used to convert speed -> sweep [m]
    speed: float = 0.0                     # commanded forward speed [m/s] (0 = trot in place)
    kp: tuple = (60.0, 60.0, 60.0)
    kd: tuple = (1.5, 1.5, 1.5)
    tau_max: tuple = (23.7, 23.7, 45.43)
    asymmetry: list = field(default_factory=list)   # list[AsymmetrySpec]


class TrotController:
    def __init__(self, p: TrotParams):
        self.p = p
        self.q0 = np.asarray(p.q_stand, dtype=float)
        self.sweep = p.speed * p.duty * p.period_s / (2.0 * p.leg_len)   # HFE half-amplitude [rad]
        self.kp = np.asarray(p.kp, dtype=float)
        self.kd = np.asarray(p.kd, dtype=float)
        self.tau_max = np.asarray(p.tau_max, dtype=float)
        self.asym = [a if isinstance(a, AsymmetrySpec) else AsymmetrySpec(**a) for a in p.asymmetry]

    # ---- template (left leg) ----------------------------------------------------------------
    def template(self, theta: np.ndarray | float) -> np.ndarray:
        """Reference joint angles of the LEFT-leg template at phase theta (mod 1). Shape (..., 3)."""
        th = np.mod(np.asarray(theta, dtype=float), 1.0)
        d = self.p.duty
        stance = th < d
        s_st = th / d
        s_sw = (th - d) / (1.0 - d)
        haa = np.zeros_like(th) + self.q0[0]
        hfe = np.where(stance, self.q0[1] + self.sweep * (2 * s_st - 1),
                       self.q0[1] + self.sweep * (1 - 2 * s_sw) - self.p.lift_hfe * np.sin(np.pi * s_sw))
        kfe = np.where(stance, self.q0[2] + 0 * th, self.q0[2] - self.p.lift_kfe * np.sin(np.pi * s_sw))
        return np.stack([haa, hfe, kfe], axis=-1)

    def reference(self, theta: float, with_velocity: bool = True):
        """q_ref (12,) and dq_ref (12,) for all legs at gait phase theta, ordered LF,RF,LH,RH x HAA,HFE,KFE."""
        def q_all(th):
            T0 = self.template(th)
            T1 = self.template(th + 0.5)
            return np.concatenate([T0, S_MIRROR * T1, T1, S_MIRROR * T0])   # LF, RF, LH, RH
        q = q_all(theta)
        if not with_velocity:
            return q, None
        eps = 1e-4
        dq = (q_all(theta + eps) - q_all(theta - eps)) / (2 * eps * self.p.period_s)
        return q, dq

    # ---- control law ------------------------------------------------------------------------
    def gains(self, t: float):
        kp = np.tile(self.kp, 4).copy()
        kd = np.tile(self.kd, 4).copy()
        bias = np.zeros(12)
        for a in self.asym:
            if a.t_start <= t < a.t_end:
                i = 3 * LEG_INDEX[a.leg] + JOINT_INDEX[a.joint]
                kp[i] = self.kp[JOINT_INDEX[a.joint]] * a.kp_gain
                kd[i] = self.kd[JOINT_INDEX[a.joint]] * a.kd_gain
                bias[i] = a.setpoint_bias
        return kp, kd, bias

    def torque(self, q: np.ndarray, dq: np.ndarray, theta: float, t: float,
               setpoint_offset: np.ndarray | None = None):
        """PD torque command (12,) given measured joint state (12,) and gait phase; returns (tau, q_ref, dq_ref)."""
        q_ref, dq_ref = self.reference(theta)
        kp, kd, bias = self.gains(t)
        q_ref = q_ref + bias
        if setpoint_offset is not None:
            q_ref = q_ref + setpoint_offset
        tau = kp * (q_ref - q) + kd * (dq_ref - dq)
        lim = np.tile(self.tau_max, 4)
        return np.clip(tau, -lim, lim), q_ref, dq_ref
