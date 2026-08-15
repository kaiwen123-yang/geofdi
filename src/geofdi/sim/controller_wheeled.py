"""Rolling-mode controller for the wheeled M1 (Sprint 7 Block W1), Σ = G (pure sagittal reflection, no phase).

Legs: PD hold of one stand posture in the uniform-axis convention (ABAD 0, HIP 0.8, KNEE -1.5 on all four legs; the
mirror image of that posture is itself, since S = diag(-1, +1, +1) fixes (0, ., .)). Wheels: velocity PD to
omega_ref = v_cmd / r on all four wheels (the wheel axis is +y on every leg, so "forward" is the same sign left and
right — consistent with the manifest sign +1 of the WHEEL channels), plus an equivariant yaw damper (differential wheel
torque +k_yaw * w_z on the left, -k_yaw * w_z on the right, which opposes a CCW yaw rate: under the reflection w_z flips
and left/right swap, so the rule maps to itself) and an equivariant roll stabilizer on the ABAD joints (same gains all legs). The speed command
ramps from 0 over `ramp_s` seconds. Everything is identical across the mirror pair by construction (A2 holds exactly);
`AsymmetryWheeled` entries break it on purpose (eps_ctrl injections for e01-W / H0'):
    {"leg": "RF", "joint": "WHEEL", "rate_gain": 1.02}     -> that wheel's omega_ref x 1.02
    {"leg": "LF", "joint": "HIP", "kp_gain": 1.02}          -> that joint's kp x 1.02

    ctrl = RollingController(RollingParams(speed=1.0))
    tau16, ref16 = ctrl.torque(q16, dq16, t, body={"roll":..., "w_x":..., "w_z":..., "v_y":...})   # GeoFDI order
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .telemetry_m1 import JOINTS, LEGS, WHEEL_R

LEG_INDEX = {l: i for i, l in enumerate(LEGS)}
JOINT_INDEX = {j: i for i, j in enumerate(JOINTS)}
LEFT = ("LF", "LH"); RIGHT = ("RF", "RH")


@dataclass
class AsymmetryWheeled:
    leg: str = "RF"
    joint: str = "WHEEL"
    rate_gain: float = 1.0        # wheel omega_ref multiplier (WHEEL only)
    kp_gain: float = 1.0          # leg PD kp multiplier
    kd_gain: float = 1.0
    setpoint_bias: float = 0.0    # rad added to the leg setpoint (legs) or rad/s to omega_ref (WHEEL)
    t_start: float = 0.0
    t_end: float = float("inf")


@dataclass
class RollingParams:
    speed: float = 0.0                       # commanded forward speed [m/s]
    ramp_s: float = 1.0                      # speed ramp from 0
    q_stand: tuple = (0.0, 0.8, -1.5)        # ABAD, HIP, KNEE
    kp: tuple = (150.0, 150.0, 150.0)
    kd: tuple = (4.0, 4.0, 4.0)
    kv_wheel: float = 3.0                    # N m per rad/s of wheel-rate error
    wheel_r: float = WHEEL_R
    k_yaw: float = 50.0                      # differential wheel torque per rad/s yaw rate (equivariant). W2 finding: with 1.0 a lightly
                                             # damped yaw/lateral mode leaves some realizations with block-to-block dependence (flip-test size
                                             # 0.17-0.4 at 1-2 m/s); 50 (0.5 N m per 0.01 rad/s) restores exchangeable blocks (size in band).
    stab_k_roll: float = 0.0                 # ABAD setpoint offset per rad of body roll (all legs, same sign) — equivariant
    stab_k_wx: float = 0.0                   # ... per rad/s roll rate
    stab_k_vy: float = 0.0                   # ... per m/s of body-frame lateral velocity (kills the slow lateral-creep mode)
    stab_max: float = 0.15                   # |ABAD offset| clip [rad]
    tau_max: tuple = (40.0, 60.0, 60.0, 20.0)  # ctrlrange (ABAD, HIP, KNEE, WHEEL)
    asymmetry: list = field(default_factory=list)


class RollingController:
    def __init__(self, p: RollingParams):
        self.p = p
        self.asym = [a if isinstance(a, AsymmetryWheeled) else AsymmetryWheeled(**a) for a in (p.asymmetry or [])]

    def _speed(self, t: float) -> float:
        if self.p.ramp_s <= 0:
            return self.p.speed
        return self.p.speed * float(min(1.0, max(0.0, t / self.p.ramp_s)))

    def references(self, t: float) -> np.ndarray:
        """(16,) leg setpoints (ABAD/HIP/KNEE) and wheel omega_ref (WHEEL slot) in GeoFDI order."""
        ref = np.zeros(16)
        w_ref = self._speed(t) / self.p.wheel_r
        for li in range(4):
            ref[4 * li:4 * li + 3] = self.p.q_stand
            ref[4 * li + 3] = w_ref
        return ref

    def torque(self, q: np.ndarray, dq: np.ndarray, t: float, body: dict | None = None):
        p = self.p; ref = self.references(t)
        kp = np.tile(np.array(p.kp), 4); kd = np.tile(np.array(p.kd), 4)      # per (leg, non-wheel joint) -> map below
        tau = np.zeros(16)
        # equivariant roll stabilizer on ABAD (all legs, same sign): body roll > 0 (left down) -> ABAD offset
        off = np.zeros(16)
        if body is not None and (p.stab_k_roll or p.stab_k_wx or p.stab_k_vy):
            d = p.stab_k_roll * body.get("roll", 0.0) + p.stab_k_wx * body.get("w_x", 0.0) + p.stab_k_vy * body.get("v_y", 0.0)
            off[0::4] = float(np.clip(d, -p.stab_max, p.stab_max))
        # asymmetries (eps_ctrl injections)
        kp_mul = np.ones(16); kd_mul = np.ones(16); rate_mul = np.ones(4); bias = np.zeros(16)
        for a in self.asym:
            if not (a.t_start <= t < a.t_end):
                continue
            li = LEG_INDEX[a.leg]; ji = JOINT_INDEX[a.joint]; k = 4 * li + ji
            if a.joint == "WHEEL":
                rate_mul[li] *= a.rate_gain; bias[k] += a.setpoint_bias
            else:
                kp_mul[k] *= a.kp_gain; kd_mul[k] *= a.kd_gain; bias[k] += a.setpoint_bias
        for li in range(4):
            for ji in range(3):
                k = 4 * li + ji; g = 3 * li + ji
                tau[k] = kp[g] * kp_mul[k] * (ref[k] + off[k] + bias[k] - q[k]) - kd[g] * kd_mul[k] * dq[k]
            k = 4 * li + 3
            tau[k] = p.kv_wheel * (ref[k] * rate_mul[li] + bias[k] - dq[k])
        # equivariant yaw damper (differential wheel torque)
        if body is not None and p.k_yaw:
            wz = body.get("w_z", 0.0)
            for li, leg in enumerate(LEGS):
                tau[4 * li + 3] += (+p.k_yaw * wz) if leg in LEFT else (-p.k_yaw * wz)      # damp: w_z > 0 (CCW) -> left faster, right slower
        lim = np.tile(np.array(p.tau_max), 4)
        return np.clip(tau, -lim, lim), ref
