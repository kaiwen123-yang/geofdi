"""Generalized-momentum observer (De Luca & Mattone 2003; Haddadin, De Luca & Albu-Schäffer 2017) for a
floating-base legged robot, in Pinocchio velocity coordinates (base linear velocity in the body frame).

Model:  M(q) dv + C(q,v) v + g(q) = S^T tau + tau_passive(v) + sum_i J_i(q)^T f_i + tau_ext
        p = M(q) v,   dp/dt = S^T tau + tau_passive + sum_i J_i^T f_i + C^T v - g + tau_ext        (Mdot = C + C^T)
Observer: r = K_O (p - p_hat),   d p_hat/dt = S^T tau + tau_passive + sum_i J_i^T f_i + C^T v - g + r
        =>  r = (K_O / (s + K_O)) tau_ext : a first-order low-pass of the unmodelled generalized force with cut-off K_O.

Inputs per step: MuJoCo-convention state (qpos, qvel), commanded joint torques (12), world-frame contact forces per
foot (4x3; sim: read from MuJoCo, hardware: estimated — the `contact_wrench` interface), contact points (4x3, world),
contact flags. Output: r in R^18 (6 base + 12 joints); the experiments use the 12 joint components, which are the same
in either velocity convention. tau_ext contains everything the nominal model does not: actuator gain/bias faults
(when tau = commanded torque), friction changes, added inertia, model error, unmodelled contacts.
"""
from __future__ import annotations

import numpy as np

from .pin_model import Go2Dynamics, quat_wxyz_to_R, _B


class MomentumObserver:
    def __init__(self, dyn: Go2Dynamics, dt: float, cutoff_hz: float = 10.0, use_contacts: bool = True, use_passive: bool = True):
        assert dyn.backend == "pin", "the momentum observer needs the Pinocchio backend (Christoffel-consistent C)"
        self.dyn = dyn; self.dt = float(dt); self.K = 2 * np.pi * float(cutoff_hz)
        self.use_contacts = use_contacts; self.use_passive = use_passive
        self.reset()

    def reset(self):
        self.p_hat = None; self.r = np.zeros(18)

    def _point_jacobians(self, qpos_mj, points_world):
        """(4, 3, 18) world-frame translational Jacobians of given world points attached to the foot frames and
        (4, 3, 18) world-frame angular Jacobians of the foot frames, PIN coords."""
        pin = self.dyn.pin; q = self.dyn._to_pin_q(qpos_mj)
        pin.computeJointJacobians(self.dyn.model, self.dyn.data, q); pin.updateFramePlacements(self.dyn.model, self.dyn.data)
        J = np.zeros((4, 3, 18)); Jw = np.zeros((4, 3, 18))
        for i, f in enumerate(self.dyn.foot_fids):
            Jf = pin.getFrameJacobian(self.dyn.model, self.dyn.data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            d = np.asarray(points_world[i]) - self.dyn.data.oMf[f].translation
            dx = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
            J[i] = Jf[:3] - dx @ Jf[3:]; Jw[i] = Jf[3:]
        return J, Jw

    def step(self, qpos_mj, qvel_mj, tau_joint, foot_forces=None, contact_points=None, contact_flags=None, foot_torques=None):
        dyn = self.dyn; pin = dyn.pin
        q = dyn._to_pin_q(qpos_mj); R = quat_wxyz_to_R(qpos_mj[3:7]); B = _B(R)
        v = B @ np.asarray(qvel_mj, dtype=float)
        M = pin.crba(dyn.model, dyn.data, q); M = np.triu(M) + np.triu(M, 1).T
        C = np.array(pin.computeCoriolisMatrix(dyn.model, dyn.data, q, v))
        g = np.array(pin.computeGeneralizedGravity(dyn.model, dyn.data, q))
        p = M @ v
        if self.p_hat is None:
            self.p_hat = p.copy(); self.r = np.zeros(18)
            return self.r.copy()
        tau_gen = np.zeros(18); tau_gen[6:] = np.asarray(tau_joint, dtype=float)
        if self.use_passive:
            tau_gen += dyn.passive_torque(v)                        # joints only; v joints == qvel joints
        if self.use_contacts and foot_forces is not None:
            flags = np.ones(4, dtype=bool) if contact_flags is None else np.asarray(contact_flags) > 0.5
            if flags.any():
                pts = contact_points if contact_points is not None else dyn.foot_positions(qpos_mj)
                J, Jw = self._point_jacobians(qpos_mj, pts)
                for i in range(4):
                    if flags[i]:
                        tau_gen += J[i].T @ np.asarray(foot_forces[i], dtype=float)
                        if foot_torques is not None:
                            tau_gen += Jw[i].T @ np.asarray(foot_torques[i], dtype=float)
        rhs = tau_gen + C.T @ v - g + self.r
        self.p_hat = self.p_hat + self.dt * rhs
        self.r = self.K * (p - self.p_hat)
        return self.r.copy()


IMU_OFFSET = (-0.02557, 0.0, 0.04232)      # URDF imu link in the base frame (== the MJCF 'imu' site)


def run_observer(df, dyn: Go2Dynamics, dt: float = 0.005, cutoff_hz: float = 10.0, torque: str = "tau_cmd",
                 base_velocity: str = "truth", use_contacts: bool = True, use_contact_torque: bool = True,
                 imu_offset=IMU_OFFSET) -> np.ndarray:
    """Run the observer over an M1-schema telemetry frame; returns r (T, 18) in PIN coordinates (joints = r[:, 6:]).

    torque: 'tau_cmd' (commanded == current-based estimate; actuator faults become visible) | 'tau_meas' (measured
    output torque; physically consistent -> blind to gain/bias faults). Base pose from the truth columns, base linear
    velocity from the truth (base_v*) or the framelinvel sensor, angular velocity from the measured gyro (body frame).
    """
    from ..sim.telemetry import JOINTS, LEGS
    q = df[[f"q_{l}_{j}" for l in LEGS for j in JOINTS]].to_numpy(); dq = df[[f"dq_{l}_{j}" for l in LEGS for j in JOINTS]].to_numpy()
    tau = df[[f"{torque}_{l}_{j}" for l in LEGS for j in JOINTS]].to_numpy()
    pos = df[["base_x", "base_y", "base_z"]].to_numpy(); quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy()
    vlin = df[["base_vx", "base_vy", "base_vz"]].to_numpy(); gyr = df[["imu_w_x", "imu_w_y", "imu_w_z"]].to_numpy()
    has_fc = f"fc_x_{LEGS[0]}" in df.columns
    fc = np.stack([df[[f"fc_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1) if has_fc else None
    cp = np.stack([df[[f"cp_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1) if has_fc else None
    has_tc = f"tc_x_{LEGS[0]}" in df.columns
    tc = np.stack([df[[f"tc_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1) if (has_fc and has_tc and use_contact_torque) else None
    # gate the contact term by the recorded (step-averaged) force itself, not by the end-of-step contact flag: the
    # lift-off step still carries force, the touchdown step's flag is already 1
    cflag = (np.linalg.norm(fc, axis=2) > 0).astype(float) if has_fc else df[[f"c_{l}" for l in LEGS]].to_numpy()
    obs = MomentumObserver(dyn, dt, cutoff_hz, use_contacts=use_contacts and has_fc)
    out = np.zeros((len(df), 18))
    # base_v* is the framelinvel of the IMU site; the free-joint velocity is that of the base-body origin:
    # v_origin = v_imu - omega_world x (R r_imu)
    r_imu = np.asarray(imu_offset, dtype=float)
    for k in range(len(df)):
        R = quat_wxyz_to_R(quat[k]); v_origin = vlin[k] - np.cross(R @ gyr[k], R @ r_imu)
        qpos = np.concatenate([pos[k], quat[k], q[k]]); qvel = np.concatenate([v_origin, gyr[k], dq[k]])
        out[k] = obs.step(qpos, qvel, tau[k], None if fc is None else fc[k], None if cp is None else cp[k], cflag[k],
                          None if tc is None else tc[k])
    return out
