"""Rigid-body dynamics of the Go2 (go2_description URDF) for residual generation.

Backends
- 'pin'    : Pinocchio 3 (`pin`) floating-base model built from the URDF (JointModelFreeFlyer). Joint order == the S0
             telemetry order (FL, FR, RL, RR x hip, thigh, calf == LF, RF, LH, RH x HAA, HFE, KFE).
- 'mujoco' : the same quantities from MuJoCo (mj_fullM, mj_rne, jacobians) on the go2_urdf MJCF — the fallback when
             Pinocchio is unavailable, and the reference for the consistency test.

Coordinates. MuJoCo: qpos = [p_w, quat(w,x,y,z), q_j], qvel = [v_lin WORLD frame, omega BODY frame, dq].
Pinocchio: q = [p_w, quat(x,y,z,w), q_j], v = [v_lin BODY (local) frame, omega BODY, dq]. The map between the two
velocity coordinates is v_pin = B v_mj with B = blockdiag(R^T, I3, I12); mass matrix / generalized forces transform
as M_mj = B^T M_pin B, tau_mj = B^T tau_pin. Everything this module returns is in the MUJOCO convention (world-frame
base linear velocity), so residuals are directly comparable between the two backends and with the telemetry.

Passive joint torques (viscous damping b dq + Coulomb friction f sign(dq)) and rotor armature are part of the nominal
model (MJCF defaults: damping 0.01, frictionloss 0.2, armature 0.01 for the URDF world; damping 2.0 for the menagerie
world) and are exposed so the momentum observer can include them.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import mujoco
import numpy as np

LEG_ORDER = ("FL", "FR", "RL", "RR")            # == LF, RF, LH, RH
FOOT_FRAMES = tuple(f"{l}_foot" for l in LEG_ORDER)
FOOT_OFFSET = np.array([-0.002, 0.0, 0.0])      # sphere centre in the URDF foot frame (collision <origin xyz="-0.002 0 0">)


def default_urdf_path() -> str:
    return str(resources.files("geofdi.sim").joinpath("assets/go2_urdf/urdf/go2_description.urdf"))


def default_mjcf_path(sym: bool = False) -> str:
    return str(resources.files("geofdi.sim").joinpath(f"assets/go2_urdf/mjcf/scene_go2_urdf{'_sym' if sym else ''}.xml"))


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _B(R):
    B = np.eye(18); B[0:3, 0:3] = R.T
    return B


class Go2Dynamics:
    """Floating-base dynamics of the Go2 in MuJoCo coordinates. nv = 18 (6 base + 12 joints)."""

    def __init__(self, backend: str = "pin", urdf_path: str | None = None, mjcf_path: str | None = None,
                 armature: float = 0.01, damping: float = 0.01, frictionloss: float = 0.2, sym: bool = False):
        self.backend = backend
        self.armature = float(armature); self.damping = float(damping); self.frictionloss = float(frictionloss)
        self.nv = 18; self.nj = 12
        if backend == "pin":
            import pinocchio as pin
            self.pin = pin
            self.model = pin.buildModelFromUrdf(urdf_path or default_urdf_path(), pin.JointModelFreeFlyer())
            self.data = self.model.createData()
            arm = np.zeros(self.model.nv); arm[6:] = self.armature
            self.model.armature = arm
            self.foot_fids = [self.model.getFrameId(f) for f in FOOT_FRAMES]
            names = [self.model.names[i] for i in range(2, self.model.njoints)]
            expected = [f"{l}_{j}_joint" for l in LEG_ORDER for j in ("hip", "thigh", "calf")]
            assert names == expected, f"pinocchio joint order {names} != {expected}"
        elif backend == "mujoco":
            self.m = mujoco.MjModel.from_xml_path(mjcf_path or default_mjcf_path(sym))
            self.d = mujoco.MjData(self.m)
            self._foot_gids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, l) for l in LEG_ORDER]
            # armature/damping/frictionloss are read from the MJCF (dof_armature etc.); keep the given values in sync
            jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{l}_{j}_joint") for l in LEG_ORDER for j in ("hip", "thigh", "calf")]
            self._dofs = np.array([self.m.jnt_dofadr[j] for j in jids])
            self.m.dof_armature[self._dofs] = self.armature
            self.m.dof_damping[self._dofs] = self.damping
            self.m.dof_frictionloss[self._dofs] = self.frictionloss
        else:
            raise ValueError(backend)

    # ---- state conversion -------------------------------------------------------------------------
    @staticmethod
    def _to_pin_q(qpos_mj):
        q = np.array(qpos_mj, dtype=float).copy()
        w, x, y, z = qpos_mj[3:7]; q[3:7] = [x, y, z, w]
        return q

    def _pin_qv(self, qpos_mj, qvel_mj):
        R = quat_wxyz_to_R(qpos_mj[3:7]); B = _B(R)
        return self._to_pin_q(qpos_mj), B @ np.asarray(qvel_mj, dtype=float), B

    @staticmethod
    def _Bdot_v(qpos_mj, qvel_mj):
        """d/dt(B) v_mj: B = blockdiag(R^T, I, I) depends on q; d(R^T)/dt = -[omega]x R^T (omega body-frame) so the
        only non-zero block is -omega x (R^T v_lin_world)."""
        R = quat_wxyz_to_R(qpos_mj[3:7]); v = np.asarray(qvel_mj, dtype=float)
        out = np.zeros(18); out[0:3] = -np.cross(v[3:6], R.T @ v[0:3])
        return out

    # ---- Pinocchio-convention quantities (base linear velocity in the BODY frame) -----------------------
    # Used by the momentum observer: there C(q,v) is Christoffel-consistent (Mdot = C + C^T). Joint components of
    # generalized forces are identical in both conventions (B is the identity on the joints).
    def to_pin_velocity(self, qpos_mj, qvel_mj):
        return self._pin_qv(qpos_mj, qvel_mj)[1]

    def mass_matrix_pin(self, qpos_mj):
        q = self._to_pin_q(qpos_mj); M = self.pin.crba(self.model, self.data, q); return np.triu(M) + np.triu(M, 1).T

    def coriolis_matrix_pin(self, qpos_mj, qvel_mj):
        q, v, _ = self._pin_qv(qpos_mj, qvel_mj); return np.array(self.pin.computeCoriolisMatrix(self.model, self.data, q, v))

    def gravity_pin(self, qpos_mj):
        return np.array(self.pin.computeGeneralizedGravity(self.model, self.data, self._to_pin_q(qpos_mj)))

    def foot_jacobians_pin(self, qpos_mj):
        """(4, 3, 18) world-frame translational Jacobians of the foot points w.r.t. the PIN velocity coordinates."""
        R = quat_wxyz_to_R(qpos_mj[3:7]); B = _B(R)
        return np.einsum("fij,jk->fik", self.foot_jacobians(qpos_mj), np.linalg.inv(B))

    # ---- quantities (MuJoCo convention) ------------------------------------------------------------
    def mass_matrix(self, qpos_mj, qvel_mj=None) -> np.ndarray:
        if self.backend == "pin":
            q, v, B = self._pin_qv(qpos_mj, np.zeros(18) if qvel_mj is None else qvel_mj)
            M = self.pin.crba(self.model, self.data, q); M = np.triu(M) + np.triu(M, 1).T
            return B.T @ M @ B
        self.d.qpos[:] = qpos_mj; self.d.qvel[:] = 0.0; mujoco.mj_forward(self.m, self.d)
        M = np.zeros((self.m.nv, self.m.nv)); mujoco.mj_fullM(self.m, self.d, M)
        return M

    def gravity(self, qpos_mj) -> np.ndarray:
        """g(q): generalized gravity torque (the term on the left-hand side: M dv + C v + g = tau)."""
        if self.backend == "pin":
            q, v, B = self._pin_qv(qpos_mj, np.zeros(18))
            return B.T @ self.pin.computeGeneralizedGravity(self.model, self.data, q)
        self.d.qpos[:] = qpos_mj; self.d.qvel[:] = 0.0; self.d.qacc[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return self.d.qfrc_bias.copy()          # at zero velocity qfrc_bias = g(q) (passive/actuator not included)

    def bias(self, qpos_mj, qvel_mj) -> np.ndarray:
        """h(q, v) = C(q, v) v + g(q) — WITHOUT the passive joint torques (see passive_torque)."""
        if self.backend == "pin":
            q, v, B = self._pin_qv(qpos_mj, qvel_mj)
            M = self.pin.crba(self.model, self.data, q); M = np.triu(M) + np.triu(M, 1).T
            return B.T @ (self.pin.nle(self.model, self.data, q, v) + M @ self._Bdot_v(qpos_mj, qvel_mj))
        self.d.qpos[:] = qpos_mj; self.d.qvel[:] = qvel_mj; self.d.qacc[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return self.d.qfrc_bias.copy()

    def coriolis_matrix(self, qpos_mj, qvel_mj) -> np.ndarray:
        """A Coriolis matrix in the MuJoCo convention with C v + g = h: B^T C_pin B + B^T M_pin Dot(B) (the second term
        treats omega as given). NOT Christoffel-consistent in these coordinates — use coriolis_matrix_pin for the observer."""
        if self.backend != "pin":
            raise NotImplementedError("Coriolis matrix needs the Pinocchio backend (fallback: use bias() and finite-difference Mdot)")
        q, v, B = self._pin_qv(qpos_mj, qvel_mj)
        C = self.pin.computeCoriolisMatrix(self.model, self.data, q, v)
        M = self.pin.crba(self.model, self.data, q); M = np.triu(M) + np.triu(M, 1).T
        R = quat_wxyz_to_R(qpos_mj[3:7]); w = np.asarray(qvel_mj)[3:6]
        wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        Dm = np.zeros((18, 18)); Dm[0:3, 0:3] = -wx @ R.T
        return B.T @ C @ B + B.T @ M @ Dm

    def passive_torque(self, qvel_mj) -> np.ndarray:
        """Nominal passive joint torques (generalized, 18): -b dq - f sign(dq) on the 12 joints (0 on the base)."""
        out = np.zeros(18); dq = np.asarray(qvel_mj)[6:]
        out[6:] = -self.damping * dq - self.frictionloss * np.sign(dq)
        return out

    def foot_positions(self, qpos_mj) -> np.ndarray:
        """(4, 3) world-frame foot (sphere centre) positions."""
        if self.backend == "pin":
            q = self._to_pin_q(qpos_mj)
            self.pin.forwardKinematics(self.model, self.data, q); self.pin.updateFramePlacements(self.model, self.data)
            return np.array([self.data.oMf[f].translation + self.data.oMf[f].rotation @ FOOT_OFFSET for f in self.foot_fids])
        self.d.qpos[:] = qpos_mj; mujoco.mj_forward(self.m, self.d)
        return np.array([self.d.geom_xpos[g] for g in self._foot_gids])

    def foot_jacobians(self, qpos_mj) -> np.ndarray:
        """(4, 3, 18) translational Jacobians of the foot points, world frame, MuJoCo velocity coordinates."""
        if self.backend == "pin":
            q = self._to_pin_q(qpos_mj); R = quat_wxyz_to_R(qpos_mj[3:7]); B = _B(R)
            self.pin.computeJointJacobians(self.model, self.data, q); self.pin.updateFramePlacements(self.model, self.data)
            J = np.zeros((4, 3, 18))
            for i, f in enumerate(self.foot_fids):
                Jf = self.pin.getFrameJacobian(self.model, self.data, f, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                d = self.data.oMf[f].rotation @ FOOT_OFFSET
                dx = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
                J[i] = (Jf[:3] - dx @ Jf[3:]) @ B                     # point p = o + R d: v_p = v_o + omega x (R d)
            return J
        self.d.qpos[:] = qpos_mj; mujoco.mj_forward(self.m, self.d)
        J = np.zeros((4, 3, 18)); jacp = np.zeros((3, self.m.nv))
        for i, g in enumerate(self._foot_gids):
            mujoco.mj_jacGeom(self.m, self.d, jacp, None, g); J[i] = jacp
        return J

    def inverse_dynamics(self, qpos_mj, qvel_mj, qacc_mj) -> np.ndarray:
        """tau = M dv + h (no passive, no contacts), MuJoCo convention."""
        if self.backend == "pin":
            q, v, B = self._pin_qv(qpos_mj, qvel_mj)
            a = B @ np.asarray(qacc_mj, dtype=float) + self._Bdot_v(qpos_mj, qvel_mj)
            return B.T @ self.pin.rnea(self.model, self.data, q, v, a)
        return self.mass_matrix(qpos_mj) @ np.asarray(qacc_mj, dtype=float) + self.bias(qpos_mj, qvel_mj)


def random_state(rng: np.random.Generator, vel_scale: float = 1.0):
    """Random floating-base state (MuJoCo convention) inside the joint ranges of the Go2."""
    lo = np.tile([-1.0472, -1.5708, -2.7227], 4); hi = np.tile([1.0472, 3.4907, -0.83776], 4)
    lo[7::3][2:] = -0.5236; hi[7::3][2:] = 4.5379            # RL/RR thigh (indices 7,10)
    qj = rng.uniform(lo, hi); quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    qpos = np.concatenate([rng.normal(0, 0.3, 3) + [0, 0, 0.4], quat, qj])
    qvel = rng.normal(0, vel_scale, 18)
    return qpos, qvel
