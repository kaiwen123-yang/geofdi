"""Go2 leg forward kinematics in the IMU frame, computed from the robot MODEL only (a private MjData used purely as
a kinematics engine — no ground-truth state is read). h_i(q) = position of foot i relative to the IMU site, expressed
in the base frame; J_i(q) = d h_i / d q_leg (3x3) by finite differences (used for the measurement covariance)."""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.env import scene_path

LEG_MODEL = ("FL", "FR", "RL", "RR")            # == LF, RF, LH, RH


class Go2Kinematics:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(scene_path("flat"))
        self.d = mujoco.MjData(self.m)
        self.jadr = [[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{L}_{j}_joint")]
                      for j in ("hip", "thigh", "calf")] for L in LEG_MODEL]
        self.foot = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, L) for L in LEG_MODEL]
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.imu_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_imu = self.m.site_pos[self.imu_site].copy()          # IMU offset in the base frame

    def _fk_all(self, q12: np.ndarray) -> np.ndarray:
        """Foot positions (4,3) in the base frame relative to the IMU site, for joint vector q12 (LF,RF,LH,RH x 3)."""
        d = self.d
        d.qpos[:] = 0.0; d.qpos[3] = 1.0                       # base at origin, identity orientation
        for li in range(4):
            for j in range(3):
                d.qpos[self.jadr[li][j]] = q12[3 * li + j]
        mujoco.mj_kinematics(self.m, d)
        return np.array([d.geom_xpos[g] for g in self.foot]) - self.r_imu

    def h(self, q12: np.ndarray, leg: int) -> np.ndarray:
        return self._fk_all(q12)[leg]

    def h_and_jac(self, q12: np.ndarray, leg: int, eps: float = 1e-6):
        h0 = self._fk_all(q12)[leg]
        J = np.zeros((3, 3))
        for j in range(3):
            qp = q12.copy(); qp[3 * leg + j] += eps
            J[:, j] = (self._fk_all(qp)[leg] - h0) / eps
        return h0, J
