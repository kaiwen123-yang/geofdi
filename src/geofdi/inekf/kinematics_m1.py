"""Wheeled-M1 leg forward kinematics in the IMU frame, from the robot MODEL only (a private MjData used purely as a
kinematics engine — no ground-truth state is read), analogous to geofdi.inekf.kinematics.Go2Kinematics.

The tracked "foot" for a rolling wheel is the GROUND-CONTACT point = wheel-geom centre dropped by the wheel radius,
h_i(q) = (wheel-centre position relative to the IMU site, base frame) - [0, 0, r]. Only the three leg joints
(ABAD, HIP, KNEE) move the wheel centre; the wheel spin (FOOT) joint does not, so J_i = d h_i / d q_leg is 3x3 over
the leg joints (used for the measurement covariance)."""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.env_m1 import scene_path_m1
from ..sim.telemetry_m1 import JOINTS, LEGS, MJCF_LEG, WHEEL_GEOMS, WHEEL_R

LEG_JOINTS = ("ABAD", "HIP", "KNEE")                     # the 3 positioning joints (WHEEL/FOOT spin excluded)


class M1Kinematics:
    def __init__(self, model: str = "m1_wheeled_sym", wheel_radius: float = WHEEL_R):
        self.m = mujoco.MjModel.from_xml_path(scene_path_m1(model))
        self.d = mujoco.MjData(self.m)
        self.r = float(wheel_radius)
        # qpos address of each (leg, ABAD/HIP/KNEE) joint
        self.jadr = [[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{MJCF_LEG[L]}_{j}_JOINT")]
                      for j in LEG_JOINTS] for L in LEGS]
        # GeoFDI q16 index of each (leg, ABAD/HIP/KNEE) = leg*4 + JOINTS.index(j)
        self.q16 = [[4 * li + JOINTS.index(j) for j in LEG_JOINTS] for li in range(4)]
        self.wheel = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, WHEEL_GEOMS[L]) for L in LEGS]
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.imu_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.r_imu = self.m.site_pos[self.imu_site].copy()          # IMU offset in the base frame

    def _fk_all(self, q16: np.ndarray) -> np.ndarray:
        """Ground-contact positions (4,3) in the base frame relative to the IMU site, for GeoFDI-order q16."""
        d = self.d
        d.qpos[:] = 0.0; d.qpos[3] = 1.0                       # base at origin, identity orientation
        for li in range(4):
            for jj, j in enumerate(LEG_JOINTS):
                d.qpos[self.jadr[li][jj]] = q16[self.q16[li][jj]]
        mujoco.mj_kinematics(self.m, d)
        centres = np.array([d.geom_xpos[g] for g in self.wheel]) - self.r_imu
        centres[:, 2] -= self.r                                 # drop wheel centre to the ground-contact point
        return centres

    def h(self, q16: np.ndarray, leg: int) -> np.ndarray:
        return self._fk_all(q16)[leg]

    def h_and_jac(self, q16: np.ndarray, leg: int, eps: float = 1e-6):
        h0 = self._fk_all(q16)[leg]
        J = np.zeros((3, 3))
        for jj in range(3):
            qp = q16.copy(); qp[self.q16[leg][jj]] += eps
            J[:, jj] = (self._fk_all(qp)[leg] - h0) / eps
        return h0, J
