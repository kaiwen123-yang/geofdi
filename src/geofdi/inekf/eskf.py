"""Baseline: standard error-state (quaternion) EKF with the same state, inputs, noise and measurements as the RIEKF.

Nominal state (R, v, p, d_i) with the classical additive/left-perturbation error (delta_theta in the BODY frame:
R = R_hat Exp(delta_theta); delta_v, delta_p, delta_d additive in the world frame). Propagation Jacobian
    F = I + dt * [[ -w_x, 0, 0, 0 ], [ -R a_x, 0, 0, 0 ], [ 0, I, 0, 0 ], [0,0,0,0] ]  (standard IMU ESKF),
noise Q = diag(sg^2, sa^2 (mapped through R), 0, sd^2). Measurement y = R^T (d_i - p) + n:
    dy = -R^T (d_i - p) x delta_theta ... : H = [ (R^T (d - p))_x , 0, -R^T, R^T ],  S = H P H^T + N (body frame).
Innovation is reported in the WORLD frame (R z_body) so that NIS values are directly comparable with the RIEKF's.
"""
from __future__ import annotations

import numpy as np

from .liegroups import exp_so3, skew

G_VEC = np.array([0.0, 0.0, -9.81])


class ESKF:
    def __init__(self, R0, v0, p0, P0_diag=(1e-4, 1e-4, 1e-4), sigma_gyro=0.01, sigma_accel=0.1, sigma_contact=1e-3,
                 sigma_kin_floor=1e-3):
        self.R = np.array(R0, dtype=float); self.v = np.array(v0, dtype=float); self.p = np.array(p0, dtype=float)
        self.d = {}
        self.P = np.diag(np.repeat(np.asarray(P0_diag, dtype=float), 3))
        self.sg, self.sa, self.sd, self.skf = sigma_gyro, sigma_accel, sigma_contact, sigma_kin_floor
        self.log = []

    @property
    def feet(self):
        return list(self.d.keys())

    def _dim(self):
        return 9 + 3 * len(self.d)

    def kick(self, dtheta_world, dv, sig_theta, sig_v):
        """Same physical perturbation as RIEKF.kick (world-frame rotation Exp(dtheta) R, velocity offset dv); in the
        body-frame attitude-error coordinates it is a pure attitude error delta_theta' = delta_theta - R'^T dtheta, so
        P gets sig_theta^2 I on the attitude block and sig_v^2 I on the velocity block."""
        self.P[0:3, 0:3] += sig_theta**2 * np.eye(3); self.P[3:6, 3:6] += sig_v**2 * np.eye(3)
        self.R = exp_so3(np.asarray(dtheta_world, dtype=float)) @ self.R; self.v = self.v + np.asarray(dv, dtype=float)

    def propagate(self, gyro, accel, dt):
        w = np.asarray(gyro, dtype=float); a = np.asarray(accel, dtype=float)
        R, v, p = self.R, self.v, self.p
        acc_w = R @ a + G_VEC
        n = self._dim(); nd = len(self.d)
        F = np.eye(n)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt
        F[3:6, 0:3] = -R @ skew(a) * dt
        F[6:9, 3:6] = np.eye(3) * dt
        Q = np.zeros((n, n)); Q[0:3, 0:3] = self.sg**2 * np.eye(3) * dt; Q[3:6, 3:6] = R @ (self.sa**2 * np.eye(3)) @ R.T * dt
        for k in range(nd):
            Q[9 + 3 * k:12 + 3 * k, 9 + 3 * k:12 + 3 * k] = self.sd**2 * np.eye(3) * dt
        self.P = F @ self.P @ F.T + Q
        self.R = R @ exp_so3(w * dt); self.v = v + acc_w * dt; self.p = p + v * dt + 0.5 * acc_w * dt * dt

    def add_contact(self, foot, h_body, cov_kin):
        if foot in self.d:
            return
        n = self._dim(); d_new = self.p + self.R @ h_body
        self.d[foot] = d_new
        F = np.zeros((n + 3, n)); F[:n, :n] = np.eye(n); F[n:, 6:9] = np.eye(3); F[n:, 0:3] = -self.R @ skew(h_body)
        G = np.zeros((n + 3, 3)); G[n:, :] = self.R
        self.P = F @ self.P @ F.T + G @ cov_kin @ G.T

    def remove_contact(self, foot):
        if foot not in self.d:
            return
        k = self.feet.index(foot); n = self._dim()
        keep = [i for i in range(n) if not (9 + 3 * k <= i < 12 + 3 * k)]
        self.P = self.P[np.ix_(keep, keep)]; del self.d[foot]

    def correct(self, measurements, t=None):
        meas = [(f, h, c) for f, h, c in measurements if f in self.d]
        if not meas:
            return None
        n = self._dim(); m = 3 * len(meas); R = self.R
        H = np.zeros((m, n)); zb = np.zeros(m); N = np.zeros((m, m))
        for j, (f, h, c) in enumerate(meas):
            k = self.feet.index(f); r = R.T @ (self.d[f] - self.p)
            H[3 * j:3 * j + 3, 0:3] = skew(r); H[3 * j:3 * j + 3, 6:9] = -R.T; H[3 * j:3 * j + 3, 9 + 3 * k:12 + 3 * k] = R.T
            zb[3 * j:3 * j + 3] = h - r
            N[3 * j:3 * j + 3, 3 * j:3 * j + 3] = c + self.skf**2 * np.eye(3)
        S = H @ self.P @ H.T + N; Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv; dx = K @ zb
        self.R = R @ exp_so3(dx[0:3]); self.v = self.v + dx[3:6]; self.p = self.p + dx[6:9]
        for k, f in enumerate(self.feet):
            self.d[f] = self.d[f] + dx[9 + 3 * k:12 + 3 * k]
        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ N @ K.T
        nis = float(zb @ Sinv @ zb)
        per_foot = [float(zb[3 * j:3 * j + 3] @ np.linalg.inv(S[3 * j:3 * j + 3, 3 * j:3 * j + 3]) @ zb[3 * j:3 * j + 3]) for j in range(len(meas))]
        # world-frame innovation for comparability
        zw = np.concatenate([R @ zb[3 * j:3 * j + 3] for j in range(len(meas))])
        rec = {"t": t, "feet": [f for f, _, _ in meas], "z": zw, "z_body": zb, "S": S.copy(), "nis": nis, "nis_per_foot": per_foot, "dof": m}
        self.log.append(rec)
        return rec
