"""Contact-aided right-invariant EKF (Hartley, Ghaffari, Eustice, Grizzle, IJRR 2020), pure NumPy, bias-free variant.

State X = [R v p d_1 ... d_N] in SE_{2+N}(3) (world-frame velocity/position/contact-foot positions), IMU propagation
    R+ = R Exp(w dt), v+ = v + (R a + g) dt, p+ = p + v dt + 1/2 (R a + g) dt^2, d_i+ = d_i,
right-invariant error eta = X_hat X^{-1} with log-linear dynamics d/dt xi = A xi + Ad_X_hat w,
    A = [[0,0,0,0],[g_x,0,0,0],[0,I,0,0],[0,0,0,0]]  (blocks R, v, p, d),
    Phi = Exp(A dt) = I + A dt + A^2 dt^2/2 (A^3 = 0),  P+ = Phi P Phi^T + Phi Qbar Phi^T dt,
    Qbar = Ad_X_hat Cov(w) Ad_X_hat^T,  w = (w_g, w_a, 0, w_d1, ...).
Kinematic measurement of contact foot i (right-invariant observation Y = X^{-1} b + V, b = e_p - e_{d_i}):
    y = h(q) = R^T (d_i - p) + n,   innovation z = (X_hat Y - b)[:3] = R_hat h(q) + p_hat - d_hat_i,
    H = [0 0 -I ... I(i) ...],  Nbar = R_hat Cov(n) R_hat^T,  S = H P H^T + Nbar,  K = P H^T S^{-1},
    X_hat+ = Exp(K z) X_hat,  P+ = (I-KH) P (I-KH)^T + K Nbar K^T.
Contact add: d_i = p_hat + R_hat h(q) with covariance from P (augmentation Jacobian); contact removal drops the block.
Outputs per correction: innovation z (world frame), S, NIS = z^T S^{-1} z, per-foot Mahalanobis (= sqrt NIS).
"""
from __future__ import annotations

import numpy as np

from .liegroups import adjoint_sek3, exp_sek3, exp_so3, skew

G_VEC = np.array([0.0, 0.0, -9.81])


class RIEKF:
    def __init__(self, R0, v0, p0, P0_diag=(1e-4, 1e-4, 1e-4), sigma_gyro=0.01, sigma_accel=0.1, sigma_contact=1e-3,
                 sigma_kin_floor=1e-3):
        self.R = np.array(R0, dtype=float); self.v = np.array(v0, dtype=float); self.p = np.array(p0, dtype=float)
        self.d = {}                                   # foot id -> world position (contact frames)
        self.P = np.diag(np.repeat(np.asarray(P0_diag, dtype=float), 3))   # 9x9 for (R,v,p)
        self.sg, self.sa, self.sd, self.skf = sigma_gyro, sigma_accel, sigma_contact, sigma_kin_floor
        self.log = []                                # per-correction records

    # ---- bookkeeping ------------------------------------------------------------------------------
    @property
    def feet(self):
        return list(self.d.keys())

    def _dim(self):
        return 9 + 3 * len(self.d)

    def _X(self):
        n = 2 + len(self.d); m = 3 + n
        X = np.eye(m); X[:3, :3] = self.R; X[:3, 3] = self.v; X[:3, 4] = self.p
        for k, f in enumerate(self.feet):
            X[:3, 5 + k] = self.d[f]
        return X

    def _set_from_X(self, X):
        self.R = X[:3, :3].copy(); self.v = X[:3, 3].copy(); self.p = X[:3, 4].copy()
        for k, f in enumerate(self.feet):
            self.d[f] = X[:3, 5 + k].copy()

    # ---- state perturbation (experiments: inconsistent re-initialisation) ---------------------------
    def kick(self, dtheta_world, dv, sig_theta, sig_v):
        """Perturb the estimate by a WORLD-frame rotation Exp(dtheta) R and a velocity offset dv, leaving p and the
        foot states untouched, and inflate P by the covariance of the induced right-invariant error: a rotation kick
        delta maps to xi = (delta, [v]x delta, [p]x delta, [d_i]x delta) (X_hat' = Exp(delta-block) X_hat), a velocity
        kick to xi_v = dv."""
        n = self._dim(); J = np.zeros((n, 3)); J[0:3] = np.eye(3); J[3:6] = skew(self.v); J[6:9] = skew(self.p)
        for k, f in enumerate(self.feet):
            J[9 + 3 * k:12 + 3 * k] = skew(self.d[f])
        self.P = self.P + sig_theta**2 * (J @ J.T); self.P[3:6, 3:6] += sig_v**2 * np.eye(3)
        self.R = exp_so3(np.asarray(dtheta_world, dtype=float)) @ self.R; self.v = self.v + np.asarray(dv, dtype=float)

    # ---- propagation ------------------------------------------------------------------------------
    def propagate(self, gyro, accel, dt):
        w = np.asarray(gyro, dtype=float); a = np.asarray(accel, dtype=float)
        R, v, p = self.R, self.v, self.p
        acc_w = R @ a + G_VEC
        R_new = R @ exp_so3(w * dt)
        v_new = v + acc_w * dt
        p_new = p + v * dt + 0.5 * acc_w * dt * dt
        n = self._dim(); nd = len(self.d)
        A = np.zeros((n, n)); A[3:6, 0:3] = skew(G_VEC); A[6:9, 3:6] = np.eye(3)
        Phi = np.eye(n) + A * dt + 0.5 * A @ A * dt * dt
        Qw = np.zeros((n, n)); Qw[0:3, 0:3] = self.sg**2 * np.eye(3); Qw[3:6, 3:6] = self.sa**2 * np.eye(3)
        for k in range(nd):
            Qw[9 + 3 * k:12 + 3 * k, 9 + 3 * k:12 + 3 * k] = self.sd**2 * np.eye(3)
        Ad = adjoint_sek3(self._X(), 2 + nd)
        Qbar = Ad @ Qw @ Ad.T
        self.P = Phi @ self.P @ Phi.T + Phi @ Qbar @ Phi.T * dt
        self.R, self.v, self.p = R_new, v_new, p_new

    # ---- contacts ---------------------------------------------------------------------------------
    def add_contact(self, foot, h_body, cov_kin):
        if foot in self.d:
            return
        n = self._dim(); d_new = self.p + self.R @ h_body
        self.d[foot] = d_new
        # right-invariant coordinates: xi_d(new) = xi_p exactly (the R-part cancels because d - p = R h), noise R n
        F = np.zeros((n + 3, n)); F[:n, :n] = np.eye(n); F[n:, 6:9] = np.eye(3)
        G = np.zeros((n + 3, 3)); G[n:, :] = self.R
        self.P = F @ self.P @ F.T + G @ cov_kin @ G.T

    def remove_contact(self, foot):
        if foot not in self.d:
            return
        k = self.feet.index(foot); n = self._dim()
        keep = [i for i in range(n) if not (9 + 3 * k <= i < 12 + 3 * k)]
        self.P = self.P[np.ix_(keep, keep)]
        del self.d[foot]

    # ---- correction -------------------------------------------------------------------------------
    def correct(self, measurements, t=None):
        """measurements: list of (foot, h_body (3,), cov_kin (3x3)) for feet currently in contact."""
        meas = [(f, h, c) for f, h, c in measurements if f in self.d]
        if not meas:
            return None
        n = self._dim(); m = 3 * len(meas)
        H = np.zeros((m, n)); z = np.zeros(m); Nb = np.zeros((m, m))
        for j, (f, h, c) in enumerate(meas):
            k = self.feet.index(f)
            H[3 * j:3 * j + 3, 6:9] = -np.eye(3); H[3 * j:3 * j + 3, 9 + 3 * k:12 + 3 * k] = np.eye(3)
            z[3 * j:3 * j + 3] = self.R @ h + self.p - self.d[f]
            Nb[3 * j:3 * j + 3, 3 * j:3 * j + 3] = self.R @ (c + self.skf**2 * np.eye(3)) @ self.R.T
        S = H @ self.P @ H.T + Nb
        Sinv = np.linalg.inv(S)
        K = self.P @ H.T @ Sinv
        delta = K @ z
        X = exp_sek3(delta, 2 + len(self.d)) @ self._X()
        self._set_from_X(X)
        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Nb @ K.T
        nis = float(z @ Sinv @ z)
        per_foot = [float(z[3 * j:3 * j + 3] @ np.linalg.inv(S[3 * j:3 * j + 3, 3 * j:3 * j + 3]) @ z[3 * j:3 * j + 3]) for j in range(len(meas))]
        rec = {"t": t, "feet": [f for f, _, _ in meas], "z": z.copy(), "S": S.copy(), "nis": nis, "nis_per_foot": per_foot, "dof": m}
        self.log.append(rec)
        return rec
