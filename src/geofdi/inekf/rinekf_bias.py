"""Bias-augmented contact-aided InEKF (Sprint 7 Block N2): the imperfect-InEKF form (Hartley et al. 2020, §VII) with
IMU gyro/accel bias, optionally plus a per-joint encoder-bias random walk.

IMU biases break the group-affine property (the bias enters the propagation of R, v through the corrected input
ω − b_g, a − b_a, which is not a linear function of the state): the filter is therefore the "imperfect InEKF" — the
InEKF error for the (R, v, p, d) block, a standard additive error for the bias block, and the cross terms in the
propagation Jacobian. This is the accepted contact-InEKF-with-bias construction; the bias states are weakly observable
(only through the kinematic contacts), which is exactly what makes a bias fault show a slow adjoint signature.

State: xi = (xi_R, xi_v, xi_p, xi_{d_i}, delta_bg, delta_ba[, delta_benc]) with the RI error on the SE_{2+N}(3) part and
additive errors on the biases. Propagation uses the bias-corrected input; the Jacobian gains A[R, bg] = -R (through
Exp), A[v, ba] = -R, A[v, bg] = -R [a]_x (second order, kept), and the bias blocks are random walks. Encoder bias enters
only the measurement: h(q - b_enc); the RIEKF update b_enc += (K z) uses H_benc = +R J_leg (see correct()).
"""
from __future__ import annotations

import numpy as np

from .liegroups import adjoint_sek3, exp_sek3, exp_so3, skew
from .rinekf import G_VEC, RIEKF


class RIEKFBias(RIEKF):
    def __init__(self, R0, v0, p0, P0_diag=(1e-4, 1e-4, 1e-4), sigma_gyro=0.01, sigma_accel=0.1, sigma_contact=1e-3,
                 sigma_kin_floor=1e-3, sigma_bg_rw=1e-4, sigma_ba_rw=1e-3, P0_bias=(1e-4, 1e-4), n_enc=0,
                 sigma_benc_rw=1e-5, P0_benc=1e-4, bg0=None, ba0=None):
        super().__init__(R0, v0, p0, P0_diag, sigma_gyro, sigma_accel, sigma_contact, sigma_kin_floor)
        self.bg = np.zeros(3) if bg0 is None else np.array(bg0, float); self.ba = np.zeros(3) if ba0 is None else np.array(ba0, float)
        self.n_enc = int(n_enc); self.benc = np.zeros(self.n_enc)
        self.sbg, self.sba, self.sbenc = sigma_bg_rw, sigma_ba_rw, sigma_benc_rw
        nb = 6 + self.n_enc
        Pb = np.concatenate([np.repeat(P0_bias, 3), np.full(self.n_enc, P0_benc)])
        self.P = np.block([[self.P, np.zeros((9, nb))], [np.zeros((nb, 9)), np.diag(Pb)]])

    def _nb(self):
        return 6 + self.n_enc

    def _core_dim(self):                                   # SE_{2+N}(3) part = 9 + 3*n_contacts
        return 9 + 3 * len(self.d)

    def _dim(self):
        return self._core_dim() + self._nb()

    # bias indices in the full covariance
    def _bg_idx(self):
        c = self._core_dim(); return slice(c, c + 3)

    def _ba_idx(self):
        c = self._core_dim(); return slice(c + 3, c + 6)

    def _benc_idx(self):
        c = self._core_dim(); return slice(c + 6, c + 6 + self.n_enc)

    def propagate(self, gyro, accel, dt):
        w = np.asarray(gyro, float) - self.bg; a = np.asarray(accel, float) - self.ba
        R, v, p = self.R, self.v, self.p; acc_w = R @ a + G_VEC
        R_new = R @ exp_so3(w * dt); v_new = v + acc_w * dt; p_new = p + v * dt + 0.5 * acc_w * dt * dt
        cdim = self._core_dim(); nd = len(self.d); n = self._dim()
        A = np.zeros((n, n)); A[3:6, 0:3] = skew(G_VEC); A[6:9, 3:6] = np.eye(3)
        # bias couplings (imperfect InEKF): attitude error driven by gyro bias, velocity error by accel + gyro bias
        A[0:3, self._bg_idx()] = -R
        A[3:6, self._ba_idx()] = -R
        A[3:6, self._bg_idx()] = -R @ skew(a)
        Phi = np.eye(n) + A * dt + 0.5 * A @ A * dt * dt
        Qw = np.zeros((n, n)); Qw[0:3, 0:3] = self.sg**2 * np.eye(3); Qw[3:6, 3:6] = self.sa**2 * np.eye(3)
        for k in range(nd):
            Qw[9 + 3 * k:12 + 3 * k, 9 + 3 * k:12 + 3 * k] = self.sd**2 * np.eye(3)
        Qw[self._bg_idx(), self._bg_idx()] = self.sbg**2 * np.eye(3); Qw[self._ba_idx(), self._ba_idx()] = self.sba**2 * np.eye(3)
        if self.n_enc:
            Qw[self._benc_idx(), self._benc_idx()] = self.sbenc**2 * np.eye(self.n_enc)
        Ad = np.eye(n); Ad[:cdim, :cdim] = adjoint_sek3(self._X(), 2 + nd)
        Qbar = Ad @ Qw @ Ad.T
        self.P = Phi @ self.P @ Phi.T + Phi @ Qbar @ Phi.T * dt
        self.R, self.v, self.p = R_new, v_new, p_new

    # contacts: the augmentation only touches the core block; pad the Jacobians to the full dim
    def add_contact(self, foot, h_body, cov_kin):
        if foot in self.d:
            return
        cdim = self._core_dim(); nb = self._nb(); n = self._dim()
        d_new = self.p + self.R @ h_body; self.d[foot] = d_new; cdim2 = self._core_dim()
        F = np.zeros((n + 3, n)); F[:cdim, :cdim] = np.eye(cdim); F[cdim:cdim + 3, 6:9] = np.eye(3); F[cdim + 3:, cdim:] = np.eye(nb)
        G = np.zeros((n + 3, 3)); G[cdim:cdim + 3, :] = self.R
        self.P = F @ self.P @ F.T + G @ cov_kin @ G.T

    def remove_contact(self, foot):
        if foot not in self.d:
            return
        k = self.feet.index(foot); n = self._dim()
        keep = [i for i in range(n) if not (9 + 3 * k <= i < 12 + 3 * k)]
        self.P = self.P[np.ix_(keep, keep)]; del self.d[foot]

    def correct(self, measurements, t=None, leg_jac=None):
        """leg_jac: optional dict foot -> (J_leg (3, n_enc_leg), enc_index (n_enc_leg,)) to include the encoder-bias
        measurement Jacobian H_benc = +R J_leg (from h(q - b_enc) at first order). Without it the encoder bias is
        unobservable (its marginal covariance only grows)."""
        meas = [(f, h, c) for f, h, c in measurements if f in self.d]
        if not meas:
            return None
        n = self._dim(); m = 3 * len(meas); H = np.zeros((m, n)); z = np.zeros(m); Nb = np.zeros((m, m))
        for j, (f, h, c) in enumerate(meas):
            k = self.feet.index(f)
            H[3 * j:3 * j + 3, 6:9] = -np.eye(3); H[3 * j:3 * j + 3, 9 + 3 * k:12 + 3 * k] = np.eye(3)
            if leg_jac is not None and f in leg_jac:
                # h(q - b_enc) => dz/d(delta b_enc) = -R J, and the RIEKF update b_enc += (K z) needs H = -dz/d(err),
                # so H_benc = +R J (verified by encoder-bias recovery: -R J diverges to the wrong sign)
                J, ei = leg_jac[f]; H[3 * j:3 * j + 3, self._benc_idx().start + np.asarray(ei)] = self.R @ J
            z[3 * j:3 * j + 3] = self.R @ h + self.p - self.d[f]
            Nb[3 * j:3 * j + 3, 3 * j:3 * j + 3] = self.R @ (c + self.skf**2 * np.eye(3)) @ self.R.T
        S = H @ self.P @ H.T + Nb; Sinv = np.linalg.inv(S); K = self.P @ H.T @ Sinv; delta = K @ z
        core = self._core_dim()
        X = exp_sek3(delta[:core], 2 + len(self.d)) @ self._X(); self._set_from_X(X)
        self.bg = self.bg + delta[self._bg_idx()]; self.ba = self.ba + delta[self._ba_idx()]
        if self.n_enc:
            self.benc = self.benc + delta[self._benc_idx()]
        IKH = np.eye(n) - K @ H; self.P = IKH @ self.P @ IKH.T + K @ Nb @ K.T
        nis = float(z @ Sinv @ z)
        per_foot = [float(z[3 * j:3 * j + 3] @ np.linalg.inv(S[3 * j:3 * j + 3, 3 * j:3 * j + 3]) @ z[3 * j:3 * j + 3]) for j in range(len(meas))]
        rec = {"t": t, "feet": [f for f, _, _ in meas], "z": z.copy(), "S": S.copy(), "nis": nis, "nis_per_foot": per_foot, "dof": m,
               "bg": self.bg.copy(), "ba": self.ba.copy(), "benc": self.benc.copy()}
        self.log.append(rec); return rec
