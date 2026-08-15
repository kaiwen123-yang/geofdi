"""Rolling-contact right-invariant EKF for the wheeled M1 (Sprint 7 Block N2; see docs/decisions/n2_rolling_contact_memo.md).

The contact-aided RIEKF (geofdi.inekf.rinekf.RIEKF) models each stance foot as a FIXED world point (d_i = const). A
rolling wheel's ground-contact point is not fixed: it translates forward at the rolling rate. The correct model is
    d_dot_i = R u_i,   u_i = [r * omega_wheel_i, 0, 0]  in the wheel frame (forward; zero lateral/vertical: no slip),
with u_i a KNOWN input (measured wheel rate + leg kinematics). Because d_dot_i = X [u_i; 0] is a navigation-equation
term, the dynamics stay group-affine (Barrau-Bonnabel 2017), and the right-invariant error of the contact block has
    d/dt xi_{d_i} = 0    (A[d_i, R] = 0, verified numerically in scratchpad/roll_check.py),
UNLIKE the plain-EKF world-position Jacobian -R[u_i]x: the InEKF group-affine structure annihilates it because R u_i
has no world-constant part (contrast the velocity block's gravity term, which yields A[v,R]=[g]x). So the rolling
contact changes ONLY (a) the mean, d_i += R u_i dt, and (b) the d_i process noise, which becomes the wheel-frame
covariance of u_i: sigma_roll^2 forward (rolling-odometry uncertainty), sigma_slip^2 lateral/vertical (the no-slip /
no-penetration soft constraint). The propagation matrix A and the kinematic correction are inherited unchanged.
"""
from __future__ import annotations

import numpy as np

from .liegroups import adjoint_sek3, exp_so3, skew
from .rinekf import G_VEC, RIEKF


class RollingRIEKF(RIEKF):
    def __init__(self, R0, v0, p0, P0_diag=(1e-4, 1e-4, 1e-4), sigma_gyro=0.01, sigma_accel=0.1, sigma_contact=1e-3,
                 sigma_kin_floor=1e-3, sigma_roll=0.05, sigma_slip=2e-3):
        super().__init__(R0, v0, p0, P0_diag, sigma_gyro, sigma_accel, sigma_contact, sigma_kin_floor)
        self.sigma_roll = float(sigma_roll)      # forward-axis (rolling-odometry) process noise on a contact, m/s
        self.sigma_slip = float(sigma_slip)      # lateral/vertical (no-slip) process noise, m/s
        self.u = {}                              # foot -> body-frame contact velocity u_i (3,)
        self.wheel_frame = {}                    # foot -> 3x3 [forward, lateral, up] axes in the body frame

    def set_rolling_inputs(self, u_body: dict, wheel_frame: dict | None = None):
        """u_body: foot -> u_i (3,) body-frame contact velocity (= R^T d_dot_i); default wheel frame = body axes."""
        self.u = {f: np.asarray(u, float) for f, u in u_body.items()}
        self.wheel_frame = {f: np.asarray(W, float) for f, W in (wheel_frame or {}).items()}

    def _wheel_cov(self, foot):
        """Body-frame process covariance of contact foot: anisotropic in the wheel frame (roll forward, slip lateral/up)."""
        W = self.wheel_frame.get(foot, np.eye(3))
        D = np.diag([self.sigma_roll**2, self.sigma_slip**2, self.sigma_slip**2])
        return W @ D @ W.T

    def propagate(self, gyro, accel, dt):
        w = np.asarray(gyro, float); a = np.asarray(accel, float)
        R, v, p = self.R, self.v, self.p
        acc_w = R @ a + G_VEC
        R_new = R @ exp_so3(w * dt); v_new = v + acc_w * dt; p_new = p + v * dt + 0.5 * acc_w * dt * dt
        # moving contacts: mean advances by R u_i dt (fixed feet: u=0 -> unchanged)
        d_new = {}
        for f in self.feet:
            ui = self.u.get(f, np.zeros(3))
            d_new[f] = self.d[f] + R @ ui * dt
        n = self._dim(); nd = len(self.d)
        A = np.zeros((n, n)); A[3:6, 0:3] = skew(G_VEC); A[6:9, 3:6] = np.eye(3)   # SAME A as fixed-foot (A[d,*]=0)
        Phi = np.eye(n) + A * dt + 0.5 * A @ A * dt * dt
        Qw = np.zeros((n, n)); Qw[0:3, 0:3] = self.sg**2 * np.eye(3); Qw[3:6, 3:6] = self.sa**2 * np.eye(3)
        for k, f in enumerate(self.feet):
            # rolling contacts carry the wheel-frame odometry/no-slip covariance; a truly fixed foot (u=0, no wheel
            # frame) falls back to the isotropic sigma_contact of the base filter
            Qw[9 + 3 * k:12 + 3 * k, 9 + 3 * k:12 + 3 * k] = self._wheel_cov(f) if f in self.u else self.sd**2 * np.eye(3)
        Ad = adjoint_sek3(self._X(), 2 + nd)
        Qbar = Ad @ Qw @ Ad.T
        self.P = Phi @ self.P @ Phi.T + Phi @ Qbar @ Phi.T * dt
        self.R, self.v, self.p = R_new, v_new, p_new
        for f in self.feet:
            self.d[f] = d_new[f]


def wheel_contact_inputs(dq_wheel: dict, radius: float, forward_axis=(1.0, 0.0, 0.0)):
    """Build set_rolling_inputs arguments for straight rolling: u_i = r * omega_wheel_i along the wheel forward axis
    (body frame), no lateral/vertical. dq_wheel: foot -> wheel angular rate (rad/s). Returns (u_body, wheel_frame)."""
    fwd = np.asarray(forward_axis, float); fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    up = np.array([0.0, 0.0, 1.0]); lat = np.cross(up, fwd); lat /= (np.linalg.norm(lat) + 1e-12); up = np.cross(fwd, lat)
    W = np.column_stack([fwd, lat, up])
    u_body = {f: float(radius) * float(om) * fwd for f, om in dq_wheel.items()}
    wheel_frame = {f: W for f in dq_wheel}
    return u_body, wheel_frame
