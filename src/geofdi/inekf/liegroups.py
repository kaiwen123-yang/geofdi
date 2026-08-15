"""SO(3) / SE_{2+N}(3) helpers for the invariant filter (Hartley et al., IJRR 2020 conventions)."""
from __future__ import annotations

import numpy as np


def skew(w):
    w = np.asarray(w, dtype=float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(phi):
    phi = np.asarray(phi, dtype=float); a = np.linalg.norm(phi)
    K = skew(phi)
    if a < 1e-10:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a**2 * K @ K


def log_so3(R):
    R = np.asarray(R, dtype=float)
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0); a = np.arccos(c)
    if a < 1e-10:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def left_jacobian_so3(phi):
    phi = np.asarray(phi, dtype=float); a = np.linalg.norm(phi); K = skew(phi)
    if a < 1e-8:
        return np.eye(3) + 0.5 * K + K @ K / 6.0
    return np.eye(3) + (1 - np.cos(a)) / a**2 * K + (a - np.sin(a)) / a**3 * K @ K


def exp_sek3(xi, n_extra: int):
    """Exp on SE_{n_extra}(3) for xi = (phi, x_1, ..., x_{n_extra}) (3(1+n_extra),) -> (3+n_extra)x(3+n_extra) matrix."""
    xi = np.asarray(xi, dtype=float); phi = xi[:3]
    R = exp_so3(phi); J = left_jacobian_so3(phi)
    m = 3 + n_extra
    X = np.eye(m); X[:3, :3] = R
    for k in range(n_extra):
        X[:3, 3 + k] = J @ xi[3 + 3 * k:6 + 3 * k]
    return X


def adjoint_sek3(X, n_extra: int):
    """Ad_X for X in SE_{n_extra}(3): block matrix acting on xi = (phi, x_1..x_n)."""
    R = X[:3, :3]; m = 3 * (1 + n_extra)
    A = np.zeros((m, m)); A[:3, :3] = R
    for k in range(n_extra):
        A[3 + 3 * k:6 + 3 * k, :3] = skew(X[:3, 3 + k]) @ R
        A[3 + 3 * k:6 + 3 * k, 3 + 3 * k:6 + 3 * k] = R
    return A


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_rpy(R):
    """roll, pitch, yaw (ZYX) from a rotation matrix."""
    return (np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0]))
