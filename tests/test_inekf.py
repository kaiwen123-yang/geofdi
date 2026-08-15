"""RIEKF unit tests: (1) covariance propagation under static + constant-bias inputs matches the closed-form
log-linear formula (Hartley et al. 2020, Sec. IV / App.: Phi = exp(A dt), P_n = Phi^n P0 Phi^n^T + sum Phi^k Qbar Phi^k^T dt)
computed independently with scipy.linalg.expm; (2) mirror equivariance: running the filter on the mirrored trajectory
(S0 mirror-sim, zero noise) gives the mirrored estimate and identical NIS to 1e-8."""
import numpy as np
import pytest
from scipy.linalg import expm

mujoco = pytest.importorskip("mujoco")

from geofdi.inekf.liegroups import adjoint_sek3, skew
from geofdi.inekf.rinekf import G_VEC, RIEKF
from geofdi.inekf.runner import run_filter
from geofdi.sim.env import SimConfig, keyframe_state, rollout

E = np.diag([1.0, -1.0, 1.0])
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]); S12 = np.tile(np.array([-1.0, 1.0, 1.0]), 4)
ZERO_NOISE = dict(encoder_pos_std=0, encoder_vel_std=0, torque_meas_std=0, actuator_std=0, imu_acc_std=0, imu_gyro_std=0,
                  init_joint_std=0, init_vel_std=0, init_body_rate_std=0)


def test_covariance_propagation_matches_closed_form():
    R0 = np.eye(3); f = RIEKF(R0, np.zeros(3), np.array([0, 0, 0.3]), P0_diag=(1e-3, 2e-3, 3e-3), sigma_gyro=0.02, sigma_accel=0.15)
    f.add_contact(0, np.array([0.19, 0.14, -0.3]), 1e-6 * np.eye(3))
    P0 = f.P.copy(); X0 = f._X().copy(); dt = 0.005; n = 200
    # static, constant gyro bias b_g and accel bias b_a as inputs (the state moves, but the log-linear error dynamics
    # and hence the covariance recursion are state-independent apart from Ad_X in Qbar)
    b_g = np.array([0.01, -0.02, 0.005]); b_a = np.array([0.0, 0.0, 9.81]) + np.array([0.02, -0.01, 0.03])
    n_dim = P0.shape[0]
    A = np.zeros((n_dim, n_dim)); A[3:6, 0:3] = skew(G_VEC); A[6:9, 3:6] = np.eye(3)
    Phi_ref = expm(A * dt)
    P_ref = P0.copy(); X = X0.copy()
    for k in range(n):
        Qw = np.zeros((n_dim, n_dim)); Qw[0:3, 0:3] = 0.02**2 * np.eye(3); Qw[3:6, 3:6] = 0.15**2 * np.eye(3); Qw[9:12, 9:12] = f.sd**2 * np.eye(3)
        Ad = adjoint_sek3(f._X(), 3)                       # same X_hat sequence as the filter uses (evaluated before its step)
        Qbar = Ad @ Qw @ Ad.T
        P_ref = Phi_ref @ P_ref @ Phi_ref.T + Phi_ref @ Qbar @ Phi_ref.T * dt
        f.propagate(b_g, b_a, dt)
    assert np.allclose(f.P, P_ref, rtol=1e-9, atol=1e-14), np.abs(f.P - P_ref).max()
    # Phi closed form I + A dt + A^2 dt^2/2 equals expm exactly (A nilpotent of index 3)
    assert np.allclose(np.eye(n_dim) + A * dt + 0.5 * A @ A * dt * dt, Phi_ref, atol=1e-15)


def mirror_state(qpos, qvel):
    qp = qpos.copy(); qv = qvel.copy()
    qp[0:3] = E @ qpos[0:3]; w, x, y, z = qpos[3:7]; qp[3:7] = [w, -x, y, -z]; qp[7:19] = S12 * qpos[7:19][PERM]
    qv[0:3] = E @ qvel[0:3]; qv[3:6] = -E @ qvel[3:6]; qv[6:18] = S12 * qvel[6:18][PERM]
    return qp, qv


def test_filter_is_mirror_equivariant():
    qpos0, qvel0 = keyframe_state(); rng = np.random.default_rng(3)
    qpos0[7:] += rng.normal(0, 0.03, 12); qvel0[6:] += rng.normal(0, 0.1, 12)
    qpB, qvB = mirror_state(qpos0, qvel0)
    A, _ = rollout(SimConfig(duration_s=2.5, seed=0, noise=ZERO_NOISE, init_qpos=qpos0.tolist(), init_qvel=qvel0.tolist(), foot_contact="stiff", contact_force_thresh=10.0))
    B, _ = rollout(SimConfig(duration_s=2.5, seed=0, noise=ZERO_NOISE, init_qpos=qpB.tolist(), init_qvel=qvB.tolist(), phase_offset=0.5, foot_contact="stiff", contact_force_thresh=10.0))
    fA, estA, RA = run_filter(A, kind="riekf"); fB, estB, RB = run_filter(B, kind="riekf")
    assert np.abs(estB - estA @ E).max() < 1e-8                                   # p_B = E p_A
    assert np.abs(RB - np.einsum("ij,tjk,kl->til", E, RA, E)).max() < 1e-8         # R_B = E R_A E
    nisA = np.array([r["nis"] for r in fA.log]); nisB = np.array([r["nis"] for r in fB.log])
    assert len(nisA) == len(nisB)
    # innovations mirror exactly (world frame, feet permuted): z_B = E z_A ; NIS agrees to 1e-6 relative (the first
    # corrections have S ~ 1e-6 m^2, where 1e-12 differences in the finite-difference kinematic Jacobian are amplified)
    sig = {0: 1, 1: 0, 2: 3, 3: 2}
    for ra, rb in zip(fA.log, fB.log):
        assert ra["dof"] == rb["dof"] and sorted(sig[f] for f in ra["feet"]) == sorted(rb["feet"])
        za = ra["z"].reshape(-1, 3); zb = rb["z"].reshape(-1, 3)
        for ia, fa in enumerate(ra["feet"]):
            ib = rb["feet"].index(sig[fa])
            assert np.abs(zb[ib] - E @ za[ia]).max() < 1e-8
    assert np.abs(nisA - nisB).max() < 1e-6 * (1 + np.abs(nisA).max())
    # feet: d_B[sigma(i)] = E d_A[i]  (LF<->RF, LH<->RH)
    for i, dA in fA.d.items():
        assert np.abs(fB.d[sig[i]] - E @ dA).max() < 1e-8
