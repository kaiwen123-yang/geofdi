"""t01 — representation & equivariance tests (run in CI via `make test`).

(1) Mirror-sim exactness: simulate trajectory A from a (deliberately asymmetric) initial state with the
    equivariant trot controller, and trajectory B from the MIRRORED initial state with the controller clock
    shifted by half a period. In an exactly mirror-symmetric world with an exactly Σ-equivariant controller,
    B must equal rho(g_s) A row by row (q, dq, tau_cmd, tau_meas, IMU a/w, contacts, temps) — we assert
    element-wise agreement to 1e-10 (noise off). This exercises the vendored model's symmetrization, the
    controller construction and every sign in the channel manifest (gyro axial, accelerometer polar, HAA -1).
(2) Negative control: with the gyro deliberately treated as a polar vector (+E instead of -E) the same
    comparison must FAIL on the gyro channels by a wide margin (>0.05 rad/s), i.e. the manifest signs are
    load-bearing, not vacuous.
(3) Algebra: rho(g_s)^2 = I; the controller reference satisfies RF(theta) = S LF(theta + 1/2) exactly.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from geofdi.groups.c2 import C2Rep
from geofdi.sim.controller import S_MIRROR, TrotController, TrotParams
from geofdi.sim.env import SimConfig, keyframe_state, rollout
from geofdi.sim.telemetry import z_channel_names

E = np.diag([1.0, -1.0, 1.0])
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])       # LF<->RF, LH<->RH in model joint order
S12 = np.tile(S_MIRROR, 4)
ZERO_NOISE = {"encoder_pos_std": 0, "encoder_vel_std": 0, "torque_meas_std": 0, "actuator_std": 0, "imu_acc_std": 0,
                  "imu_gyro_std": 0, "init_joint_std": 0, "init_vel_std": 0, "init_body_rate_std": 0}


def mirror_state(qpos, qvel):
    qp = qpos.copy(); qv = qvel.copy()
    qp[0:3] = E @ qpos[0:3]
    w, x, y, z = qpos[3:7]; qp[3:7] = [w, -x, y, -z]          # quaternion of E R E
    qp[7:19] = S12 * qpos[7:19][PERM]
    qv[0:3] = E @ qvel[0:3]; qv[3:6] = -E @ qvel[3:6]         # linear polar, angular axial
    qv[6:18] = S12 * qvel[6:18][PERM]
    return qp, qv


def _pair_of_rollouts(duration=3.0):
    qpos0, qvel0 = keyframe_state()
    rng = np.random.default_rng(123)
    qpos0[7:] += rng.normal(0, 0.05, 12); qvel0[6:] += rng.normal(0, 0.2, 12); qvel0[3:6] += rng.normal(0, 0.1, 3)
    qpB, qvB = mirror_state(qpos0, qvel0)
    A, man = rollout(SimConfig(duration_s=duration, seed=0, noise=ZERO_NOISE, init_qpos=qpos0.tolist(), init_qvel=qvel0.tolist()))
    B, _ = rollout(SimConfig(duration_s=duration, seed=0, noise=ZERO_NOISE, init_qpos=qpB.tolist(), init_qvel=qvB.tolist(),
                             phase_offset=0.5))
    return A, B, man


def test_mirror_sim_equals_rho_of_original_to_1e10():
    A, B, man = _pair_of_rollouts()
    rep = C2Rep(man)
    chans = z_channel_names(man)
    ZA = A[chans].to_numpy(); ZB = B[chans].to_numpy()
    rhoA = rep.mirror_only(ZA)                                # (T, d) rows, mirror only (clock already shifted)
    err = np.abs(ZB - rhoA)
    worst = {c: float(err[:, i].max()) for i, c in enumerate(chans)}
    assert err.max() <= 1e-10, f"mirror-sim mismatch {err.max():.3e}; worst channels: {sorted(worst.items(), key=lambda kv: -kv[1])[:5]}"
    assert np.allclose(np.mod(B['theta'].to_numpy() - A['theta'].to_numpy(), 1.0), 0.5, atol=1e-12)


def test_gyro_sign_negative_control():
    A, B, man = _pair_of_rollouts(duration=2.0)
    bad = C2Rep(man, gyro_sign_bug=True)
    chans = z_channel_names(man)
    ZA = A[chans].to_numpy(); ZB = B[chans].to_numpy()
    err = np.abs(ZB - bad.mirror_only(ZA))
    gyro = [chans.index(f"imu_w_{a}") for a in "xyz"]
    others = [i for i in range(len(chans)) if i not in gyro]
    assert err[:, gyro].max() > 0.05, "the wrong gyro sign was not detected — invariance test is vacuous"
    assert err[:, others].max() <= 1e-10


def test_rho_is_involution_and_reference_equivariant():
    _, _, man = _pair_of_rollouts(duration=0.05)
    rep = C2Rep(man)
    assert np.allclose(rep.P @ rep.P, np.eye(rep.d))
    ctrl = TrotController(TrotParams(speed=0.3))
    for th in np.linspace(0, 1, 13):
        q, _ = ctrl.reference(th); q2, _ = ctrl.reference(th + 0.5)
        assert np.allclose(q[3:6], S_MIRROR * q2[0:3], atol=1e-14)    # RF(theta) = S LF(theta+1/2)
        assert np.allclose(q[9:12], S_MIRROR * q2[6:9], atol=1e-14)   # RH(theta) = S LH(theta+1/2)
        assert np.allclose(q[6:9], q2[0:3], atol=1e-14)               # LH(theta) = LF(theta+1/2)  (trot)
