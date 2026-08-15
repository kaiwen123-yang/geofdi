"""t01 for the wheeled M1 worlds (Sprint 7 Block W1): mirror-sim exactness in rolling mode.

Rollout A from a deliberately asymmetric initial state under the equivariant rolling controller; rollout B from the
MIRRORED initial state (Σ = G in rolling mode: no phase offset). In the symmetrized world B must equal rho(g_s) A row by
row on every data-element channel (q, dq, tau_cmd, tau_meas, IMU a/w, contacts) to 1e-10 (noise off); the original
world's residual is recorded as the eps_dyn candidate (chiral details: base com_y, RAR knee mass, base products).
Also: the manifest's mirror matrix is an involution and the wheel-angle channels are excluded from Z.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from geofdi.groups.c2 import C2Rep                                                     # noqa: E402
from geofdi.sim.env_m1 import SimConfigM1, keyframe_state_m1, rollout_m1               # noqa: E402
from geofdi.sim.telemetry_m1 import MJCF_LEG_ORDER, z_channel_names                     # noqa: E402

E = np.diag([1.0, -1.0, 1.0])
# MJCF joint order FAR, FBL, RAR, RBL x (ABAD, HIP, KNEE, FOOT): mirror = swap the leg blocks, ABAD sign flips
PERM = np.concatenate([np.arange(4) + 4, np.arange(4), np.arange(4) + 12, np.arange(4) + 8])
SGN = np.tile([-1.0, 1.0, 1.0, 1.0], 4)
ZERO_NOISE = {"encoder_pos_std": 0, "encoder_vel_std": 0, "torque_meas_std": 0, "actuator_std": 0, "imu_acc_std": 0,
              "imu_gyro_std": 0, "init_joint_std": 0, "init_vel_std": 0, "init_body_rate_std": 0}


def mirror_state(qpos, qvel):
    qp = qpos.copy(); qv = qvel.copy()
    qp[0:3] = E @ qpos[0:3]; w, x, y, z = qpos[3:7]; qp[3:7] = [w, -x, y, -z]
    qp[7:23] = SGN * qpos[7:23][PERM]
    qv[0:3] = E @ qvel[0:3]; qv[3:6] = -E @ qvel[3:6]; qv[6:22] = SGN * qvel[6:22][PERM]
    return qp, qv


def _pair(model, duration=4.0, speed=1.0):
    qpos0, qvel0 = keyframe_state_m1(model)
    rng = np.random.default_rng(11)
    qpos0[7:23] += rng.normal(0, 0.05, 16) * np.tile([1, 1, 1, 0], 4); qvel0[6:22] += rng.normal(0, 0.2, 16); qvel0[3:6] += rng.normal(0, 0.1, 3); qvel0[0:3] += rng.normal(0, 0.1, 3)
    qpB, qvB = mirror_state(qpos0, qvel0)
    A, man = rollout_m1(SimConfigM1(model=model, speed=speed, duration_s=duration, seed=0, noise=ZERO_NOISE, init_qpos=qpos0.tolist(), init_qvel=qvel0.tolist()))
    B, _ = rollout_m1(SimConfigM1(model=model, speed=speed, duration_s=duration, seed=0, noise=ZERO_NOISE, init_qpos=qpB.tolist(), init_qvel=qvB.tolist()))
    return A, B, man


def _mirror_error(A, B, man):
    rep = C2Rep(man); chans = z_channel_names(man)
    err = np.abs(B[chans].to_numpy() - rep.mirror_only(A[chans].to_numpy()))
    worst = sorted({c: float(err[:, i].max()) for i, c in enumerate(chans)}.items(), key=lambda kv: -kv[1])[:5]
    return float(err.max()), worst


def test_manifest_involution_and_wheel_angle_exclusion():
    from geofdi.sim.telemetry_m1 import build_manifest
    man = build_manifest(); rep = C2Rep(man)
    assert np.allclose(rep.P @ rep.P, np.eye(rep.d))
    assert not any(n.startswith("q_") and n.endswith("WHEEL") for n in rep.names)
    assert any(n.startswith("dq_") and n.endswith("WHEEL") for n in rep.names)
    assert rep.d == 16 * 4 - 4 + 6 + 4


def test_m1_wheeled_sym_world_is_mirror_equivariant():
    A, B, man = _pair("m1_wheeled_sym")
    err, worst = _mirror_error(A, B, man)
    assert A["base_x"].iloc[-1] > 1.0                          # it actually rolled
    assert err < 1e-10, worst


def test_m1_stepping_mode_is_mirror_equivariant_with_half_period_shift():
    """Stepping mode (equivariant PD trot on the leg joints, wheels held): B from the mirrored state with the controller
    clock offset by half a period must equal rho(g_s) A row by row (Sigma = (g_s, 1/2))."""
    qpos0, qvel0 = keyframe_state_m1("m1_wheeled_sym"); rng = np.random.default_rng(5)
    qpos0[7:23] += rng.normal(0, 0.03, 16) * np.tile([1, 1, 1, 0], 4); qvel0[6:22] += rng.normal(0, 0.1, 16)
    qpB, qvB = mirror_state(qpos0, qvel0)
    A, man = rollout_m1(SimConfigM1(model="m1_wheeled_sym", mode="stepping", speed=0.0, duration_s=3.0, seed=0, noise=ZERO_NOISE, init_qpos=qpos0.tolist(), init_qvel=qvel0.tolist()))
    B, _ = rollout_m1(SimConfigM1(model="m1_wheeled_sym", mode="stepping", speed=0.0, duration_s=3.0, seed=0, noise=ZERO_NOISE, init_qpos=qpB.tolist(), init_qvel=qvB.tolist(), phase_offset=0.5))
    err, worst = _mirror_error(A, B, man)
    assert err < 1e-10, worst


def test_m1_wheeled_original_world_is_chiral():
    """The original zgws numbers (base com_y, RAR knee, products) break the mirror symmetry measurably: recorded as the
    eps_dyn candidate; the test asserts the ORDER of the effect (>1e-6) so that a silently symmetrized build is noticed."""
    A, B, man = _pair("m1_wheeled")
    err, worst = _mirror_error(A, B, man)
    assert err > 1e-6, worst
