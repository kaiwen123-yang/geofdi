"""Pinocchio floating-base model of the Go2 (from the URDF) vs MuJoCo (go2_urdf MJCF): mass matrix, gravity, bias
forces, foot positions and Jacobians agree in random configurations to < 1e-6 relative error."""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from geofdi.dynamics.pin_model import Go2Dynamics, random_state   # noqa: E402


@pytest.fixture(scope="module")
def dyn():
    return Go2Dynamics("pin"), Go2Dynamics("mujoco")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def test_mass_matrix_gravity_bias_match(dyn):
    P, Mj = dyn; rng = np.random.default_rng(0)
    worst = {"M": 0.0, "g": 0.0, "h": 0.0, "id": 0.0}
    for _ in range(20):
        qpos, qvel = random_state(rng)
        worst["M"] = max(worst["M"], _rel(P.mass_matrix(qpos), Mj.mass_matrix(qpos)))
        worst["g"] = max(worst["g"], _rel(P.gravity(qpos), Mj.gravity(qpos)))
        worst["h"] = max(worst["h"], _rel(P.bias(qpos, qvel), Mj.bias(qpos, qvel)))
        qacc = rng.normal(0, 3, 18)
        worst["id"] = max(worst["id"], _rel(P.inverse_dynamics(qpos, qvel, qacc), Mj.inverse_dynamics(qpos, qvel, qacc)))
    assert all(v < 1e-6 for v in worst.values()), worst


def test_foot_kinematics_match(dyn):
    P, Mj = dyn; rng = np.random.default_rng(1)
    for _ in range(10):
        qpos, qvel = random_state(rng)
        assert np.max(np.abs(P.foot_positions(qpos) - Mj.foot_positions(qpos))) < 1e-9
        assert _rel(P.foot_jacobians(qpos), Mj.foot_jacobians(qpos)) < 1e-8


def test_coriolis_matrix_consistent_with_bias(dyn):
    P, _ = dyn; rng = np.random.default_rng(2)
    for _ in range(5):
        qpos, qvel = random_state(rng)
        C = P.coriolis_matrix(qpos, qvel)
        assert _rel(C @ qvel + P.gravity(qpos), P.bias(qpos, qvel)) < 1e-9


def test_pin_convention_coriolis_is_christoffel_consistent(dyn):
    """In the Pinocchio coordinates Mdot = C + C^T (finite-difference check) — what the momentum observer relies on."""
    P, _ = dyn; rng = np.random.default_rng(3); pin = P.pin
    for _ in range(3):
        qpos, qvel = random_state(rng, vel_scale=0.5)
        q = P._to_pin_q(qpos); v = P.to_pin_velocity(qpos, qvel); h = 1e-6
        q2 = pin.integrate(P.model, q, v * h)
        M1 = P.mass_matrix_pin(qpos)
        Mm = pin.crba(P.model, P.data, q2); M2 = np.triu(Mm) + np.triu(Mm, 1).T
        Mdot = (M2 - M1) / h; C = P.coriolis_matrix_pin(qpos, qvel)
        assert np.max(np.abs(Mdot - (C + C.T))) < 1e-4 * (1 + np.max(np.abs(Mdot)))
