"""Equivariant DeLaN (Sprint 6, Block Q): exact C2 equivariance by weight sharing.

(1) delta_f of an equivariant model on random inputs is < 1e-6 (double precision; float32 lands at ~1e-6-1e-5 because
    of non-associative rounding, reported not asserted), for both template layouts (2 and 1);
(2) a plain per-leg DeLaN (independent random weights) has a clearly nonzero defect;
(3) the residual of a mirrored trajectory equals rho_R times the original residual (mirror-equivariant residual map);
(4) M_sigma(q) = S M_0(S q) S is symmetric positive definite (structure preserved under the mirror map).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geofdi.dynamics.delan import LEG_ORDER, DeLaNQuadruped              # noqa: E402
from geofdi.dynamics.delan_equiv import EquivariantDeLaN, equivariance_defect, mirror_maps   # noqa: E402
from geofdi.residuals.mirror_pairs import residual_rep                    # noqa: E402


def _random_inputs(n=256, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.5, 1.5, (n, 3)); dq = rng.normal(0, 3, (n, 3)); ddq = rng.normal(0, 30, (n, 3))
    a = rng.normal(0, 2, (n, 3)) + np.array([0, 0, 9.81])
    return q, dq, ddq, a


def _to_double(model):
    for net in (model.templates.values() if hasattr(model, "templates") else model.nets.values()):
        net.double()
    return model


def _randomize_stats(model, rng):
    """Non-trivial input normalisation buffers (as after training) so that the test is not vacuous."""
    for net in (model.templates.values() if hasattr(model, "templates") else model.nets.values()):
        net.q_mu.copy_(torch.as_tensor(rng.normal(0, 0.5, 3), dtype=net.q_mu.dtype)); net.q_sd.copy_(torch.as_tensor(rng.uniform(0.5, 2, 3), dtype=net.q_sd.dtype))
        net.a_mu.copy_(torch.as_tensor(rng.normal(0, 1, 3), dtype=net.a_mu.dtype)); net.a_sd.copy_(torch.as_tensor(rng.uniform(0.5, 2, 3), dtype=net.a_sd.dtype))


@pytest.mark.parametrize("n_templates", [2, 1])
def test_equivariant_model_has_zero_defect(n_templates):
    torch.manual_seed(1)
    quad = _to_double(EquivariantDeLaN.build(n_templates=n_templates, hidden=32, depth=2))
    _randomize_stats(quad, np.random.default_rng(3))
    q, dq, ddq, a = _random_inputs()
    d = equivariance_defect(quad, q, dq, ddq, a)
    scale = np.abs(quad.predict("LF", q, dq, ddq, a)).max()
    assert scale > 1e-3                                  # outputs are not degenerate
    assert d["max"] < 1e-6, d
    # float32 level, reported for the record (rounding only)
    quad32 = EquivariantDeLaN.build(n_templates=n_templates, hidden=32, depth=2); _randomize_stats(quad32, np.random.default_rng(3))
    d32 = equivariance_defect(quad32, q, dq, ddq, a)
    assert d32["max"] < 1e-3


def test_plain_model_has_nonzero_defect():
    torch.manual_seed(2)
    quad = _to_double(DeLaNQuadruped.build(hidden=32, depth=2))
    _randomize_stats(quad, np.random.default_rng(4))
    q, dq, ddq, a = _random_inputs()
    d = equivariance_defect(quad, q, dq, ddq, a)
    assert d["q95"] > 1e-2, d                          # independent per-leg weights: clearly not equivariant


def test_residual_of_mirrored_trajectory_is_rho_of_residual():
    """r(rho z) = rho_R r(z) for the equivariant model: leg <-> partner with the manifest signs, phase kept."""
    torch.manual_seed(5)
    quad = _to_double(EquivariantDeLaN.build(n_templates=2, hidden=32, depth=2))
    _randomize_stats(quad, np.random.default_rng(6))
    maps = mirror_maps(); rng = np.random.default_rng(7)
    T = 300
    traj = {}
    for leg in LEG_ORDER:
        q, dq, ddq, _ = _random_inputs(T, seed=10 + LEG_ORDER.index(leg))
        traj[leg] = {"q": q, "dq": dq, "ddq": ddq, "y": rng.normal(0, 5, (T, 3))}
    a = _random_inputs(T, seed=99)[3]                     # one trunk specific force for all legs
    # mirrored trajectory: partner leg carries S * (leg data), a -> E a
    mtraj = {}
    for leg in LEG_ORDER:
        p = maps["partner"][leg]; S = maps["S"][leg]
        mtraj[p] = {k: traj[leg][k] * S for k in ("q", "dq", "ddq", "y")}
    am = a * maps["E"]
    r = np.concatenate([traj[l]["y"] - quad.predict(l, traj[l]["q"], traj[l]["dq"], traj[l]["ddq"], a) for l in LEG_ORDER], axis=1)   # (T, 12)
    rm = np.concatenate([mtraj[l]["y"] - quad.predict(l, mtraj[l]["q"], mtraj[l]["dq"], mtraj[l]["ddq"], am) for l in LEG_ORDER], axis=1)
    rep = residual_rep(include_base=False)
    r_expected = rep.mirror_only(r)                       # rho_R on the channel axis (T, 12)
    assert np.abs(rm - r_expected).max() < 1e-6
    # and the plain model violates it
    plain = _to_double(DeLaNQuadruped.build(hidden=32, depth=2)); _randomize_stats(plain, np.random.default_rng(8))
    rp = np.concatenate([traj[l]["y"] - plain.predict(l, traj[l]["q"], traj[l]["dq"], traj[l]["ddq"], a) for l in LEG_ORDER], axis=1)
    rpm = np.concatenate([mtraj[l]["y"] - plain.predict(l, mtraj[l]["q"], mtraj[l]["dq"], mtraj[l]["ddq"], am) for l in LEG_ORDER], axis=1)
    assert np.abs(rpm - rep.mirror_only(rp)).max() > 1e-2


def test_mirrored_mass_matrix_is_spd():
    torch.manual_seed(9)
    quad = _to_double(EquivariantDeLaN.build(n_templates=2, hidden=32, depth=2))
    q = torch.as_tensor(_random_inputs(64, seed=11)[0], dtype=torch.float64)
    for leg in ("RF", "RH", "LF"):
        M = quad.mass_matrix(leg, q).detach().numpy()
        assert np.allclose(M, np.swapaxes(M, 1, 2), atol=1e-12)
        assert np.linalg.eigvalsh(M).min() > 0
    # congruence identity M_RF(q) = S M_LF(S q) S
    S = mirror_maps()["S"]["RF"]
    M_rf = quad.mass_matrix("RF", q).detach().numpy(); M_lf_m = quad.mass_matrix("LF", q * torch.as_tensor(S)).detach().numpy()
    assert np.abs(M_rf - S[None, :, None] * M_lf_m * S[None, None, :]).max() < 1e-12
