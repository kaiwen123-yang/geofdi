"""Exactness of the Hemerik–Goeman random-subset test and calibrator sanity (synthetic, fast)."""
import numpy as np
from scipy import stats

from geofdi.detect.evalue import eprocess, p_to_e
from geofdi.detect.permutation import hg_permutation_tests


class SwapRep:
    """Toy representation: two channels that are mirror partners with sign +1, no phase structure."""
    def apply(self, s, Z):
        Z = np.asarray(Z)
        return Z[:, ::-1, :] if s == "s" else Z.copy()


def test_pvalues_uniform_under_h0():
    rng = np.random.default_rng(7)
    ps = {"paired_energy": [], "energy_distance": []}
    for _ in range(200):
        Z = rng.normal(size=(20, 2, 3)) + 1.0
        out = hg_permutation_tests(Z, SwapRep(), M=64, rng=rng)
        for k in ps:
            ps[k].append(out[k]["p"])
    for k, v in ps.items():
        v = np.array(v)
        assert v.min() >= 1 / 64 - 1e-12 and v.max() <= 1.0
        assert stats.kstest(v, "uniform").pvalue > 0.01, k
        assert np.mean(v <= 0.05) <= 0.09, k        # 200 draws: binomial 95% upper bound ~ 0.08


def test_power_against_asymmetry():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40, 2, 3)); Z[:, 0, :] += 0.8       # channel 0 shifted vs its mirror partner
    out = hg_permutation_tests(Z, SwapRep(), M=256, rng=rng)
    assert out["paired_energy"]["p"] < 0.05 and out["energy_distance"]["p"] < 0.05


def test_calibrator_integrates_to_one_and_ville_alarm():
    p = np.linspace(1e-6, 1, 400001)
    assert abs(np.trapezoid(p_to_e(p), p) - 1.0) < 2e-3
    E, alarm = eprocess([0.5, 0.5, 0.001, 0.001], alpha=0.05)
    assert alarm == 3 and E[3] >= 20
