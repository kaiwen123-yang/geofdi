"""Sprint 8 Block G: per-stance-event gate (detect.stance_event) FAR control and the pi_gating weight modes."""
import numpy as np

from geofdi.detect.stance_event import EventLibrary, StanceEventTracker


def _nominal_events(rng, n, dist):
    return [float(x) for x in dist(rng, n)]


def test_conformal_per_event_far_is_about_alpha():
    """Under H0 (monitoring events drawn from the same law as the library), the per-event conformal p is ~uniform, so the
    fraction of events with p <= alpha (the per-event false-alarm rate) is ~alpha."""
    rng = np.random.default_rng(0)
    lib = EventLibrary(1)
    for x in _nominal_events(rng, 2000, lambda r, n: np.abs(r.normal(0, 1, n))):
        lib.add(0, x)
    lib.finalize()
    mon = _nominal_events(rng, 5000, lambda r, n: np.abs(r.normal(0, 1, n)))
    p = np.array([lib.conformal_p(0, s) for s in mon])
    far = float(np.mean(p <= 0.05))
    assert 0.03 < far < 0.07, far                       # ~alpha


def test_tracker_gates_a_sustained_anomaly_not_a_clean_event():
    lib = EventLibrary(1); rng = np.random.default_rng(1)
    for x in np.abs(rng.normal(0, 1, 3000)):
        lib.add(0, float(x))
    lib.finalize()
    tr = StanceEventTracker(1, alpha=0.05)
    # a clean stance event (scores ~ N(0,1)) then an anomalous one (scores ~ 6): hard weight drops only on the anomaly
    for s in np.abs(rng.normal(0, 1, 20)):
        tr.update(0, True, float(s), lib)
    w_clean = tr.weights("hard")[0]; tr.update(0, False, np.nan, lib)
    for _ in range(20):
        tr.update(0, True, 6.0, lib)
    w_anom = tr.weights("hard")[0]
    assert w_clean == 1.0 and w_anom == 0.0
    assert tr.weights("soft")[0] < 0.5                  # soft gate down-weights the anomaly


def test_soft_weight_is_gentle_under_h0():
    lib = EventLibrary(1); rng = np.random.default_rng(2)
    for x in np.abs(rng.normal(0, 1, 3000)):
        lib.add(0, float(x))
    lib.finalize()
    tr = StanceEventTracker(1, alpha=0.05)
    ws = []
    for _ in range(200):
        for s in np.abs(rng.normal(0, 1, 8)):
            tr.update(0, True, float(s), lib)
        ws.append(tr.weights("soft")[0]); tr.update(0, False, np.nan, lib)
    # under H0 the soft weight stays close to 1 on average (little spurious down-weighting)
    assert np.mean(ws) > 0.8
