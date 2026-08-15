"""Injector interface: every fault/nuisance type constructs, schedules and applies without error."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from geofdi.sim.env import scene_path
from geofdi.sim.faults import FAULT_TYPES, NUISANCE_TYPES, FaultBank, FaultSpec


@pytest.mark.parametrize("ftype", FAULT_TYPES + NUISANCE_TYPES)
def test_each_type_applies(ftype):
    m = mujoco.MjModel.from_xml_path(scene_path())
    spec = FaultSpec(type=ftype, t_onset=0.5, leg="LF", joint="KFE", magnitude=0.1, schedule="ramp", ramp_s=1.0,
                     params={"tau_s": 5.0, "offset_y": 0.03, "duration_s": 1.0})
    fb = FaultBank([spec], m, 0.005, np.random.default_rng(0))
    for t in (0.0, 0.5, 1.0, 2.0):
        fb.advance(t); fb.model_update(t)
        tau = fb.torque(np.ones(12), t); q = fb.measure(np.zeros(12), t); off = fb.setpoint_offset(t)
        assert tau.shape == (12,) and q.shape == (12,) and off.shape == (12,)
        assert np.all(np.isfinite(tau)) and np.all(np.isfinite(q))
    assert spec.mask().sum() == 1


def test_schedule_and_mask_semantics():
    s = FaultSpec(type="actuator_gain", t_onset=1.0, magnitude=-0.4, schedule="ramp", ramp_s=2.0)
    assert s.s(0.5) == 0 and abs(s.s(2.0) - 0.5) < 1e-12 and s.s(10) == 1
    assert FaultSpec(type="deadzone", leg="RH").mask().sum() == 3 and FaultSpec(type="deadzone").mask().sum() == 12
    with pytest.raises(ValueError):
        FaultSpec(type="not_a_fault")
