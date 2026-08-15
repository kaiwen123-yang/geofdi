"""Every injector, wired into the closed-loop rollout, changes the telemetry the way its docstring says
(same seed with/without the fault, so differences are the fault alone)."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from geofdi.sim.env import SimConfig, rollout

BASE = dict(duration_s=8.0, seed=11)


def _pair(fault=None, nuisance=None):
    a, _ = rollout(SimConfig(**BASE))
    b, _ = rollout(SimConfig(**BASE, faults=[fault] if fault else [], nuisance=[nuisance] if nuisance else []))
    return a, b


def _post(df, t=3.0):
    return df[df.t > t]


def test_actuator_gain_scales_measured_torque():
    a, b = _pair(dict(type="actuator_gain", t_onset=3.0, leg="LF", joint="KFE", magnitude=-0.4))
    r = np.median(np.abs(_post(b).tau_meas_LF_KFE) / (np.abs(_post(b).tau_cmd_LF_KFE) + 1e-9))
    assert 0.5 < r < 0.7, r
    r_other = np.median(np.abs(_post(b).tau_meas_RF_KFE) / (np.abs(_post(b).tau_cmd_RF_KFE) + 1e-9))
    assert 0.9 < r_other < 1.1


def test_actuator_bias_shifts_measured_torque():
    a, b = _pair(dict(type="actuator_bias", t_onset=3.0, leg="LF", joint="HFE", magnitude=1.0))
    d = np.median(_post(b).tau_meas_LF_HFE - _post(b).tau_cmd_LF_HFE)
    assert 0.8 < d < 1.2, d


def test_deadzone_zeroes_small_torques():
    a, b = _pair(dict(type="deadzone", t_onset=3.0, leg="RF", joint="HAA", magnitude=3.0))
    fa = np.mean(np.abs(_post(a).tau_meas_RF_HAA) < 0.6); fb = np.mean(np.abs(_post(b).tau_meas_RF_HAA) < 0.6)
    assert fb > fa + 0.1, (fa, fb)


def test_delay_lags_command():
    a, b = _pair(dict(type="delay", t_onset=0.0, leg="LH", joint="KFE", magnitude=0.02))   # 4 control steps
    x = _post(b).tau_cmd_LH_KFE.to_numpy(); y = _post(b).tau_meas_LH_KFE.to_numpy()
    lags = range(8); cc = [np.corrcoef(x[:len(x) - 8], y[l:len(y) - 8 + l])[0, 1] for l in lags]
    assert int(np.argmax(cc)) == 4, cc


def test_encoder_bias_jumps_measurement_at_onset():
    a, b = _pair(dict(type="encoder_bias", t_onset=3.0, leg="LF", joint="HFE", magnitude=0.05))
    d = (b.q_LF_HFE - a.q_LF_HFE).to_numpy(); t = a.t.to_numpy()
    j = int(np.argmax(np.abs(d) > 1e-12))                  # first row that differs = first faulted measurement
    assert 3.0 <= t[j] <= 3.011 and abs(d[j] - 0.05) < 1e-9, (t[j], d[j])


@pytest.mark.parametrize("ftype,kw", [("friction_scale", dict(leg="LF", joint="KFE", magnitude=2.0)),
                                      ("inertia_add", dict(leg="LF", joint="KFE", magnitude=0.2)),
                                      ("inertia_add", dict(leg="LF", joint="HFE", magnitude=0.2)),
                                      ("foot_friction", dict(leg="LF", magnitude=-0.5))])
def test_model_faults_change_dynamics(ftype, kw):
    a, b = _pair(dict(type=ftype, t_onset=3.0, **kw))
    pre = np.abs(_post(b, 0.0)[b.t < 3.0][["q_LF_KFE", "tau_meas_LF_KFE"]].to_numpy() - a[a.t < 3.0][["q_LF_KFE", "tau_meas_LF_KFE"]].to_numpy()).max()
    post = np.abs(_post(b)[["q_LF_KFE", "tau_meas_LF_KFE"]].to_numpy() - _post(a)[["q_LF_KFE", "tau_meas_LF_KFE"]].to_numpy()).max()
    assert pre < 1e-9 and post > 1e-3, (pre, post)


def test_payloads_and_drifts():
    a, b = _pair(nuisance=dict(type="payload_symmetric", t_onset=0.0, magnitude=2.0))
    assert _post(b).base_z.mean() < _post(a).base_z.mean() - 0.002
    a, c = _pair(nuisance=dict(type="payload_asymmetric", t_onset=0.0, magnitude=1.0, params={"offset_y": 0.05}))
    assert abs(_post(c).imu_a_y.mean() - _post(a).imu_a_y.mean()) > 0.01
    a, d = _pair(nuisance=dict(type="drift_lateral", t_onset=0.0, magnitude=0.03, params={"tau_s": 1.0}))
    assert d.qref_LF_HAA.std() > 1.5 * a.qref_LF_HAA.std()      # (the yaw-damping stabilizer already moves qref_HAA a little)
    a, e = _pair(nuisance=dict(type="drift_symmetric", t_onset=0.0, magnitude=0.1, params={"tau_s": 5.0}))
    ratio = (_post(e).tau_meas_LF_KFE - _post(e).tau_cmd_LF_KFE)   # drift multiplies applied torque -> residual scales with tau
    assert np.corrcoef(ratio, _post(e).tau_cmd_LF_KFE)[0, 1] ** 2 > 0.05
