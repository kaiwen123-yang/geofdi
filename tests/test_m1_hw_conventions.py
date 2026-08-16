"""Sprint 8 Block D: hardware conventions of the M1 loader — vendor-frame mirror signs, IMU sensor->body rotation and
g -> m/s^2 conversion, name matching (verified names and bare fallback), and the kinematic straight-segment fallback."""
import numpy as np
import pandas as pd
import yaml

from geofdi.io.m1_sdk import load_m1_session, load_mapping, sdk_name
from geofdi.phase.registration import straight_mask_kinematic
from geofdi.sim.telemetry_m1 import JOINT_SIGN

G0 = 9.80665


def _vendor_session(tmp_path, bare=False, wz=-0.2):
    mp = load_mapping(); n = 2000; t = np.arange(n) / 200.0
    js = {"t": t}
    # a mirror-symmetric rolling posture in VENDOR coordinates: all four joints flip between fl/fr and bl/br
    base = {"ABAD": -0.5, "HIP": 1.85, "KNEE": 1.6}
    for leg, sgn in (("LF", 1), ("RF", -1), ("LH", 1), ("RH", -1)):
        for j in ("ABAD", "HIP", "KNEE", "WHEEL"):
            name = sdk_name(mp, leg, j, bare=bare)
            js[f"{name}_pos"] = sgn * (base.get(j, 0.0) + 0.01 * np.sin(2 * np.pi * 0.5 * t)) if j != "WHEEL" else sgn * np.mod(6.5 * t + np.pi, 2 * np.pi) - np.pi
            js[f"{name}_vel"] = sgn * (0.01 * np.cos(2 * np.pi * 0.5 * t)) if j != "WHEEL" else sgn * (-6.5 + 0.1 * np.sin(t))
            js[f"{name}_eff"] = sgn * (10.0 + 0.5 * np.sin(3 * t))
    pd.DataFrame(js).to_csv(tmp_path / "joint_states.csv", index=False)
    # IMU in the sensor frame (x right, y back, z down), specific force in g, yaw-left rate 0.2 rad/s
    imu = pd.DataFrame({"t": t, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "wx": 0.0, "wy": 0.0, "wz": wz, "ax": 0.0, "ay": -0.1, "az": -0.987})
    imu.to_csv(tmp_path / "imu.csv", index=False)
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump({"robot": "m1", "source": "hardware", "imu_accel_units": "g", "rate_hz": 200}))
    return tmp_path


def test_vendor_signs_make_mirror_pairs_equal(tmp_path):
    df, man, rep = load_m1_session(_vendor_session(tmp_path))
    assert rep["sign_convention"] == "vendor" and rep["mapping_unverified"] is False and not [m for m in rep["missing"] if m != "cmd.csv (v_cmd)"]
    for grp in ("q", "dq", "tau_meas"):
        for j in ("ABAD", "HIP", "KNEE", "WHEEL"):
            if grp == "q" and j == "WHEEL":
                continue
            l, r = df[f"{grp}_LF_{j}"].to_numpy(), df[f"{grp}_RF_{j}"].to_numpy()
            assert np.allclose(l, JOINT_SIGN[j] * r), (grp, j)          # mirror image == partner after per_leg_sign


def test_bare_names_are_accepted(tmp_path):
    df, man, rep = load_m1_session(_vendor_session(tmp_path, bare=True))
    assert rep["joint_names_used"]["LF_ABAD"] == "fl1" and np.isfinite(df["q_RF_HIP"]).all()


def test_imu_rotated_to_body_flu_and_scaled(tmp_path):
    df, man, rep = load_m1_session(_vendor_session(tmp_path))
    a = df[["imu_a_x", "imu_a_y", "imu_a_z"]].iloc[0].to_numpy(); w = df[["imu_w_x", "imu_w_y", "imu_w_z"]].iloc[0].to_numpy()
    assert np.allclose(a, [0.1 * G0, 0.0, 0.987 * G0], atol=1e-9)      # sensor -y (back) -> body +x; z down -> z up
    assert np.allclose(w, [0.0, 0.0, 0.2])                              # yaw-left positive in the body frame
    assert rep["imu"]["accel_scale"] == G0


def test_sim_convention_is_untouched(tmp_path):
    p = _vendor_session(tmp_path)
    (p / "meta.yaml").write_text(yaml.safe_dump({"robot": "m1", "source": "sim (test)", "rate_hz": 200}))
    df, man, rep = load_m1_session(p)
    assert rep["sign_convention"] == "geofdi_uniform_axis" and rep["imu"]["convention"] == "geofdi_body_mps2"
    assert np.isclose(df["imu_a_z"].iloc[0], -0.987)                    # no rotation, no scaling


def test_kinematic_straight_mask_finds_the_straight_stretch(tmp_path):
    df, man, rep = load_m1_session(_vendor_session(tmp_path, wz=0.0))
    t = df["t"].to_numpy(); turn = (t > 4) & (t < 6)
    df.loc[turn, "imu_w_z"] = 1.0                                        # a 2-s turn in the middle
    mask, info = straight_mask_kinematic(df)
    assert info["n_runs"] == 2 and 5.0 < info["masked_duration_s"] < 8.5 and not mask[turn].any()
