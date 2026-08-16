"""Sprint 9 Block Q: the QUADRIC-GINS Go2 loader — transcript parser tolerance, mirror representation (t01-style),
and the straight-segment dual criterion. No data volume required: the transcript fixture is built inline."""
import numpy as np
import pandas as pd
import pytest

from geofdi.groups.c2 import C2Rep
from geofdi.io.go2_highlevel_txt import parse_sportmodestate
from geofdi.io.go2_quadric import LEGS, MIRROR_LEG, build_go2_quadric_manifest, straight_mask_go2

REC = """stamp:
  sec: {sec}
  nanosec: {nsec}
error_code: 0
imu_state:
  quaternion:
  - 1.0
  - 0.0
  - 0.0
  - 0.0
  gyroscope:
  - 0.01
  - 0.02
  - {wz}
  accelerometer:
  - 0.05
  - -0.07
  - 9.36
  rpy:
  - 0.0
  - 0.0
  - 0.0
  temperature: 79
mode: 3
progress: 0.0
gait_type: 1
foot_raise_height: 0.09
position:
- {px}
- 0.0
- 0.32
body_height: 0.32
velocity:
- {vx}
- 0.0
- 0.0
yaw_speed: {wz}
range_obstacle:
- 2.0
- 2.0
- 2.0
- 2.0
foot_force:
- 23
- 20
- 18
- 25
foot_position_body:
- 0.178
- -0.126
- -0.315
- 0.178
- 0.121
- -0.319
- -0.197
- -0.125
- -0.313
- -0.200
- 0.127
- -0.319
foot_speed_body:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
---
"""


def _transcript(tmp_path, n=40, wz=0.0, vx=1.0, ansi=True, extra_field=False):
    p = tmp_path / "sess.txt"
    head = ("Script started on 2026-01-05 20:24:49+08:00 [TERM=\"xterm-256color\"]\n"
            "\x1b]0;user@host: ~\x07\x1b[01;32muser@host\x1b[00m:\x1b[01;34m~\x1b[00m$ ros2 topic echo /sportmodestate \x08\x08\x08\r\n") if ansi else ""
    body = ""
    for i in range(n):
        r = REC.format(sec=1767615915 + i // 200, nsec=(i % 200) * 5_000_000, wz=wz, vx=vx, px=0.1 * i)
        if extra_field and i == 0:
            r = r.replace("body_height: 0.32", "body_height: 0.32\nvendor_new_field: 42.0")
        body += r
    p.write_bytes((head + body).replace("\n", "\r\n").encode())
    return p


def test_parser_survives_ansi_and_reports_rate(tmp_path):
    df, rep = parse_sportmodestate(_transcript(tmp_path, n=60))
    assert rep["n_records"] == 60 and rep["n_time_backjumps"] == 0
    assert np.isclose(rep["rate_hz_median"], 200.0, rtol=0.05)
    assert not rep["has_motorstate"]                       # SportModeState carries no joint stream
    assert np.isclose(df["acc2"].iloc[0], 9.36) and np.isclose(df["foot_force0"].iloc[0], 23)
    assert df["foot_pos1"].iloc[0] < 0 < df["foot_pos4"].iloc[0]     # FR y < 0 < FL y (Unitree leg order)


def test_parser_keeps_unknown_fields(tmp_path):
    """A vendor field the schema does not know must be KEPT (as col_<path>), never silently dropped."""
    df, rep = parse_sportmodestate(_transcript(tmp_path, n=5, extra_field=True))
    assert "vendor_new_field" in rep["extra_paths"] and "col_vendor_new_field" in df.columns
    assert np.isclose(np.nanmax(df["col_vendor_new_field"]), 42.0)


def test_mirror_representation_is_an_involution_with_the_right_signs():
    man = build_go2_quadric_manifest(); rep = C2Rep(man)
    P = rep.P
    assert np.allclose(P @ P, np.eye(len(P))) and np.allclose(P.T, np.linalg.inv(P))
    names = [c["name"] for c in man["channels"] if c["in_Z"]]
    assert len(names) == 34                                 # foot_pos 12 + foot_vel 12 + foot_force 4 + imu 6
    for leg in LEGS:
        m = MIRROR_LEG[leg]
        for stem in ("foot_pos", "foot_vel"):
            assert P[names.index(f"{stem}_{m}_x"), names.index(f"{stem}_{leg}_x")] == +1     # polar: x keeps sign
            assert P[names.index(f"{stem}_{m}_y"), names.index(f"{stem}_{leg}_y")] == -1     # y flips
            assert P[names.index(f"{stem}_{m}_z"), names.index(f"{stem}_{leg}_z")] == +1
        assert P[names.index(f"foot_force_{m}"), names.index(f"foot_force_{leg}")] == +1     # magnitude
    assert P[names.index("imu_w_y"), names.index("imu_w_y")] == +1                            # axial: y keeps, x/z flip
    assert P[names.index("imu_a_y"), names.index("imu_a_y")] == -1                            # polar: y flips


def test_mirror_symmetric_element_is_a_fixed_point():
    """t01-style exactness: an element built symmetric by construction must satisfy rho Z = Z exactly."""
    man = build_go2_quadric_manifest(); rep = C2Rep(man)
    names = [c["name"] for c in man["channels"] if c["in_Z"]]
    rng = np.random.default_rng(0); K, N = 6, 16
    Z = np.zeros((K, len(names), N))
    for i, n in enumerate(names):
        Z[:, i, :] = rng.normal(size=(K, N))
    Zs = rep.apply("s", Z)                                   # symmetrize, then the image must be a fixed point
    Zsym = 0.5 * (Z + Zs)
    assert np.allclose(rep.apply("s", Zsym), Zsym, atol=1e-12)


def _frame(n=4000, wz=0.0, vx=1.0, rate=200.0):
    t = np.arange(n) / rate
    df = pd.DataFrame({"t": t, "imu_w_z": wz + 0.4 * np.sin(2 * np.pi * 2.0 * t),   # gait wobble at 2 Hz
                       "base_vx": vx, "base_vy": 0.0, "mode": 3.0})
    df["rtk_yaw"] = np.rad2deg(wz * t); df["rtk_fix_ok"] = True
    return df


def test_straight_mask_ignores_gait_wobble_but_catches_turns():
    """The mean-based criterion must pass a straight trot whose per-cycle yaw wobble is >> the threshold, and reject a
    genuine turn of the same wobble amplitude."""
    m_straight, i_s = straight_mask_go2(_frame(wz=0.0))
    m_turn, i_t = straight_mask_go2(_frame(wz=0.5))          # 0.5 rad/s sustained turn
    assert i_s["fraction"] > 0.8 and i_s["n_runs"] == 1        # 17 of 20 s: the 3 s warm-up is excluded by design
    assert i_t["fraction"] == 0.0
    assert i_s["used_rtk"] and np.isclose(i_s["median_speed_mps"], 1.0, atol=0.05)


def test_straight_mask_requires_locomotion_mode_and_motion():
    df = _frame(); df["mode"] = 1.0                          # balance-stand
    assert straight_mask_go2(df)[1]["fraction"] == 0.0
    df2 = _frame(vx=0.05)                                    # moving too slowly
    assert straight_mask_go2(df2)[1]["fraction"] == 0.0
