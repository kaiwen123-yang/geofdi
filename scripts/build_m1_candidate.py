#!/usr/bin/env python3
"""Block M: derive headless GeoFDI worlds from the MATRiX `zgws` (GENISOM M1, WHEELED) MJCF.

    python scripts/build_m1_candidate.py <path/to/MATRiX_Python_SDK/model/zgws/zgws.xml>

Writes into src/geofdi/sim/assets/m1/:
  zgws_source.xml                 verbatim copy of the MATRiX MJCF (BSD-3-Clause, GENISOM AI / ZsiBot; meshes NOT copied)
  m1_wheeled_headless.xml         mesh-free: base box from the mesh extents, existing box collisions, wheels as cylinders
  m1_pointfoot_candidate.xml      CANDIDATE point-foot M1 v0.1: wheel (FOOT) joints removed, calf length 0.325 m and foot-pad
                                  sphere r = 0.03 m from the STEP dimension check (zgws wheeled: 0.28 m + wheel r 0.096),
                                  wheel-link masses KEPT as placeholders (a point foot is lighter: flagged), joint
                                  ranges/actuators as zgws; IMU site as zgws (base origin)
  scene_m1_pointfoot_candidate.xml / scene_m1_wheeled_headless.xml
and runs (a) a 60 s standing smoke test (PD hold) and (b) the mirror-symmetry check on the point-foot candidate;
prints the numbers for docs/protocol/m1_model_audit.md.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "geofdi" / "sim" / "assets" / "m1"
LEGS = ("FAR", "FBL", "RAR", "RBL")            # FR, FL, RR, RL (MATRiX naming: A = right, B = left)
BASE_BOX = (0.8333 / 2, 0.2302 / 2, 0.2082 / 2)      # from BASE_LINK.STL extents (m)
WHEEL_R, WHEEL_HALF_W = 0.096, 0.025          # FOOT_LINK.STL extents 0.192 x 0.0867 x 0.192
FOOT_R = 0.03                                 # point-foot pad radius: STEP shows a lateral-axis foot pad cylinder r = 0.030 m
CALF_LEN_STEP = 0.325                         # STEP: knee axis -> foot-pad axis 0.325 m (zgws wheeled: 0.28 m to the wheel axle)


def strip_meshes(root, wheeled: bool):
    for asset in root.findall("asset"):
        root.remove(asset)
    comp = root.find("compiler")
    if comp is not None:
        comp.attrib.pop("meshdir", None); comp.set("autolimits", "true")
    for body in root.iter("body"):
        for g in list(body.findall("geom")):
            if g.get("type") == "mesh":
                body.remove(g)
        name = body.get("name", "")
        if name == "base_link":
            ET.SubElement(body, "geom", type="box", size=" ".join(f"{v:.4f}" for v in BASE_BOX), pos="0 0 -0.02", rgba="0.7 0.7 0.7 1")
        if name.endswith("_FOOT_LINK"):
            if wheeled:
                ET.SubElement(body, "geom", name=name.replace("_FOOT_LINK", "_wheel"), type="cylinder", size=f"{WHEEL_R} {WHEEL_HALF_W}",
                              quat="0.7071068 0.7071068 0 0", pos="0 -0.0394 0" if name.startswith(("FAR", "RAR")) else "0 0.0394 0",
                              friction="0.8 0.02 0.01", condim="6", priority="1")
            else:
                for j in list(body.findall("joint")):
                    body.remove(j)                     # wheel joint removed -> fixed foot
                pos = [float(v) for v in body.get("pos").split()]; pos[2] = -CALF_LEN_STEP     # STEP-derived calf length
                body.set("pos", " ".join(f"{v:g}" for v in pos))
                ET.SubElement(body, "geom", name=name[:3], type="sphere", size=f"{FOOT_R}", pos="0 0 0", friction="0.8 0.02 0.01",
                              condim="6", priority="1", solimp="0.015 1 0.022")


def fix_actuators(root, wheeled: bool):
    act = root.find("actuator")
    if act is None:
        return
    for m in list(act):
        if (not wheeled) and m.get("joint", "").endswith("_FOOT_JOINT"):
            act.remove(m); continue
        m.set("ctrlrange", "-150 150"); m.set("ctrllimited", "true")


def add_sensors_keyframe(root, wheeled: bool):
    for s in root.findall("sensor"):
        root.remove(s)
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "accelerometer", name="imu_acc", site="imu"); ET.SubElement(sen, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sen, "framequat", name="base_quat", objtype="site", objname="imu"); ET.SubElement(sen, "framelinvel", name="base_linvel", objtype="site", objname="imu")
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    stand = [0.0, 0.8, -1.5] + ([0.0] if wheeled else [])
    q = [0, 0, 0.42, 1, 0, 0, 0] + stand * 4
    ET.SubElement(kf, "key", name="stand", qpos=" ".join(f"{v:g}" for v in q))


def build(src: Path, wheeled: bool) -> Path:
    tree = ET.parse(src); root = tree.getroot()
    root.set("model", "m1_wheeled_headless" if wheeled else "m1_pointfoot_candidate")
    strip_meshes(root, wheeled); fix_actuators(root, wheeled); add_sensors_keyframe(root, wheeled)
    hdr = ET.Comment(f" derived by scripts/build_m1_candidate.py from MATRiX_Python_SDK/model/zgws/zgws.xml (GENISOM AI / ZsiBot, BSD-3-Clause; "
                     f"MJCF of the WHEELED M1 'zgws'). {'Wheels as cylinders (r 0.096).' if wheeled else 'CANDIDATE point foot v0.1: wheel joints removed, calf length 0.325 m and foot-pad sphere r=0.03 m from the STEP dimension check, wheel-link masses KEPT as placeholders (flagged), hip lateral offset unverified.'} "
                     "Meshes replaced by primitives (base box from the STL extents). Not validated against the real M1 (Day-0 audit pending). ")
    root.insert(0, hdr)
    ET.indent(root, space="  ")
    name = "m1_wheeled_headless" if wheeled else "m1_pointfoot_candidate"
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / f"{name}.xml"
    tree.write(path, encoding="unicode", xml_declaration=False)
    (OUT / f"scene_{name}.xml").write_text(f'<mujoco model="{name} scene">\n  <include file="{name}.xml"/>\n  <worldbody>\n    <geom name="floor" size="0 0 0.05" type="plane" friction="0.8 0.02 0.01"/>\n  </worldbody>\n</mujoco>\n')
    return path


def standing_smoke(scene: Path, seconds: float = 60.0, kp=150.0, kd=4.0):
    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    qref = d.qpos[7:].copy(); z0 = []; rp = []
    nsteps = int(seconds / m.opt.timestep)
    for k in range(nsteps):
        d.ctrl[:] = np.clip(kp * (qref - d.qpos[7:]) - kd * d.qvel[6:], -150, 150)
        mujoco.mj_step(m, d)
        if k % 200 == 0:
            z0.append(d.qpos[2]); q = d.qpos[3:7]
            rp.append([np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)), np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))])
    z0 = np.array(z0); rp = np.degrees(np.array(rp))
    return {"z_final": float(z0[-1]), "z_min": float(z0.min()), "z_max": float(z0.max()), "roll_max_deg": float(np.abs(rp[:, 0]).max()), "pitch_max_deg": float(np.abs(rp[:, 1]).max()), "nsteps": nsteps, "dt": float(m.opt.timestep)}


def mirror_check(scene: Path, seconds: float = 3.0, kp=150.0, kd=4.0):
    """Mirror the initial state (y -> -y, FR<->FL, RR<->RL, ABAD sign flip) and compare mirrored trajectories under the
    same mirrored PD controller: max |q_B(t) - S q_A(t)| over 3 s. Chiral model details show up as a nonzero residual."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    nj = m.nu; per = nj // 4
    perm = np.concatenate([np.arange(per) + per, np.arange(per), np.arange(per) + 3 * per, np.arange(per) + 2 * per])
    sgn = np.tile([-1.0] + [1.0] * (per - 1), 4)
    def run(qpos0, qvel0, mirrored):
        d = mujoco.MjData(m); d.qpos[:] = qpos0; d.qvel[:] = qvel0; mujoco.mj_forward(m, d)
        qref = m.key_qpos[0][7:].copy()
        if mirrored:
            qref = sgn * qref[perm]
        traj = []
        for k in range(int(seconds / m.opt.timestep)):
            d.ctrl[:] = np.clip(kp * (qref - d.qpos[7:]) - kd * d.qvel[6:], -150, 150)
            mujoco.mj_step(m, d)
            if k % 4 == 0:
                traj.append(np.concatenate([d.qpos[7:].copy(), d.qvel[6:].copy(), d.qpos[:3].copy(), d.qvel[:6].copy()]))
        return np.array(traj)
    rng = np.random.default_rng(3); d0 = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d0, 0)
    qA = d0.qpos.copy(); vA = d0.qvel.copy(); qA[7:] += rng.normal(0, 0.05, nj); vA[6:] += rng.normal(0, 0.2, nj); vA[3:6] += rng.normal(0, 0.1, 3)
    qB = qA.copy(); vB = vA.copy(); qB[1] = -qA[1]; w, x, y, z = qA[3:7]; qB[3:7] = [w, -x, y, -z]
    qB[7:] = sgn * qA[7:][perm]; vB[0:3] = vA[0:3] * [1, -1, 1]; vB[3:6] = -vA[3:6] * [1, -1, 1]; vB[6:] = sgn * vA[6:][perm]
    A = run(qA, vA, False); B = run(qB, vB, True)
    qerr = np.abs(B[:, :nj] - sgn * A[:, :nj][:, perm]).max(); dqerr = np.abs(B[:, nj:2 * nj] - sgn * A[:, nj:2 * nj][:, perm]).max()
    pos_err = np.abs(B[:, 2 * nj:2 * nj + 3] - A[:, 2 * nj:2 * nj + 3] * [1, -1, 1]).max()
    return {"q_mirror_residual_rad": float(qerr), "dq_mirror_residual": float(dqerr), "base_pos_mirror_residual_m": float(pos_err), "seconds": seconds}


def main():
    src = Path(sys.argv[1]); assert src.exists(), src
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zgws_source.xml").write_text(src.read_text())
    res = {}
    for wheeled in (True, False):
        p = build(src, wheeled); m = mujoco.MjModel.from_xml_path(str(OUT / f"scene_{p.stem}.xml"))
        res[p.stem] = {"nbody": m.nbody, "njnt": m.njnt, "nu": m.nu, "mass": float(m.body_subtreemass[1])}
        print(p.stem, res[p.stem])
    sc = OUT / "scene_m1_pointfoot_candidate.xml"
    res["standing_smoke_pointfoot"] = standing_smoke(sc); print("standing smoke (point-foot candidate, 60 s):", res["standing_smoke_pointfoot"])
    res["mirror_check_pointfoot"] = mirror_check(sc); print("mirror check (point-foot candidate):", res["mirror_check_pointfoot"])
    import json
    (REPO / "results" / "m1_model_audit" / "m1_candidate_checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
