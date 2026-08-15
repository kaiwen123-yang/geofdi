#!/usr/bin/env python3
"""Sprint 7 Block W1: the wheeled M1 (`zgws`) worlds of record.

    python scripts/build_m1_wheeled_worlds.py [--with-meshes ~/research/third_party/MATRiX_Python_SDK/model/zgws/assets]

From src/geofdi/sim/assets/m1/zgws_source.xml (verbatim MATRiX MJCF, BSD-3-Clause, GENISOM AI / ZsiBot) writes
  m1_wheeled.xml       ORIGINAL: every number from the source (incl. the chiral details: base com_y +3.4 mm, base inertia
                       products, RAR knee 3.3 g lighter, mesh-fit pos/quat artefacts on the left knees), meshes replaced by
                       primitives (base box from the STL extents, existing box collisions, wheels as cylinders r 0.096 with solref 0.05 = tire compliance).
  m1_wheeled_sym.xml   SYMMETRIZED: left legs = exact mirror of the right-leg templates (front: FAR, hind: RBL mirrored into
                       RAR, i.e. the majority knee mass 0.86312 kg on all four), base com_y = 0, base inertia products
                       (xy, yz) removed by I <- (I + E I E)/2.
  scene_*.xml          floor plane; option timestep 0.0025.
Both: joint damping 0.05 (all 16), armature 0.01, ctrlrange HIP/KNEE +-60, ABAD +-40, WHEEL +-20 N m (start values,
recorded), actuatorfrcrange as in the source (legs +-150, wheels +-40), IMU site at the base origin, S0-style sensors,
'stand' keyframe (ABAD 0, HIP 0.8, KNEE -1.5, WHEEL 0; z 0.42).
--with-meshes writes an additional m1_wheeled_meshes.xml whose meshdir points OUTSIDE the repo (never committed).
"""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src" / "geofdi" / "sim" / "assets" / "m1"
SRC = OUT / "zgws_source.xml"
BASE_BOX = (0.8333 / 2, 0.2302 / 2, 0.2082 / 2)      # BASE_LINK.STL extents (m)
WHEEL_R, WHEEL_HALF_W = 0.096, 0.025
CTRLRANGE = {"ABAD": 40.0, "HIP": 60.0, "KNEE": 60.0, "FOOT": 20.0}
DAMPING, ARMATURE = 0.05, 0.01
E = np.diag([1.0, -1.0, 1.0])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    # robust conversion
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def fmt(v):
    return " ".join(f"{float(x):.8g}" for x in v)


def parse(v, n=None):
    a = np.array([float(x) for x in v.split()]); return a


# ------------------------------------------------------------------ primitives (as in build_m1_candidate.py)
def strip_meshes(root):
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
            ET.SubElement(body, "geom", name="base_box", type="box", size=fmt(BASE_BOX), pos="0 0 -0.02", rgba="0.7 0.7 0.7 1")
        if name.endswith("_FOOT_LINK"):
            right = name.startswith(("FAR", "RAR"))
            ET.SubElement(body, "geom", name=name.replace("_FOOT_LINK", "_wheel"), type="cylinder", size=f"{WHEEL_R} {WHEEL_HALF_W}",
                          quat="0.7071068 0.7071068 0 0", pos="0 -0.0394 0" if right else "0 0.0394 0",
                          friction="0.8 0.02 0.01", condim="6", priority="1", solref="0.05 1")   # tire compliance: 50 ms time constant (chatter-free rolling at 1-2 m/s)


def set_joint_dynamics(root):
    for j in root.iter("joint"):
        j.set("damping", f"{DAMPING}"); j.set("armature", f"{ARMATURE}")


def fix_actuators(root):
    act = root.find("actuator")
    for m in list(act):
        jn = m.get("joint", "")
        kind = jn.split("_")[1]
        m.set("ctrlrange", f"-{CTRLRANGE[kind]:g} {CTRLRANGE[kind]:g}"); m.set("ctrllimited", "true")


def add_sensors_keyframe(root):
    for s in root.findall("sensor"):
        root.remove(s)
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "accelerometer", name="imu_acc", site="imu"); ET.SubElement(sen, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sen, "framequat", name="base_quat", objtype="site", objname="imu"); ET.SubElement(sen, "framelinvel", name="base_linvel", objtype="site", objname="imu")
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    q = [0, 0, 0.42, 1, 0, 0, 0] + [0.0, 0.8, -1.5, 0.0] * 4
    ET.SubElement(kf, "key", name="stand", qpos=" ".join(f"{v:g}" for v in q))


# ------------------------------------------------------------------ mirroring
def mirror_quat(q):
    w, x, y, z = q; return np.array([w, -x, y, -z])


def mirror_element(el: ET.Element, prefix_from: str, prefix_to: str):
    """Mirror one element in place: pos y -> -y, quat -> (w,-x,y,-z), joint ranges of x-axis joints negated+swapped,
    names re-prefixed."""
    if "pos" in el.attrib:
        p = parse(el.get("pos")); p[1] = -p[1]; el.set("pos", fmt(p))
    if "quat" in el.attrib:
        el.set("quat", fmt(mirror_quat(parse(el.get("quat")))))
    if "name" in el.attrib and el.get("name").startswith(prefix_from):
        el.set("name", prefix_to + el.get("name")[len(prefix_from):])
    if el.tag == "joint":
        ax = parse(el.get("axis", "0 0 1"))
        # rotation about an in-plane axis (x or z) reverses sense under the sagittal reflection: negate + swap the range;
        # rotation about y (the mirror normal) keeps its sense.
        if abs(ax[1]) < 1e-9 and "range" in el.attrib:
            lo, hi = parse(el.get("range")); el.set("range", fmt([-hi, -lo]))
    return el


def mirror_subtree(body: ET.Element, prefix_from: str, prefix_to: str) -> ET.Element:
    nb = copy.deepcopy(body)
    for el in nb.iter():
        mirror_element(el, prefix_from, prefix_to)
    return nb


def symmetrize(root):
    base = None
    for b in root.iter("body"):
        if b.get("name") == "base_link":
            base = b; break
    legs = {b.get("name")[:3]: b for b in base.findall("body")}
    # front pair: FBL := mirror(FAR); hind pair: RAR := mirror(RBL) (majority knee mass on all four legs)
    for src_p, dst_p in (("FAR", "FBL"), ("RBL", "RAR")):
        new = mirror_subtree(legs[src_p], src_p, dst_p)
        old = legs[dst_p]; idx = list(base).index(old); base.remove(old); base.insert(idx, new)
    # base inertial: com_y = 0, products (xy, yz) removed
    inr = base.find("inertial")
    pos = parse(inr.get("pos")); pos[1] = 0.0; inr.set("pos", fmt(pos))
    R = quat_to_R(parse(inr.get("quat"))); D = np.diag(parse(inr.get("diaginertia"))); I = R @ D @ R.T
    Is = 0.5 * (I + E @ I @ E)
    w, V = np.linalg.eigh(Is)
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    inr.set("diaginertia", fmt(w)); inr.set("quat", fmt(R_to_quat(V)))
    inr.attrib.pop("fullinertia", None)


def build(with_meshes: str | None):
    tree = ET.parse(SRC); root = tree.getroot()
    for variant in ("m1_wheeled", "m1_wheeled_sym"):
        r = copy.deepcopy(root); r.set("model", variant)
        strip_meshes(r); set_joint_dynamics(r); fix_actuators(r); add_sensors_keyframe(r)
        if variant.endswith("_sym"):
            symmetrize(r)
        r.insert(0, ET.Comment(f" {variant}: derived by scripts/build_m1_wheeled_worlds.py from zgws_source.xml (MATRiX_Python_SDK/model/zgws/zgws.xml, "
                               "GENISOM AI / ZsiBot, BSD-3-Clause; the WHEELED M1). Meshes replaced by primitives; joint damping 0.05, armature 0.01; "
                               "ctrlrange ABAD 40 / HIP,KNEE 60 / WHEEL 20 N m (start values); actuatorfrcrange from the source. "
                               + ("Symmetrized: left legs = mirror of the right templates (front FAR, hind RBL), base com_y 0, base inertia products removed. "
                                  if variant.endswith("_sym") else "Original numbers incl. the chiral details (base com_y 3.4 mm, base products, RAR knee 3.3 g lighter). ")
                               + "Not validated against the real M1 (Day-0 audit pending). "))
        ET.indent(r, space="  ")
        p = OUT / f"{variant}.xml"; ET.ElementTree(r).write(p, encoding="unicode", xml_declaration=False)
        (OUT / f"scene_{variant}.xml").write_text(
            f'<mujoco model="{variant} scene">\n  <option timestep="0.0025"/>\n  <include file="{variant}.xml"/>\n'
            f'  <worldbody>\n    <geom name="floor" size="0 0 0.05" type="plane" friction="0.8 0.02 0.01"/>\n  </worldbody>\n</mujoco>\n')
        print("wrote", p)
    if with_meshes:
        r = copy.deepcopy(root); r.set("model", "m1_wheeled_meshes"); comp = r.find("compiler"); comp.set("meshdir", str(Path(with_meshes).resolve()))
        set_joint_dynamics(r); fix_actuators(r); add_sensors_keyframe(r); ET.indent(r, space="  ")
        p = OUT / "m1_wheeled_meshes.xml"; ET.ElementTree(r).write(p, encoding="unicode", xml_declaration=False); print("wrote", p, "(meshdir outside the repo; do not commit)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--with-meshes", default=None); a = ap.parse_args()
    build(a.with_meshes)


if __name__ == "__main__":
    main()
