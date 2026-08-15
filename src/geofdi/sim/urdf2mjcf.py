"""URDF -> headless MJCF converter for GeoFDI worlds.

Design (kept deliberately simple and inspectable; used for the Unitree go2_description package and later for M1):
- every URDF link becomes an MJCF body placed by its parent joint's <origin>; revolute/continuous -> hinge,
  prismatic -> slide, fixed -> no joint (MuJoCo's `fusestatic` folds the static body into its parent);
- inertials are copied EXPLICITLY from the URDF (`inertiafromgeom="false"`, `fullinertia` in the body frame; an
  inertial rpy is applied as R I R^T) — links without <inertial> are massless (only allowed when welded);
- collision primitives (box / cylinder / sphere / capsule) are copied; visual meshes are NOT emitted (headless);
  collision meshes are skipped with a warning (the Go2 package uses primitives everywhere);
- selected fixed links can be emitted as sites (IMU); foot links get a named foot geom class (S0 conventions);
- actuators: one torque motor per hinge with ctrlrange = +-effort from the URDF <limit>, names = joint name minus
  '_joint'; sensors: accelerometer / gyro / framequat / framelinvel at the IMU site with the S0 names;
- optional symmetrization for a mirror plane y = 0: I <- (I + E I E)/2 for the base (zeros ixy, iyz), com_y <- 0,
  and right-leg collision primitives replaced by the mirror image of the left-leg ones (or vice versa).

The result is validated by compiling it with MuJoCo. Everything numeric comes from the URDF; the only additions
are what a URDF cannot express (joint armature, contact parameters, keyframe, sensors), passed in explicitly.
"""
from __future__ import annotations

import hashlib
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------ small geometry helpers
def rpy_to_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx                     # URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), MuJoCo order."""
    m = R; t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _f(x, nd=9):
    """Compact float formatting (drop trailing zeros; keep enough digits to be exact for URDF-precision inputs)."""
    s = f"{x:.{nd}g}"
    return "0" if s in ("-0", "0.0") else s


def _fmt(v):
    return " ".join(_f(float(x)) for x in v)


def _xyz(el, key="xyz", default=(0.0, 0.0, 0.0)):
    if el is None or el.get(key) is None:
        return np.array(default, dtype=float)
    return np.array([float(t) for t in el.get(key).split()], dtype=float)


# ------------------------------------------------------------------------------------ URDF data model
@dataclass
class Link:
    name: str
    mass: float | None = None
    com: np.ndarray | None = None
    inertia: np.ndarray | None = None          # 3x3, link frame at com
    collisions: list = field(default_factory=list)   # (type, size-tuple, pos, quat)


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    pos: np.ndarray
    rot: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


def parse_urdf(path: str | Path):
    root = ET.parse(str(path)).getroot()
    links, joints = {}, []
    for l in root.findall("link"):
        L = Link(l.get("name"))
        inn = l.find("inertial")
        if inn is not None:
            o = inn.find("origin"); L.mass = float(inn.find("mass").get("value"))
            L.com = _xyz(o); rpy = _xyz(o, "rpy"); Ia = inn.find("inertia").attrib
            I = np.array([[float(Ia["ixx"]), float(Ia["ixy"]), float(Ia["ixz"])],
                          [float(Ia["ixy"]), float(Ia["iyy"]), float(Ia["iyz"])],
                          [float(Ia["ixz"]), float(Ia["iyz"]), float(Ia["izz"])]])
            R = rpy_to_mat(rpy); L.inertia = R @ I @ R.T
        for c in l.findall("collision"):
            o = c.find("origin"); g = c.find("geometry")[0]
            pos = _xyz(o); quat = mat_to_quat(rpy_to_mat(_xyz(o, "rpy")))
            if g.tag == "box":
                L.collisions.append(("box", tuple(float(t) / 2 for t in g.get("size").split()), pos, quat))
            elif g.tag == "cylinder":
                L.collisions.append(("cylinder", (float(g.get("radius")), float(g.get("length")) / 2), pos, quat))
            elif g.tag == "sphere":
                L.collisions.append(("sphere", (float(g.get("radius")),), pos, quat))
            elif g.tag == "capsule":
                L.collisions.append(("capsule", (float(g.get("radius")), float(g.get("length")) / 2), pos, quat))
            else:
                warnings.warn(f"link {L.name}: collision geometry '{g.tag}' skipped (mesh collisions are not converted)")
        links[L.name] = L
    for j in root.findall("joint"):
        o = j.find("origin"); a = j.find("axis"); lim = j.find("limit")
        J = Joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                  _xyz(o), rpy_to_mat(_xyz(o, "rpy")), _xyz(a, default=(1.0, 0.0, 0.0)))
        if lim is not None:
            J.lower = float(lim.get("lower")) if lim.get("lower") is not None else None
            J.upper = float(lim.get("upper")) if lim.get("upper") is not None else None
            J.effort = float(lim.get("effort")) if lim.get("effort") is not None else None
            J.velocity = float(lim.get("velocity")) if lim.get("velocity") is not None else None
        joints.append(J)
    children = {l: [] for l in links}
    for J in joints:
        children[J.parent].append(J)
    roots = [l for l in links if not any(J.child == l for J in joints)]
    assert len(roots) == 1, f"expected one root link, got {roots}"
    return links, joints, children, roots[0]


# ------------------------------------------------------------------------------------ conversion
@dataclass
class ConvertOptions:
    model_name: str = "robot"
    joint_damping: float = 0.01          # URDF/xacro <dynamics damping>; go2 const.xacro: 0.01
    joint_frictionloss: float = 0.2      # xacro <dynamics friction>: 0.2
    joint_armature: float = 0.01         # rotor inertia (not expressible in URDF; menagerie value)
    site_links: tuple = ("imu",)         # fixed links emitted as sites on their parent (name kept)
    drop_links: tuple = ("radar",)       # fixed links dropped entirely
    foot_link_suffix: str = "_foot"      # links whose collision sphere becomes the named foot geom (name = prefix)
    foot_solimp: tuple = (0.015, 1.0, 0.022)
    foot_friction: tuple = (0.8, 0.02, 0.01)
    foot_condim: int = 6
    geom_friction: float = 0.6
    geom_margin: float = 0.001
    keyframe_qpos: tuple | None = None   # full qpos incl. free joint (7 + n)
    keyframe_ctrl: tuple | None = None
    base_pos: tuple = (0.0, 0.0, 0.445)
    imu_sensors: bool = True
    # symmetrization about the plane y = 0
    sym_base_inertia: bool = False       # I_base <- (I + E I E)/2, com_y <- 0
    sym_leg_collisions: str | None = None   # None | 'right' (right legs = mirror of left) | 'left' (left = mirror of right)
    leg_pairs: tuple = (("FL", "FR"), ("RL", "RR"))


def _mirror_geom(col):
    """Mirror a collision primitive about y = 0: pos_y -> -pos_y; quaternion (w,x,y,z) -> (w,-x,y,-z)."""
    typ, size, pos, quat = col
    return (typ, size, pos * np.array([1, -1, 1]), quat * np.array([1, -1, 1, -1]))


def _sym_inertia(I):
    E = np.diag([1.0, -1.0, 1.0]); return 0.5 * (I + E @ I @ E)


def convert(urdf_path: str | Path, opts: ConvertOptions) -> str:
    links, joints, children, root = parse_urdf(urdf_path)
    if opts.sym_base_inertia:
        L = links[root]; L.inertia = _sym_inertia(L.inertia); L.com = L.com * np.array([1, 0, 1])
    if opts.sym_leg_collisions:
        for left, right in opts.leg_pairs:
            for name in list(links):
                if name.startswith(left + "_"):
                    other = right + name[len(left):]
                    if other not in links:
                        continue
                    if opts.sym_leg_collisions == "right":
                        links[other].collisions = [_mirror_geom(c) for c in links[name].collisions]
                    else:
                        links[name].collisions = [_mirror_geom(c) for c in links[other].collisions]

    mj = ET.Element("mujoco", model=opts.model_name)
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", inertiafromgeom="false", fusestatic="true")
    ET.SubElement(mj, "option", cone="elliptic", impratio="100")
    dflt = ET.SubElement(mj, "default")
    cls = ET.SubElement(dflt, "default", **{"class": opts.model_name})
    ET.SubElement(cls, "geom", friction=_f(opts.geom_friction), margin=_f(opts.geom_margin), condim="1", group="3")
    ET.SubElement(cls, "joint", damping=_f(opts.joint_damping), armature=_f(opts.joint_armature), frictionloss=_f(opts.joint_frictionloss))
    foot = ET.SubElement(cls, "default", **{"class": "foot"})
    ET.SubElement(foot, "geom", priority="1", solimp=_fmt(opts.foot_solimp), condim=str(opts.foot_condim), friction=_fmt(opts.foot_friction))
    ET.SubElement(mj, "asset")
    wb = ET.SubElement(mj, "worldbody")
    hinge_joints = []      # (Joint) in URDF order

    def emit(link_name, parent_el, pos, rot, is_root=False):
        L = links[link_name]
        attrs = {"name": link_name}
        if not is_root:
            attrs["pos"] = _fmt(pos)
            q = mat_to_quat(rot)
            if not np.allclose(q, [1, 0, 0, 0], atol=1e-12):
                attrs["quat"] = _fmt(q)
        else:
            attrs["pos"] = _fmt(opts.base_pos); attrs["childclass"] = opts.model_name
        b = ET.SubElement(parent_el, "body", **attrs)
        if L.mass is not None:
            I = L.inertia
            ET.SubElement(b, "inertial", pos=_fmt(L.com), mass=_f(L.mass),
                          fullinertia=_fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        if is_root:
            ET.SubElement(b, "freejoint")
        # geoms
        for k, (typ, size, gpos, gquat) in enumerate(L.collisions):
            gattrs = {"type": typ, "size": _fmt(size), "pos": _fmt(gpos)}
            if not np.allclose(gquat, [1, 0, 0, 0], atol=1e-12):
                gattrs["quat"] = _fmt(gquat)
            if link_name.endswith(opts.foot_link_suffix) and typ == "sphere":
                gattrs["name"] = link_name[: -len(opts.foot_link_suffix)]; gattrs["class"] = "foot"
            ET.SubElement(b, "geom", **gattrs)
        # children
        for J in children[link_name]:
            if J.child in opts.drop_links:
                continue
            if J.child in opts.site_links:
                sattrs = {"name": J.child, "pos": _fmt(J.pos)}
                q = mat_to_quat(J.rot)
                if not np.allclose(q, [1, 0, 0, 0], atol=1e-12):
                    sattrs["quat"] = _fmt(q)
                ET.SubElement(b, "site", **sattrs)
                continue
            if J.jtype in ("revolute", "continuous") :
                hinge_joints.append(J)                # PRE-order: actuator order == URDF joint order (hip, thigh, calf)
            cb = emit(J.child, b, J.pos, J.rot)
            if J.jtype in ("revolute", "continuous", "prismatic"):
                jattrs = {"name": J.name, "axis": _fmt(J.axis)}
                if J.jtype == "prismatic":
                    jattrs["type"] = "slide"
                if J.jtype != "continuous" and J.lower is not None:
                    jattrs["range"] = _fmt([J.lower, J.upper])
                jel = ET.Element("joint", **jattrs)
                idx = 1 if (cb.find("inertial") is not None) else 0
                cb.insert(idx, jel)
        return b

    emit(root, wb, np.zeros(3), np.eye(3), is_root=True)
    act = ET.SubElement(mj, "actuator")
    for J in hinge_joints:
        aattrs = {"name": J.name[:-6] if J.name.endswith("_joint") else J.name, "joint": J.name}
        if J.effort is not None:
            aattrs["ctrlrange"] = _fmt([-J.effort, J.effort]); aattrs["ctrllimited"] = "true"
        ET.SubElement(act, "motor", **aattrs)
    if opts.imu_sensors:
        sen = ET.SubElement(mj, "sensor")
        ET.SubElement(sen, "accelerometer", name="imu_acc", site="imu")
        ET.SubElement(sen, "gyro", name="imu_gyro", site="imu")
        ET.SubElement(sen, "framequat", name="base_quat", objtype="site", objname="imu")
        ET.SubElement(sen, "framelinvel", name="base_linvel", objtype="site", objname="imu")
    if opts.keyframe_qpos is not None:
        kf = ET.SubElement(mj, "keyframe")
        kattrs = {"name": "home", "qpos": _fmt(opts.keyframe_qpos)}
        if opts.keyframe_ctrl is not None:
            kattrs["ctrl"] = _fmt(opts.keyframe_ctrl)
        ET.SubElement(kf, "key", **kattrs)
    ET.indent(mj, space="  ")
    return ET.tostring(mj, encoding="unicode")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
