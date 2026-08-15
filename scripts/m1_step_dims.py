#!/usr/bin/env python3
"""M1 point-foot STEP (Creo AP203, open shells, no mass/joints) — dimension check against the MATRiX `zgws` (wheeled M1)
MJCF. NOT a reverse-engineering: bounding box + cylindrical-face axis clustering to locate the joint axes (hip roll: axes
parallel to x; hip pitch / knee: axes parallel to y), from which link lengths and offsets are read.

    python scripts/m1_step_dims.py <step file> [--out results/m1_step_dims.json]

Falls back to the bounding box only if the STEP cannot be parsed or has too few cylindrical faces.
Units: the STEP is assumed in mm (Creo default) — detected from the bounding box magnitude and reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REF = {  # MATRiX zgws (wheeled M1) MJCF, metres — see docs/protocol/m1_model_audit.md
    "hip_x_half_spacing": 0.2698, "abad_y_half_spacing": 0.065, "abad_to_hip_x": 0.0587, "abad_to_hip_y": 0.045,
    "hip_axis_half_spacing_y": 0.110, "thigh_length": 0.26, "thigh_knee_y_offset": 0.0522, "calf_length": 0.28,
    "base_mesh_extents": [0.8333, 0.2302, 0.2082], "wheel_radius": 0.096,
}


def cluster_1d(vals, tol):
    vals = np.sort(np.asarray(vals)); groups = []; cur = [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    return [(float(np.mean(g)), len(g)) for g in groups]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--out", default=None); ap.add_argument("--min-radius", type=float, default=0.012)
    args = ap.parse_args()
    out = {"file": args.step, "reference_zgws_m": REF}
    t0 = time.time()
    try:
        import cadquery as cq
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
        shape = cq.importers.importStep(args.step)
        bb = shape.val().BoundingBox()
        ext = np.array([bb.xlen, bb.ylen, bb.zlen]); scale = 1e-3 if ext.max() > 20 else 1.0
        out["units_assumed"] = "mm" if scale == 1e-3 else "m"; out["bbox_m"] = (ext * scale).tolist()
        out["bbox_min_m"] = [bb.xmin * scale, bb.ymin * scale, bb.zmin * scale]; out["bbox_max_m"] = [bb.xmax * scale, bb.ymax * scale, bb.zmax * scale]
        cyl = []
        for solid in shape.vals():
            exp = TopExp_Explorer(solid.wrapped, TopAbs_FACE)
            while exp.More():
                f = TopoDS.Face_s(exp.Current()); ad = BRepAdaptor_Surface(f)
                if ad.GetType() == GeomAbs_Cylinder:
                    c = ad.Cylinder(); ax = c.Axis(); d = ax.Direction(); loc = ax.Location()
                    cyl.append((c.Radius() * scale, np.array([d.X(), d.Y(), d.Z()]), np.array([loc.X(), loc.Y(), loc.Z()]) * scale))
                exp.Next()
        out["n_cylindrical_faces"] = len(cyl); out["parse_seconds"] = time.time() - t0
        big = [c for c in cyl if c[0] >= args.min_radius]
        # axes parallel to x (hip roll) and y (pitch joints); a cylinder axis line is (loc, dir): its position is the
        # projection of loc onto the plane orthogonal to dir
        def axis_key(dirv, loc):
            return tuple(np.round(loc - dirv * (loc @ dirv), 4))
        groups = {"x": defaultdict(list), "y": defaultdict(list), "z": defaultdict(list)}
        for r, d, loc in big:
            for name, e in (("x", np.array([1, 0, 0.])), ("y", np.array([0, 1, 0.])), ("z", np.array([0, 0, 1.]))):
                if abs(abs(d @ e) - 1) < 1e-3:
                    groups[name][axis_key(e, loc)].append(r)
        summ = {}
        for name in ("x", "y", "z"):
            axes = [(k, len(v), float(np.max(v))) for k, v in groups[name].items()]
            axes.sort(key=lambda t: -t[1])
            summ[name] = [{"axis_point_m": list(k), "n_faces": n, "max_radius_m": rmax} for k, n, rmax in axes[:24]]
        out["axes"] = summ
        # derived candidates: y-axes (pitch joints) grouped by z level -> hip pitch and knee levels; x-axes (roll) -> hip roll lines
        y_axes = [(np.array(a["axis_point_m"]), a["n_faces"], a["max_radius_m"]) for a in summ["y"]]
        if y_axes:
            zs = cluster_1d([p[2] for p, _, _ in y_axes], 0.02); xs = cluster_1d([p[0] for p, _, _ in y_axes], 0.02)
            out["y_axis_z_levels_m"] = zs; out["y_axis_x_levels_m"] = xs
            if len(zs) >= 2:
                zs_sorted = sorted([z for z, n in zs], reverse=True)
                out["derived"] = {"pitch_axis_z_gaps_m": [float(zs_sorted[i] - zs_sorted[i + 1]) for i in range(len(zs_sorted) - 1)]}
        x_axes = [(np.array(a["axis_point_m"]), a["n_faces"], a["max_radius_m"]) for a in summ["x"]]
        if x_axes:
            out["x_axis_y_levels_m"] = cluster_1d([p[1] for p, _, _ in x_axes], 0.01); out["x_axis_z_levels_m"] = cluster_1d([p[2] for p, _, _ in x_axes], 0.01)
    except Exception as e:                                # noqa: BLE001
        out["error"] = repr(e); out["parse_seconds"] = time.time() - t0
    js = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(js)
    print(js[:6000])


if __name__ == "__main__":
    main()
