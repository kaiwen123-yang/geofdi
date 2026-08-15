#!/usr/bin/env python3
"""bag_inventory.py — topic × type × frequency inventory for rosbag2 (sqlite3) recordings.

Usage:
    bag_inventory.py [--sample-rows N] [--decode] [--markdown OUT.md] [--json OUT.json] BAGDIR [BAGDIR ...]

For every bag directory (metadata.yaml + *.db3):
  * parses metadata.yaml (rosbag2 metadata v4–v7 layouts) — topics, types, counts, duration;
  * opens each .db3 read-only and cross-checks: schema/ros_distro, topics table vs metadata,
    per-file message counts and time ranges vs metadata (this is the consistency check that
    fix_metadata.py repairs when it fails);
  * measures the actual publish rate of every topic from the first --sample-rows messages of
    the first file (median dt, jitter) — metadata only gives count/duration;
  * with --decode (needs the `rosbags` package): decodes one sample of each standard-typed
    topic (JointState names/count, Imu frame, Odometry frames, TF frames); custom types
    (e.g. robots_dog_msgs/*) are listed but not decoded.
Writes a Markdown report (per-bag tables + a "key determination" block) and/or JSON.
Pure Python (sqlite3 + PyYAML); no ROS installation needed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import yaml

JOINT_TYPE = "sensor_msgs/msg/JointState"


def _num_suffix(p: Path) -> int:
    m = re.search(r"_(\d+)\.db3$", p.name)
    return int(m.group(1)) if m else -1


def read_metadata(bag: Path) -> dict | None:
    f = bag / "metadata.yaml"
    if not f.exists():
        return None
    with open(f) as fh:
        y = yaml.safe_load(fh)
    return y.get("rosbag2_bagfile_information", y)


def db_files(bag: Path) -> list[Path]:
    return sorted(bag.glob("*.db3"), key=_num_suffix)


def sqlite_summary(db: Path, sample_rows: int) -> dict:
    """Index-only totals + a time-ordered sample of (topic_id, timestamp) rows."""
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    cur = con.cursor()
    out: dict = {"file": db.name, "size_bytes": db.stat().st_size}
    try:
        out["schema"] = cur.execute("SELECT schema_version, ros_distro FROM schema").fetchall()
    except sqlite3.Error:
        out["schema"] = None
    out["topics"] = [
        {"id": r[0], "name": r[1], "type": r[2], "serialization": r[3], "qos_empty": (r[4] or "") == ""}
        for r in cur.execute("SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics")
    ]
    n, tmin, tmax = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
    out.update(message_count=n, t_min=tmin, t_max=tmax)
    if sample_rows > 0:
        rows = cur.execute(
            "SELECT topic_id, timestamp FROM messages ORDER BY timestamp LIMIT ?", (sample_rows,)
        ).fetchall()
        per: dict[int, list[int]] = {}
        for tid, ts in rows:
            per.setdefault(tid, []).append(ts)
        stats = {}
        for tid, tss in per.items():
            if len(tss) < 3:
                stats[tid] = {"n": len(tss)}
                continue
            dts = [(b - a) / 1e6 for a, b in zip(tss[:-1], tss[1:])]  # ms
            dts_sorted = sorted(dts)
            med = dts_sorted[len(dts_sorted) // 2]
            mean = sum(dts) / len(dts)
            var = sum((d - mean) ** 2 for d in dts) / len(dts)
            stats[tid] = {"n": len(tss), "median_dt_ms": med, "jitter_ms": var ** 0.5,
                          "rate_hz": (1000.0 / med) if med > 0 else None,
                          "span_s": (tss[-1] - tss[0]) / 1e9}
        out["sample"] = stats
    con.close()
    return out


def decode_samples(bag: Path, topics: list[str]) -> dict:
    """Decode one message per (standard-typed) topic with rosbags; returns {topic: description}."""
    try:
        from rosbags.rosbag2 import Reader  # type: ignore
        from rosbags.typesys import Stores, get_typestore  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"_error": f"rosbags not importable ({e}); pip install rosbags"}
    ts = get_typestore(Stores.ROS2_HUMBLE)
    res: dict = {}
    with Reader(bag) as r:
        conns = {c.topic: c for c in r.connections}
        for name in topics:
            c = conns.get(name)
            if c is None:
                continue
            if c.msgtype not in ts.types:
                res[name] = f"type {c.msgtype} not decodable without its message package"
                continue
            try:
                for conn, t, raw in r.messages(connections=[c]):
                    m = ts.deserialize_cdr(raw, conn.msgtype)
                    res[name] = describe(m, conn.msgtype)
                    break
                else:
                    res[name] = "no messages"
            except Exception as e:  # pragma: no cover
                res[name] = f"decode error: {type(e).__name__}: {e}"
    return res


def describe(m, msgtype: str) -> str:
    if msgtype == JOINT_TYPE:
        names = list(m.name)
        pos = [round(float(x), 3) for x in m.position]
        eff = [round(float(x), 2) for x in m.effort]
        return (f"{len(names)} joints: {names}; position sample {pos}; effort sample {eff}; "
                f"velocity/effort arrays present: {len(m.velocity)}/{len(m.effort)}")
    if msgtype == "sensor_msgs/msg/Imu":
        o = m.orientation
        return (f"frame_id='{m.header.frame_id}'; orientation sample ({o.x:.3f},{o.y:.3f},{o.z:.3f},{o.w:.3f}); "
                f"gyro ({m.angular_velocity.x:.4f},{m.angular_velocity.y:.4f},{m.angular_velocity.z:.4f}); "
                f"accel ({m.linear_acceleration.x:.3f},{m.linear_acceleration.y:.3f},{m.linear_acceleration.z:.3f})")
    if msgtype == "nav_msgs/msg/Odometry":
        p = m.pose.pose.position
        return f"frame '{m.header.frame_id}' -> child '{m.child_frame_id}'; position sample ({p.x:.2f},{p.y:.2f},{p.z:.2f})"
    if msgtype == "tf2_msgs/msg/TFMessage":
        return "transforms: " + ", ".join(f"{t.header.frame_id}->{t.child_frame_id}" for t in m.transforms)
    if msgtype == "sensor_msgs/msg/BatteryState":
        return f"voltage {m.voltage}, percentage {m.percentage}, current {m.current}"
    if msgtype == "sensor_msgs/msg/PointCloud2":
        return f"frame_id='{m.header.frame_id}', {m.width}x{m.height} points, {len(m.fields)} fields"
    if msgtype == "geometry_msgs/msg/Twist":
        return f"linear ({m.linear.x:.3f},{m.linear.y:.3f},{m.linear.z:.3f}) angular ({m.angular.x:.3f},{m.angular.y:.3f},{m.angular.z:.3f})"
    return f"decoded {msgtype}"


def inventory(bag: Path, sample_rows: int, decode: bool) -> dict:
    meta = read_metadata(bag)
    files = db_files(bag)
    rep: dict = {"bag": str(bag), "name": bag.name, "metadata_present": meta is not None,
                 "n_db3": len(files), "size_bytes": sum(f.stat().st_size for f in files),
                 "issues": [], "topics": []}
    if meta is None:
        rep["issues"].append("metadata.yaml missing — run fix_metadata.py")
    else:
        rep.update(version=meta.get("version"), storage=meta.get("storage_identifier"),
                   duration_s=meta.get("duration", {}).get("nanoseconds", 0) / 1e9,
                   start_ns=meta.get("starting_time", {}).get("nanoseconds_since_epoch"),
                   message_count=meta.get("message_count"),
                   ros_distro=meta.get("ros_distro"))
        listed = meta.get("relative_file_paths", [])
        present = [f.name for f in files]
        if sorted(listed) != sorted(present):
            rep["issues"].append(f"relative_file_paths ({len(listed)}) != files present ({len(present)})")
    # sqlite side
    per_file = []
    for i, f in enumerate(files):
        per_file.append(sqlite_summary(f, sample_rows if i == 0 else 0))
    rep["files"] = [{k: v for k, v in pf.items() if k != "sample" and k != "topics"} for pf in per_file]
    if per_file:
        rep["schema"] = per_file[0]["schema"]
        db_topics = {t["name"]: t for t in per_file[0]["topics"]}
    else:
        db_topics = {}
    # cross-checks against metadata
    if meta is not None:
        mfiles = {f["path"]: f for f in meta.get("files", [])}
        for pf in per_file:
            mf = mfiles.get(pf["file"])
            if mf is None:
                rep["issues"].append(f"{pf['file']}: not in metadata files[]")
                continue
            if mf.get("message_count") != pf["message_count"]:
                rep["issues"].append(f"{pf['file']}: message_count db={pf['message_count']} meta={mf.get('message_count')}")
            ms = mf.get("starting_time", {}).get("nanoseconds_since_epoch")
            if ms is not None and pf["t_min"] is not None and abs(ms - pf["t_min"]) > 1_000_000:
                rep["issues"].append(f"{pf['file']}: start db={pf['t_min']} meta={ms}")
        total_db = sum(pf["message_count"] for pf in per_file)
        if meta.get("message_count") not in (None, total_db):
            rep["issues"].append(f"total message_count db={total_db} meta={meta.get('message_count')}")
        dur = rep.get("duration_s") or 0.0
        for t in meta.get("topics_with_message_count", []):
            tm = t.get("topic_metadata", {})
            name, typ, cnt = tm.get("name"), tm.get("type"), t.get("message_count", 0)
            row = {"topic": name, "type": typ, "count": cnt,
                   "nominal_hz": (cnt / dur) if dur > 0 and cnt else 0.0,
                   "serialization": tm.get("serialization_format"),
                   "qos_empty": (tm.get("offered_qos_profiles") or "") == ""}
            dbt = db_topics.get(name)
            if dbt is None:
                rep["issues"].append(f"{name}: in metadata but not in db topics table")
            else:
                if dbt["type"] != typ:
                    rep["issues"].append(f"{name}: type db='{dbt['type']}' meta='{typ}'")
                if "::" in (dbt["type"] or "") or "::" in (typ or ""):
                    rep["issues"].append(f"{name}: DDS-style type name '{dbt['type'] or typ}' — normalize with fix_metadata.py")
                st = per_file[0].get("sample", {}).get(dbt["id"]) if per_file else None
                if st:
                    row.update(measured_hz=st.get("rate_hz"), median_dt_ms=st.get("median_dt_ms"),
                               jitter_ms=st.get("jitter_ms"), sample_n=st.get("n"))
            rep["topics"].append(row)
        for name in db_topics:
            if name not in {t["topic"] for t in rep["topics"]}:
                rep["issues"].append(f"{name}: in db topics table but not in metadata")
    else:
        for name, dbt in db_topics.items():
            rep["topics"].append({"topic": name, "type": dbt["type"], "count": None, "nominal_hz": None})
    rep["topics"].sort(key=lambda r: r["topic"])
    if decode and files:
        rep["decoded"] = decode_samples(bag, [t["topic"] for t in rep["topics"] if (t.get("count") or 0) > 0])
    js = [t for t in rep["topics"] if t["type"] == JOINT_TYPE]
    rep["joint_state_topics"] = [{"topic": t["topic"], "count": t["count"], "nominal_hz": t.get("nominal_hz"),
                                  "measured_hz": t.get("measured_hz")} for t in js]
    return rep


def fmt_hz(x):
    return "" if x is None else f"{x:.1f}"


def to_markdown(reps: list[dict], sample_rows: int) -> str:
    L = []
    L.append(f"_Generated by `scripts/m1_bag_tools/bag_inventory.py` on {_dt.date.today().isoformat()}; "
             f"measured rates from the first {sample_rows} time-ordered rows of file 0 of each bag._\n")
    for r in reps:
        L.append(f"## {r['name']}\n")
        L.append(f"- path: `{r['bag']}`")
        if r["metadata_present"]:
            start = _dt.datetime.fromtimestamp(r["start_ns"] / 1e9).isoformat(sep=" ", timespec="seconds") if r.get("start_ns") else "?"
            L.append(f"- metadata v{r.get('version')}, storage `{r.get('storage')}`, schema {r.get('schema')}; "
                     f"{r['n_db3']} db3 files, {r['size_bytes']/2**30:.1f} GiB, start {start} (local), "
                     f"duration {r.get('duration_s', 0):.1f} s, {r.get('message_count')} messages")
        else:
            L.append(f"- **metadata.yaml missing**; {r['n_db3']} db3 files, {r['size_bytes']/2**30:.1f} GiB")
        L.append(f"- consistency check (metadata vs sqlite): " + ("**OK — no repair needed**" if not r["issues"] else "; ".join(r["issues"])))
        L.append("")
        L.append("| topic | type | count | nominal Hz (count/duration) | measured Hz (median dt) | jitter ms | notes |")
        L.append("|---|---|---:|---:|---:|---:|---|")
        dec = r.get("decoded", {})
        for t in r["topics"]:
            note = ""
            if t["type"] == JOINT_TYPE:
                note = "**joint states**"
            if t["type"].startswith("robots_dog_msgs"):
                note = (note + " " if note else "") + "custom type (no msg pkg here)"
            if (t.get("count") or 0) == 0:
                note = (note + " " if note else "") + "empty"
            d = dec.get(t["topic"])
            if d and t["type"] not in (JOINT_TYPE,):
                note = (note + " " if note else "") + d
            jit = "" if t.get("jitter_ms") is None else f"{t['jitter_ms']:.2f}"
            cnt = "" if t.get("count") is None else str(t["count"])
            L.append(f"| `{t['topic']}` | `{t['type']}` | {cnt} | "
                     f"{fmt_hz(t.get('nominal_hz'))} | {fmt_hz(t.get('measured_hz'))} | {jit} | {note} |")
        L.append("")
        for t in r["topics"]:
            if t["type"] == JOINT_TYPE and dec.get(t["topic"]):
                L.append(f"- `{t['topic']}` sample: {dec[t['topic']]}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", type=Path)
    ap.add_argument("--sample-rows", type=int, default=60000, help="time-ordered rows of file 0 used for measured rates")
    ap.add_argument("--decode", action="store_true", help="decode one sample per standard-typed topic (needs rosbags)")
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    reps = []
    for b in a.bags:
        if not b.is_dir():
            print(f"skip {b}: not a directory", file=sys.stderr)
            continue
        print(f"inventory: {b}", file=sys.stderr)
        reps.append(inventory(b, a.sample_rows, a.decode))
    md = to_markdown(reps, a.sample_rows)
    if a.markdown:
        a.markdown.write_text(md)
        print(f"wrote {a.markdown}", file=sys.stderr)
    else:
        print(md)
    if a.json:
        a.json.write_text(json.dumps(reps, indent=1, default=str))
        print(f"wrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
