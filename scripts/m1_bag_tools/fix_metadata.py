#!/usr/bin/env python3
"""fix_metadata.py — check / regenerate rosbag2 (sqlite3) metadata.yaml from the .db3 files.

Why this exists (the "zenoh metadata" repair flow): recordings made on the M1 through
rmw_zenoh / the zenoh-DDS bridge, or cut short by a crash / power loss, can leave a bag with
  (a) no metadata.yaml at all (recorder never closed the bag),
  (b) a metadata.yaml that lists files / message counts / durations that do not match the
      .db3 files actually present (partial last file, files copied selectively),
  (c) DDS-style type names such as `sensor_msgs::msg::dds_::Imu_` instead of
      `sensor_msgs/msg/Imu` (bridge-recorded topics), which stock tooling cannot map, or
  (d) empty `offered_qos_profiles` strings, which make `ros2 bag play` reject the topic.
`ros2 bag reindex` fixes (a)/(b) only when a ROS installation with matching plugins is
available; this script needs only Python + sqlite3 + PyYAML and never touches raw bags unless
--in-place is given (raw data is immutable — write the repaired copy elsewhere).

Usage:
    fix_metadata.py --check BAGDIR                 # report only (exit 0 = consistent, 1 = repair needed)
    fix_metadata.py --out DIR BAGDIR               # write DIR/metadata.yaml (repaired copy)
    fix_metadata.py --in-place BAGDIR              # overwrite BAGDIR/metadata.yaml (keeps metadata.yaml.bak)
    add --normalize-db-types together with --in-place to also rewrite DDS-style names in the
    sqlite `topics` table (needed for (c) if the reader consults the db instead of the yaml).

Regeneration reads, per file: MIN/MAX timestamp and COUNT(*) (index-only, fast) and the
per-topic counts (full table scan — minutes on multi-GB bags). The output follows the
rosbag2 metadata **version 5** layout used by Humble (what these M1 bags were recorded with).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import yaml

DEFAULT_QOS = ("- history: 1\n  depth: 10\n  reliability: 1\n  durability: 2\n  deadline:\n    sec: 9223372036\n"
               "    nsec: 854775807\n  lifespan:\n    sec: 9223372036\n    nsec: 854775807\n  liveliness: 1\n"
               "  liveliness_lease_duration:\n    sec: 9223372036\n    nsec: 854775807\n"
               "  avoid_ros_namespace_conventions: false")

DDS_RE = re.compile(r"^(?P<pkg>[A-Za-z0-9_]+)::(?P<ns>msg|srv|action)::dds_::(?P<name>[A-Za-z0-9_]+)_$")


def normalize_type(t: str) -> str:
    """`pkg::msg::dds_::Name_` -> `pkg/msg/Name`; `pkg/Name` -> `pkg/msg/Name`; else unchanged."""
    m = DDS_RE.match(t or "")
    if m:
        return f"{m.group('pkg')}/{m.group('ns')}/{m.group('name')}"
    parts = (t or "").split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return t


def _num_suffix(p: Path) -> int:
    m = re.search(r"_(\d+)\.db3$", p.name)
    return int(m.group(1)) if m else -1


def scan(bag: Path, per_topic: bool) -> dict:
    files = sorted(bag.glob("*.db3"), key=_num_suffix)
    if not files:
        raise SystemExit(f"no .db3 files in {bag}")
    topics: dict[str, dict] = {}
    finfo = []
    for f in files:
        con = sqlite3.connect(f"file:{f}?mode=ro&immutable=1", uri=True)
        cur = con.cursor()
        tmap = {}
        for tid, name, typ, ser, qos in cur.execute("SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics"):
            tmap[tid] = name
            t = topics.setdefault(name, {"name": name, "type": normalize_type(typ), "raw_type": typ,
                                         "serialization_format": ser or "cdr",
                                         "offered_qos_profiles": qos if qos else DEFAULT_QOS, "count": 0,
                                         "qos_was_empty": not qos, "type_was_dds": normalize_type(typ) != typ})
        n, tmin, tmax = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
        finfo.append({"path": f.name, "count": n, "t_min": tmin, "t_max": tmax})
        if per_topic:
            for tid, c in cur.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"):
                topics[tmap[tid]]["count"] += c
        con.close()
    starts = [fi["t_min"] for fi in finfo if fi["t_min"] is not None]
    ends = [fi["t_max"] for fi in finfo if fi["t_max"] is not None]
    t0 = min(starts) if starts else 0
    t1 = max(ends) if ends else 0
    return {"files": finfo, "topics": topics, "t0": t0, "t1": t1, "count": sum(fi["count"] for fi in finfo)}


def build_metadata(s: dict) -> dict:
    return {"rosbag2_bagfile_information": {
        "version": 5,
        "storage_identifier": "sqlite3",
        "duration": {"nanoseconds": int(s["t1"] - s["t0"])},
        "starting_time": {"nanoseconds_since_epoch": int(s["t0"])},
        "message_count": int(s["count"]),
        "topics_with_message_count": [
            {"topic_metadata": {"name": t["name"], "type": t["type"], "serialization_format": t["serialization_format"],
                                "offered_qos_profiles": t["offered_qos_profiles"]},
             "message_count": int(t["count"])} for t in s["topics"].values()],
        "compression_format": "", "compression_mode": "",
        "relative_file_paths": [fi["path"] for fi in s["files"]],
        "files": [{"path": fi["path"], "starting_time": {"nanoseconds_since_epoch": int(fi["t_min"] or 0)},
                   "duration": {"nanoseconds": int((fi["t_max"] or 0) - (fi["t_min"] or 0))},
                   "message_count": int(fi["count"])} for fi in s["files"]],
    }}


def check(bag: Path, s: dict) -> list[str]:
    issues = []
    mf = bag / "metadata.yaml"
    if not mf.exists():
        return ["metadata.yaml missing"]
    meta = yaml.safe_load(mf.read_text()).get("rosbag2_bagfile_information", {})
    listed = meta.get("relative_file_paths", [])
    present = [fi["path"] for fi in s["files"]]
    if sorted(listed) != sorted(present):
        issues.append(f"relative_file_paths {listed} != present {present}")
    mfiles = {f["path"]: f for f in meta.get("files", [])}
    for fi in s["files"]:
        m = mfiles.get(fi["path"])
        if m is None:
            issues.append(f"{fi['path']}: missing from files[]")
        elif m.get("message_count") != fi["count"]:
            issues.append(f"{fi['path']}: message_count meta={m.get('message_count')} db={fi['count']}")
    if meta.get("message_count") != s["count"]:
        issues.append(f"message_count meta={meta.get('message_count')} db={s['count']}")
    if abs(meta.get("starting_time", {}).get("nanoseconds_since_epoch", 0) - s["t0"]) > 1_000_000:
        issues.append("starting_time mismatch")
    if abs(meta.get("duration", {}).get("nanoseconds", 0) - (s["t1"] - s["t0"])) > 1_000_000:
        issues.append("duration mismatch")
    mt = {t["topic_metadata"]["name"]: t["topic_metadata"] for t in meta.get("topics_with_message_count", [])}
    for name, t in s["topics"].items():
        if name not in mt:
            issues.append(f"{name}: in db but not in metadata")
            continue
        if normalize_type(mt[name].get("type")) != t["type"]:
            issues.append(f"{name}: type meta='{mt[name].get('type')}' db='{t['raw_type']}'")
        if "::" in (mt[name].get("type") or ""):
            issues.append(f"{name}: DDS-style type name in metadata")
        if not mt[name].get("offered_qos_profiles"):
            issues.append(f"{name}: empty offered_qos_profiles")
    for name in mt:
        if name not in s["topics"]:
            issues.append(f"{name}: in metadata but not in db")
    for name, t in s["topics"].items():
        if t["type_was_dds"]:
            issues.append(f"{name}: DDS-style type name in sqlite topics table")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--out", type=Path)
    g.add_argument("--in-place", action="store_true")
    ap.add_argument("--normalize-db-types", action="store_true")
    a = ap.parse_args()
    if a.check:
        s = scan(a.bag, per_topic=False)
        issues = check(a.bag, s)
        if issues:
            print("REPAIR NEEDED:\n  - " + "\n  - ".join(issues))
            sys.exit(1)
        print(f"OK: {a.bag} — metadata.yaml consistent with {len(s['files'])} db3 files "
              f"({s['count']} messages, {(s['t1']-s['t0'])/1e9:.1f} s); no DDS-style type names; no empty QoS.")
        return
    s = scan(a.bag, per_topic=True)
    meta = build_metadata(s)
    text = yaml.safe_dump(meta, sort_keys=False, default_style=None, width=10**6)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "metadata.yaml").write_text(text)
        print(f"wrote {a.out/'metadata.yaml'}")
    else:
        mf = a.bag / "metadata.yaml"
        if mf.exists():
            shutil.copy2(mf, mf.with_suffix(".yaml.bak"))
        mf.write_text(text)
        print(f"rewrote {mf} (backup: metadata.yaml.bak)")
        if a.normalize_db_types:
            for f in sorted(a.bag.glob("*.db3")):
                con = sqlite3.connect(f)
                for tid, typ in con.execute("SELECT id, type FROM topics").fetchall():
                    nt = normalize_type(typ)
                    if nt != typ:
                        con.execute("UPDATE topics SET type=? WHERE id=?", (nt, tid))
                con.commit(); con.close()
            print("normalized DDS-style type names in sqlite topics tables")
    changed = [t["name"] for t in s["topics"].values() if t["type_was_dds"] or t["qos_was_empty"]]
    if changed:
        print("topics with normalized type / default QoS:", ", ".join(changed))


if __name__ == "__main__":
    main()
