"""M1 rosbag2 (sqlite3) recordings -> GeoFDI SDK-layout session (Sprint 8 Block D).

The vendor stack records `/joint_shm_controller/joint_states` (sensor_msgs/JointState, 16 joints, 200 Hz),
`/imu_driver/imu_central` (sensor_msgs/Imu, 200 Hz), `/odom/mc_odom` (nav_msgs/Odometry, ~40 Hz, vendor motion-controller
odometry) and optionally `/cmd_vel`. This module reads the `.db3` files directly through sqlite3 and decodes CDR with the
pure-python `rosbags` typestore — no ROS environment, no rosbag2 `Reader` (so a bag whose recording was aborted, leaving
0-byte `.db3` files and an inconsistent `metadata.yaml`, still reads: empty/unreadable files are skipped and reported).

    extract_bag_session(bag_dir, out_dir)   # -> joint_states.csv, imu.csv, imu_front.csv, imu_rear.csv, odom.csv,
                                            #    cmd.csv (only if messages exist), meta.yaml, extract_report.json
    is_rosbag2(session_dir)                 # metadata.yaml + *.db3 present

Time base: `t` in every CSV is seconds since the first joint-state HEADER stamp (`t0_epoch_ns` in meta.yaml); the
receive-time (bag) stamps are kept in `t_bag` for the audit (header−bag skew). Values are written exactly as reported
(positions rad, velocities rad/s, efforts vendor units, IMU accel in g, gyro rad/s) — unit conversion is the loader's job.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TOPICS = {
    "joint_states": "/joint_shm_controller/joint_states",
    "imu": "/imu_driver/imu_central",
    "imu_front": "/front_lidar/imu",
    "imu_rear": "/rear_lidar/imu",
    "odom": "/odom/mc_odom",
    "slam_odom": "/odom/slam_odom",
    "cmd": "/cmd_vel",
}


def _num_suffix(p: Path) -> int:
    m = re.search(r"_(\d+)\.db3$", p.name)
    return int(m.group(1)) if m else -1


def is_rosbag2(session_dir: str | Path) -> bool:
    d = Path(session_dir)
    return (d / "metadata.yaml").exists() and any(d.glob("*.db3"))


def db3_files(bag_dir: str | Path) -> tuple[list[Path], list[dict]]:
    """Openable `.db3` files in recording order, plus a list of skipped files (empty / not a database).

    A file that opens but is partially corrupt (aborted copy: readable head, garbage tail) is KEPT — iter_messages reads
    it up to the first bad page and reports the truncation.
    """
    good, skipped = [], []
    for f in sorted(Path(bag_dir).glob("*.db3"), key=_num_suffix):
        if f.stat().st_size == 0:
            skipped.append({"file": f.name, "reason": "0-byte file (recording/copy aborted before flush?)"}); continue
        try:
            con = sqlite3.connect(f"file:{f}?mode=ro&immutable=1", uri=True)
            con.execute("SELECT id, name, type FROM topics").fetchall(); con.close()
            good.append(f)
        except sqlite3.Error as e:      # header/topics table unreadable
            skipped.append({"file": f.name, "reason": f"sqlite error: {e}"})
    return good, skipped


def _typestore():
    from rosbags.typesys import Stores, get_typestore    # optional dependency (pure python)
    return get_typestore(Stores.ROS2_HUMBLE)


def iter_messages(bag_dir: str | Path, topics: list[str], truncations: list | None = None):
    """Yield (topic, t_bag_ns, decoded_msg) for the requested topics, file by file in rowid (= insertion) order.

    rosbag2 inserts in receive order, so rowid order is time order per topic. The scan goes through the table b-tree
    (not the timestamp index), so a partially corrupt file yields every row up to the first bad page; the truncation
    (file, rows read, last timestamp, error) is appended to `truncations` if given.
    """
    ts = _typestore()
    files, _ = db3_files(bag_dir)
    want = set(topics)
    for f in files:
        con = sqlite3.connect(f"file:{f}?mode=ro&immutable=1", uri=True)
        tmap = {tid: (name, typ) for tid, name, typ in con.execute("SELECT id, name, type FROM topics")}
        ids = [tid for tid, (name, _) in tmap.items() if name in want]
        if not ids:
            con.close(); continue
        q = f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({','.join('?' * len(ids))}) ORDER BY id"
        n = 0; last_t = None
        try:
            for tid, tsn, raw in con.execute(q, ids):
                name, typ = tmap[tid]; n += 1; last_t = int(tsn)
                yield name, int(tsn), ts.deserialize_cdr(raw, typ)
        except sqlite3.DatabaseError as e:
            if truncations is not None:
                truncations.append({"file": f.name, "rows_read_before_error": n, "last_timestamp_ns": last_t, "error": str(e)})
        finally:
            con.close()


def _hdr(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _quantile_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max()), "std": float(x.std())} if x.size else {}


def extract_bag_session(bag_dir: str | Path, out_dir: str | Path, time_source: str = "header", overwrite: bool = False) -> dict:
    """Decode the standard topics of one M1 rosbag2 directory into the SDK session layout under out_dir.

    time_source: 'header' (sensor stamps, default) or 'bag' (rosbag2 receive stamps) for the `t` columns.
    Returns the extraction report (also written to out_dir/extract_report.json). Idempotent unless overwrite=True.
    """
    bag_dir = Path(bag_dir); out_dir = Path(out_dir)
    if (out_dir / "joint_states.csv").exists() and (out_dir / "extract_report.json").exists() and not overwrite:
        return json.loads((out_dir / "extract_report.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    files, skipped = db3_files(bag_dir)
    buf: dict[str, list] = {k: [] for k in TOPICS.values()}
    truncations: list = []
    for topic, tsn, m in iter_messages(bag_dir, list(TOPICS.values()), truncations):
        buf[topic].append((tsn, m))
    rep: dict = {"bag": str(bag_dir), "bag_name": bag_dir.name, "extracted": _dt.datetime.now().isoformat(timespec="seconds"),
                 "time_source": time_source, "db3_files_read": [f.name for f in files], "db3_files_skipped": skipped,
                 "db3_files_truncated": truncations,
                 "message_counts": {k: len(v) for k, v in buf.items()}, "topics": {}}
    js = buf[TOPICS["joint_states"]]
    if not js:
        rep["error"] = "no joint_states messages"; (out_dir / "extract_report.json").write_text(json.dumps(rep, indent=1)); return rep
    names = list(js[0][1].name)
    t_bag = np.array([t for t, _ in js], dtype=float) * 1e-9
    t_hdr = np.array([_hdr(m) for _, m in js])
    t_ref = t_hdr if time_source == "header" else t_bag
    t0 = float(t_ref[0]); rep["t0_epoch_ns"] = int(round(t0 * 1e9))
    rep["t0_local"] = _dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds")

    def _tcol(msgs):
        tb = np.array([t for t, _ in msgs], dtype=float) * 1e-9; th = np.array([_hdr(m) for _, m in msgs])
        return (th if time_source == "header" else tb) - t0, tb - t0, th - tb

    def _topic_stats(key, msgs, t):
        d = {"topic": TOPICS[key], "n": len(msgs)}
        if len(msgs) > 2:
            dt = np.diff(t)
            _, tb, skew = _tcol(msgs)
            d.update(duration_s=float(t[-1] - t[0]), rate_hz_median=float(1.0 / np.median(dt)), dt_ms=_quantile_stats(dt * 1e3),
                     n_nonmonotone=int(np.sum(dt <= 0)), header_minus_bag_ms=_quantile_stats(skew * 1e3),
                     first_t_s=float(t[0]), last_t_s=float(t[-1]), frame_id=str(getattr(msgs[0][1].header, "frame_id", "")))
        return d

    # joint states
    t, tb, _ = _tcol(js)
    P = np.array([m.position for _, m in js]); V = np.array([m.velocity for _, m in js]); E = np.array([m.effort for _, m in js])
    if V.ndim != 2 or V.shape[1] != len(names): V = np.full_like(P, np.nan)
    if E.ndim != 2 or E.shape[1] != len(names): E = np.full_like(P, np.nan)
    cols = {"t": t, "t_bag": tb}
    for j, nm in enumerate(names):
        cols[f"{nm}_pos"] = P[:, j]; cols[f"{nm}_vel"] = V[:, j]; cols[f"{nm}_eff"] = E[:, j]
    pd.DataFrame(cols).to_csv(out_dir / "joint_states.csv", index=False, float_format="%.9g")
    rep["topics"]["joint_states"] = {**_topic_stats("joint_states", js, t), "names": names, "n_joints": len(names),
                                     "velocity_present": bool(np.isfinite(V).any()), "effort_present": bool(np.isfinite(E).any())}
    # IMUs
    for key in ("imu", "imu_front", "imu_rear"):
        msgs = buf[TOPICS[key]]
        if not msgs:
            rep["topics"][key] = {"topic": TOPICS[key], "n": 0}; continue
        ti, tbi, _ = _tcol(msgs)
        df = pd.DataFrame({"t": ti, "t_bag": tbi,
                           "qw": [m.orientation.w for _, m in msgs], "qx": [m.orientation.x for _, m in msgs], "qy": [m.orientation.y for _, m in msgs], "qz": [m.orientation.z for _, m in msgs],
                           "wx": [m.angular_velocity.x for _, m in msgs], "wy": [m.angular_velocity.y for _, m in msgs], "wz": [m.angular_velocity.z for _, m in msgs],
                           "ax": [m.linear_acceleration.x for _, m in msgs], "ay": [m.linear_acceleration.y for _, m in msgs], "az": [m.linear_acceleration.z for _, m in msgs]})
        df.to_csv(out_dir / f"{key}.csv", index=False, float_format="%.9g")
        st = _topic_stats(key, msgs, ti)
        st.update(orientation_constant=bool(df[["qw", "qx", "qy", "qz"]].std().max() < 1e-9),
                  accel_mean=[float(x) for x in df[["ax", "ay", "az"]].mean()], gyro_mean=[float(x) for x in df[["wx", "wy", "wz"]].mean()],
                  covariance_zero=bool(np.allclose(np.asarray(msgs[0][1].linear_acceleration_covariance), 0)))
        rep["topics"][key] = st
    # odometry
    for key in ("odom", "slam_odom"):
        msgs = buf[TOPICS[key]]
        if not msgs:
            rep["topics"][key] = {"topic": TOPICS[key], "n": 0}; continue
        to, tbo, _ = _tcol(msgs)
        df = pd.DataFrame({"t": to, "t_bag": tbo,
                           "x": [m.pose.pose.position.x for _, m in msgs], "y": [m.pose.pose.position.y for _, m in msgs], "z": [m.pose.pose.position.z for _, m in msgs],
                           "qw": [m.pose.pose.orientation.w for _, m in msgs], "qx": [m.pose.pose.orientation.x for _, m in msgs], "qy": [m.pose.pose.orientation.y for _, m in msgs], "qz": [m.pose.pose.orientation.z for _, m in msgs],
                           "vx": [m.twist.twist.linear.x for _, m in msgs], "vy": [m.twist.twist.linear.y for _, m in msgs], "vz": [m.twist.twist.linear.z for _, m in msgs],
                           "wx": [m.twist.twist.angular.x for _, m in msgs], "wy": [m.twist.twist.angular.y for _, m in msgs], "wz": [m.twist.twist.angular.z for _, m in msgs]})
        df.to_csv(out_dir / f"{key}.csv", index=False, float_format="%.9g")
        st = _topic_stats(key, msgs, to)
        xy = df[["x", "y"]].to_numpy()
        st.update(child_frame_id=str(msgs[0][1].child_frame_id), path_length_m=float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
                  start_xyz=[float(v) for v in df[["x", "y", "z"]].iloc[0]], end_xyz=[float(v) for v in df[["x", "y", "z"]].iloc[-1]],
                  pose_covariance_zero=bool(np.allclose(np.asarray(msgs[0][1].pose.covariance), 0)))
        rep["topics"][key] = st
    # command
    msgs = buf[TOPICS["cmd"]]
    if msgs:
        tc, tbc, _ = _tcol(msgs)
        pd.DataFrame({"t": tc, "t_bag": tbc, "vx": [m.linear.x for _, m in msgs], "vy": [m.linear.y for _, m in msgs], "wz": [m.angular.z for _, m in msgs]}).to_csv(out_dir / "cmd.csv", index=False, float_format="%.9g")
        rep["topics"]["cmd"] = {"topic": TOPICS["cmd"], "n": len(msgs)}
    else:
        rep["topics"]["cmd"] = {"topic": TOPICS["cmd"], "n": 0}
    meta = {"robot": "m1", "source": "hardware", "recording": "rosbag2 sqlite3", "bag_name": bag_dir.name, "date": rep["t0_local"][:10],
            "locomotion": "unknown (fill from audit: wheeled rolling | standing | stepping)", "rate_hz": rep["topics"]["joint_states"].get("rate_hz_median"),
            "efforts_semantics": "unknown", "imu_accel_units": "g", "imu_gyro_units": "rad/s", "imu_frame_id": rep["topics"].get("imu", {}).get("frame_id", ""),
            "joint_names": names, "time_source": time_source, "t0_epoch_ns": rep["t0_epoch_ns"], "extract_report": "extract_report.json",
            "topics": {k: v.get("n", 0) for k, v in rep["topics"].items()}, "db3_files_skipped": skipped, "db3_files_truncated": truncations, "notes": ""}
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (out_dir / "extract_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract an M1 rosbag2 directory into the SDK-layout session (CSV + meta.yaml).")
    ap.add_argument("bag_dir"); ap.add_argument("out_dir"); ap.add_argument("--time-source", choices=["header", "bag"], default="header")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    rep = extract_bag_session(a.bag_dir, a.out_dir, a.time_source, a.overwrite)
    print(json.dumps({k: rep[k] for k in ("bag_name", "db3_files_read", "db3_files_skipped", "message_counts")}, indent=1))


if __name__ == "__main__":
    main()
