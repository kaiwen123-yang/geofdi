# m1_bag_tools — rosbag2 (sqlite3) helpers for M1 recordings

Pure Python (sqlite3 + PyYAML; `rosbags` optional for decoding). No ROS install needed, so the
tools run on the desktop against bags on the data volume.

| tool | purpose |
|---|---|
| `bag_inventory.py` | topic × type × count × rate table per bag (metadata + measured median dt/jitter from file 0), metadata-vs-sqlite consistency check, optional sample decoding (`--decode`; JointState names/count, Imu frame, Odometry/TF frames). Writes Markdown/JSON. |
| `fix_metadata.py` | the metadata repair flow: `--check` a bag (missing/inconsistent `metadata.yaml`, DDS-style type names from zenoh-bridge recordings, empty QoS strings), or regenerate a version-5 `metadata.yaml` from the `.db3` files (`--out DIR` for a repaired copy — raw bags stay immutable — or `--in-place`). |

Typical flow for a batch of legacy bags:

```sh
V=~/venvs/geofdi/bin/python
for b in "$GEOFDI_DATA_ROOT"/data/raw/m1/legacy-aug/*/; do PYTHONPATH= $V scripts/m1_bag_tools/fix_metadata.py --check "$b" || echo "needs repair: $b"; done
PYTHONPATH= $V scripts/m1_bag_tools/bag_inventory.py --decode --markdown docs/protocol/legacy_aug_inventory.md "$GEOFDI_DATA_ROOT"/data/raw/m1/legacy-aug/*/
```

If `--check` fails, write the repaired metadata next to a *copy* of the bag (or into a scratch
dir) with `--out`, verify with `bag_inventory.py`, and only then ingest. `PYTHONPATH=` is cleared
because a host ROS install leaks its site-packages into the venv.
