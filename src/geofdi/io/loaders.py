"""Dataset access: M1 telemetry, rosbags, and public benchmarks.

Future home of: readers for raw sessions ingested by
scripts/ingest_session.sh, meta.yaml/checksum validation, and loaders
for the public sets (liu-a1-fault, street-a1, legkilo-go1) used by
experiment e03. All paths resolve through the repo data/ symlink or
$GEOFDI_DATA_ROOT — never hard-code machine mounts. Workstream N3.
"""
