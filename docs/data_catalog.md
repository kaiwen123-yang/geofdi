# Data catalog

One row per ingested raw session. Rows are appended automatically by
`scripts/ingest_session.sh`; fill `gait` / `terrain` / `notes` by hand when the
session name does not encode them. `sha256(8)` is the first 8 hex chars of the
sha256 of the session's `checksums.sha256` (a stable payload fingerprint).

| date | session | category | gait | terrain | sha256(8) | notes |
|------|---------|----------|------|---------|-----------|-------|
