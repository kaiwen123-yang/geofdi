# Data catalog

One row per ingested raw session. Rows are appended automatically by
`scripts/ingest_session.sh`; fill `gait` / `terrain` / `notes` by hand when the
session name does not encode them. `sha256(8)` is the first 8 hex chars of the
sha256 of the session's `checksums.sha256` (a stable payload fingerprint).

| date | session | category | gait | terrain | sha256(8) | notes |
|------|---------|----------|------|---------|-----------|-------|
| 2026-08-15 | grufd-ftc_84ca180 | public/liu-a1-fault | trot | flat (provenance unresolved) | 859e7b42 | Liu et al. RA-L 2025 GRUFD-FTC, git 84ca180; 10 CSV, 100 Hz rows (README says 50), knee faults η∈{0.4,0.6}; audit: docs/protocol/liu_a1_audit.md |
