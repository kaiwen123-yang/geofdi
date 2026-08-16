# Go2 QUADRIC-GINS corpus — session metadata form (Sprint 9 Q1.5)

11 sessions, Unitree Go2, recorded for the QUADRIC-GINS project and re-used here as GeoFDI hardware corpus.
Rows marked **[user]** are the operator's own account (dictated 2026-08-16); rows marked **[inferred]** were derived from
the data (RTK track geometry / GNSS quality / foot-force + IMU vibration texture) and are the audit's *hypothesis*, to be
corrected by the operator. Empty cells are genuinely unknown — please fill.

## Corpus-wide conditions [user]
| item | value |
|---|---|
| weather | clear/sunny, all 11 sessions |
| commanded speed | low, ≈ 1 m/s (not precise) — **data agree: median body speed 1.00–1.07 m/s on the straight segments** |
| payload | ≈ 5 kg or slightly more, **constant across every session** (position on the trunk not recorded) |
| foot IMU | one board on the **LEFT-HIND (LH) leg only**, two redundant IMUs on it (imu_0/imu_1), 200 Hz. Single leg ⇒ it is a *validation source only* (phase / touch-down cross-check), never a mirror channel |
| robot age | in service > 1 year at the time of recording |
| gait | Go2 sport-mode locomotion (`mode = 3`, `gait_type = 1`) throughout the moving part of every session |

**Consequence of the constant payload (design change vs the original spec):** there is no payload-variation contrast in
this corpus, so R3 does not run a payload-nuisance row; "constant ≈ 5 kg payload" is recorded as a corpus-wide condition
instead. If a lateral asymmetry shows up, the two candidate true causes are (a) payload offset from the centreline and
(b) the LH foot-IMU board + its cable — both are *known* asymmetries of this robot, not faults. See pre-registration P-LH.

## Sites [user]
| site | description |
|---|---|
| A | indoor–outdoor transition; half semi-rough tile, half smooth tile — **a surface switch occurs within a session** |
| B | "soft/degraded" scenario: smoother ground, GNSS-restricted |
| C | rough ground |

## Per-session table
`fix_ok` = fraction of the session with Fixposition GNSS status ∈ {5, 8}; `pvar_x` = median reported position variance.

| # | session | day | start (UTC) | dur [s] | site | terrain | payload | cmd speed | gait | weather | anomalies / notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | nmb1 | 2026-01-05 | 11:16:05 | 408 | **B** [inferred] | smooth, GNSS-restricted [inferred] | ≈5 kg [user] | ≈1 m/s [user] | trot [user] | clear [user] | fix_ok 0.55, pvar_x ≈ 0.4 m |
| 2 | nmb2 | 2026-01-05 | 11:25:05 | 363 | **B** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | fix_ok 0.52 |
| 3 | nmb3 | 2026-01-05 | 11:32:45 | 365 | **B** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | fix_ok 0.57 |
| 4 | nmb4 | 2026-01-05 | 11:39:51 | 348 | **B** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | fix_ok 0.51 |
| 5 | xb1 | 2026-01-05 | 12:25:15 | 396 | **A** [inferred] | tile, rough→smooth switch [inferred] | ≈5 kg | ≈1 m/s | trot | clear | fix_ok 0.82, pvar_x ≈ 7.6 m (indoor/outdoor GNSS swings) |
| 6 | xb2 | 2026-01-05 | 12:33:39 | — | **A** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | |
| 7 | xb3 | 2026-01-05 | 12:40:55 | — | **A** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | |
| 8 | xb4 | 2026-01-05 | 12:49:28 | — | **A** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | pvar_x ≈ 34 m |
| 9 | by1 | 2026-03-06 | 07:52:23 | 446 | **C** [inferred] | rough ground [inferred] | ≈5 kg | ≈1 m/s | trot | clear | fix_ok high, pvar_x ≈ 0.002 m (clean open sky); no foot IMU recorded |
| 10 | by2 | 2026-03-06 | 08:00:55 | 302 | **C** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | no foot IMU |
| 11 | by3 | 2026-03-06 | 08:06:41 | 297 | **C** [inferred] | " | ≈5 kg | ≈1 m/s | trot | clear | no foot IMU |

### Basis of the site inference [inferred] — see `go2_quadric_audit.md` §5 for the numbers
1. **The three name groups occupy three geographically distinct places.** Session-median WGS-84 position:
   `nmb*` 39.9783 N / 116.3456 E, `xb*` 39.9834 N / 116.3394 E (≈ 700 m from nmb), `by*` 39.9846 N / 116.3424 E.
   So the group name (nmb / xb / by) *is* the site label; no session mixes two of them.
2. **nmb → B (GNSS-restricted).** Lowest fix quality of the corpus (RTK-fixed only ~12 % of status samples, `fix_ok`
   0.51–0.57, reported position variance 0.4–2.2 m) — the signature of a GNSS-restricted place, matching the operator's
   "soft/degraded scenario, GNSS-restricted".
3. **xb → A (indoor–outdoor transition).** Intermediate and *highly variable* GNSS: `fix_ok` 0.82 but position variance
   swinging to 7.6–34 m — i.e. the receiver repeatedly loses and regains sky view, which is what walking in and out of a
   building does. Cross-checked against the foot-force/IMU vibration texture for the within-session surface switch (§5).
4. **by → C (rough ground).** Cleanest GNSS of the corpus (variance 0.002–0.003 m, wide open sky) yet the *highest*
   vibration texture (§5), i.e. rough ground under an open sky.

**Please correct any [inferred] cell.** The cross-period comparison (R2) pairs sessions by site, so a wrong site label
would mix conditions; until corrected, R2 additionally reports the weaker "same-day" pairing as a fallback.
