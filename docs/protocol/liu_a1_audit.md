# Audit — Liu et al. (RA-L 2025) Unitree A1 joint-partial-failure dataset (`liu-a1-fault`)

Date: 2026-08-15 · Auditor: Claude (session with K. Yang) · Status: **audited, ingested**

## 1. Source

| item | value |
|---|---|
| paper | K. Liu, Z. Wang, B. Li, L. Zhu, H. Ding, "Fault Joint Detection and Adaptive Fault-Tolerant Control of Legged Robots Under Joint Partial Failures", *IEEE RA-L* 10(10):10234–10241, Oct 2025, DOI [10.1109/LRA.2025.3598620](https://doi.org/10.1109/LRA.2025.3598620) (closed access; abstract via Crossref/OpenAlex). Framework name in the paper: GRUFD-FTC. |
| dataset repo | <https://github.com/zhongrenjiexing/GRUFD-FTC> (author K. Liu, HUST). Cloned 2026-08-15, HEAD `84ca180` ("change readme"), 10 commits, no license file, no code — **data only** (the GRU detector / FTC code is *not* released). |
| contents | 10 CSV files (89 MB total, ~9.3 MB each), no header row, 69 numeric fields per row + trailing comma. `README.md` (457 B) is the only documentation. |
| local copy | `$GEOFDI_DATA_ROOT/data/raw/public/liu-a1-fault/grufd-ftc_84ca180/` (ingested via `scripts/ingest_session.sh`; `SOURCE.md` + `GRUFD-FTC_84ca180.zip` archive + extracted CSVs; catalog row in `docs/data_catalog.md`). |

## 2. What a row contains (README, column indices verified against the data)

| columns | content | notes |
|---|---|---|
| 0–2 | body angles θ_B | col 0 = **yaw** (range 2π, wraps), col 1 = **pitch**, col 2 = **roll** — established from Δθ vs. rate integration (pitch–ω_y slope 0.98, roll–ω_x slope 0.90) |
| 3–5 | body angular velocity ω_B | col 3 = ω_x (roll rate), col 4 = ω_y (pitch rate), col 5 = ω_z (yaw rate) |
| 6–17 | joint fault rates η (12) | 1.0 = healthy; faulty values ∈ {0.4, 0.6} = **torque retention rate** |
| 18–29 | joint positions q (12) | rad |
| 30–41 | desired joint positions q_des (12) | controller output (MPC + WBC per the paper) |
| 42–53 | joint velocities dq (12) | rad/s |
| 54–65 | desired joint velocities dq_des (12) | never zero |
| 66–68 | body command (v_x, v_y, ω_z) | piecewise constant: v_x ∈ {0, 0.25, 0.5} m/s, v_y = 0, ω_z ∈ {0, ±0.25, ±0.5, ±1.0} rad/s |

Per leg the joint triple is (hip-abduction, thigh, calf); leg blocks are 0–2, 3–5, 6–8, 9–11.
**Not present:** timestamps, joint torques / motor currents, foot-contact flags, base linear
velocity/position, IMU accelerometer, temperatures.

**Leg ordering (data-derived, no documentation).** Zero-lag thigh correlations: legs (0,3) and
(1,2) in phase (ρ = 0.98/0.99), all other pairs anti-phase at a lag of half a period → (0,3),
(1,2) are the trot **diagonal** pairs. Hip means: legs 0,1 ≈ −0.10…−0.14 rad, legs 2,3 ≈
+0.11…+0.14 rad, and thigh/calf marginals match for (0,2) and (1,3) (KS 0.03–0.04) but not for
(0,1),(2,3) (calf KS ≈ 0.2, thigh means 0.80 vs 0.84) → the **left–right mirror pairs are (0,2)
and (1,3)**, with the hip sign flipping between mirror legs (Unitree sign convention). Hence
the order is `[R_X, R_Y, L_X, L_Y]` with X/Y ∈ {front, rear}; which of X/Y is front is not
determinable from the data alone (thigh means 0.80 for X, 0.84 for Y). This does not affect
the mirror-pairing rule (Def. pairing in theory Part 0), which only needs the mirror pairs and
the diagonal pairs.

## 3. Sampling rate — README says 50 Hz, the data says 10 ms rows

Two independent internal checks give **row spacing = 10 ms (100 Hz)**, not 20 ms:

* `dq` vs. central finite difference of `q` at dt = 20 ms: correlation 0.99–0.999 on every
  joint but a slope of exactly 2.0 (e.g. thigh std 1.497 vs 0.742) → dt = 10 ms.
* Δyaw vs. ω_z·dt: regression slope 0.998 (rot file) / 0.943 (vel file) with dt = 10 ms,
  0.499 / 0.471 with dt = 20 ms.

Consequently the gait period is 59 rows ≈ **0.59 s** (autocorrelation peaks at 59, 118, 175
rows), which is the usual A1 MPC trot period; at 50 Hz it would be an implausible 1.18 s. All
durations below use 10 ms/row; the 50 Hz reading (double all durations) is kept in brackets
where it matters. Possible reading of the README: 50 Hz is the rate the GRU consumed after
2× decimation. **e03 must treat the rows as 100 Hz.**

## 4. Fault labels

* Faulty joints: **only the four calf (knee) joints**, indices 2, 5, 8, 11 (one per leg).
* Torque retention η ∈ {**0.4, 0.6**} (i.e. 60 % / 40 % torque loss).
* Files `*_Single_*`: one faulty joint at a time — 8 (joint, η) combinations × 600 rows each.
  Files `*_Double_*`: two faulty joints simultaneously with the same η — all 6 joint pairs ×
  2 η values = 12 combinations × 400 rows each. Both: 4 800 fault rows = **48 s** per file.
* Timeline is identical across files: healthy from row 0; **first onset at row 2 868–3 040
  (28.7–30.4 s)**; then **25 fault episodes of 100 or 200 rows (1 s or 2 s)** separated by
  exactly **400-row (4 s) healthy gaps**; onsets and offsets are exact rows (label is a step
  function; no ramp). Body command changes 2–3 times per file (every ~4 800 rows).
* Structural note for GeoFDI: the double-fault combinations (2,8) and (5,11) put the **same** η
  on a **mirror pair** — precisely the bilateral-synchronized case that is invisible to a pure
  Σ-invariance test (theory Part 0, Example *bilateral*). Combinations (2,5), (8,11) (same-side
  pairs) and (2,11), (5,8) (diagonal pairs) break the mirror symmetry and are detectable.
  e03 must report the mirror-pair subset separately.

## 5. Nominal (η ≡ 1) material

| quantity | rows | at 10 ms/row | [at 20 ms/row] |
|---|---|---|---|
| total | 173 868 | 29.0 min | [58.0 min] |
| healthy (all η = 1) | 125 874 | **21.0 min** | [42.0 min] |
| healthy **and** trotting (thigh oscillating) | ≈ 119 800 | ≈ 20.0 min | [40 min] |
| fault | 47 994 | 8.0 min (250 episodes) | [16 min] |

Per file: ≈ 12 500 healthy rows = 125 s, of which one contiguous block of ≈ 29–30 s at the
start (trot in place, zero command) and 25 blocks of 4 s between fault episodes (with body
commands). Longest healthy block **with** a non-zero command: 4 s ≈ 7 gait cycles. Robot trots
from row ≈ 99 to the end of every file (no standing segments to speak of).

## 6. Signal-quality notes and simulation-vs-hardware evidence

* Tracking `q − q_des`: RMSE 1–6 mrad; the residual is broadband (25–36 % of its power above
  20 Hz), i.e. white at the mrad level.
* Gyro: x/y channels have 2–6 % of power above 20 Hz (floor ≈ 1.5–3.6 × 10⁻⁶ (rad/s)²/Hz);
  yaw rate is clean (0.2 %).
* Fault labels are exact software step functions with 100/200/400-row segment lengths;
  numbers are printed with 6 significant digits (`%g`), so encoder quantization is invisible.
* The paper's abstract states "simulations and real-world experiments"; neither the README nor
  the repo says which produced the CSVs.

**Verdict on provenance: UNRESOLVED — most likely simulation-grade data** (exact η bookkeeping,
mrad tracking, perfectly regular schedule, no contacts/torques/timestamps), but a hardware run
with software-scaled torque commands would look the same at this level of logging. Action: get
the paper's data-collection section (institutional access) and settle it before e03 results are
quoted as "hardware".

## 7. Controller symmetry — decidable?

**Yes, in principle.** The controller's output (`q_des`, `dq_des`) and the states it acts on
(`q`, `dq`, body angles/rates, command) are all logged, so both the state-matched Gate-1
statistic (theory Part 0, A2 audit) and its unconditional version can be computed on the
healthy segments; the controller is an MPC + WBC stack, symmetric by construction unless gains
differ per leg. Pilot (unconditional, mirror pairs (0,2)/(1,3), half-period shift, healthy
moving rows of `cmd_vel_Single_trot`): thigh KS = 0.031/0.036, calf KS = 0.038/0.043 (n ≈ 7 000
each; small effect but statistically non-zero), hip KS = 0.19/0.13 with mean offsets 5–10 mrad
and std ratios up to 1.5. Reading: near-symmetric with a small stable asymmetry — the
per-leg-bias-calibration degradation path (A1 in Part 0) or H0′ would absorb it. Limits: no
full state (no base velocity/height, no contacts) for state matching; state must be proxied by
(q, dq, body rates).

## 8. e03 feasibility verdict

**可行 — FEASIBLE**, as the *data* side of the head-to-head, with these binding conditions:

1. Sampling: treat rows as **100 Hz** (§3); phase from thigh-angle oscillation (no contact
   flags); gait period ≈ 59 rows.
2. Nominal calibration material: ≈ 20 min of healthy trot in total, but fragmented (29–30 s
   in-place block + 4-s gaps per file); per-cycle statistics are fine, long-run
   false-alarm-per-hour estimates will be coarse.
3. Fault episodes are **1–2 s (≈ 2–3 gait cycles)** — detection delay must be reported at
   sub-second / per-cycle resolution; 250 episodes total, η ∈ {0.4, 0.6}, knees only.
4. Report the mirror-symmetric double-fault subset (2,8), (5,11) separately (structural blind
   spot of GeoFDI, expected miss).
5. Baseline: the paper's GRU (GRUFD) is **not released** — the `supervised_classifier` baseline
   in `experiments/e03_liu_a1_headtohead` must be our re-implementation trained on this
   dataset (train/test split protocol unknown → define our own and state it).
6. Provenance caveat of §6 travels with every e03 figure until resolved.

The degraded option ("reproduce their GRU on our simulation data") is **not** needed for the
data side; it remains relevant only for the baseline (item 5) if a faithful GRU re-implementation
proves impossible from the paper text.
