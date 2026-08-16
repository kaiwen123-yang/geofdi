# Sprint 9 report — QUADRIC-GINS Go2 corpus + background backlog cleared (2026-08-16)

All three blocks shipped: **Q** (corpus audit + ingest), **R** (own-Go2 hardware experiments), **B** (nine backlog items,
9/9). Packs rp031–rp033; tags `theory-part3-v1`, `real-data-milestone-2`. 65 tests pass; `make theory` 0 error /
0 undefined / 0 overfull / **0 remaining TODOs**. Progress: `sprint9_progress.md`.

## 1. Q1 determination checklist (the格 that decides everything in bold)

| item | verdict |
|---|---|
| **per-joint `motorState` (q / dq / τ_est)** | **ABSENT — there is no joint stream.** The files are `script(1)` transcripts of `ros2 topic echo /sportmodestate`, and the Unitree high-level API carries no `motorState` at all (not 20 motors, not 12: zero) |
| IMU (quaternion / gyro / accel / rpy / temp) | present, ≈ 250 Hz, body FLU |
| `foot_force[4]` | present (raw counts) |
| **`foot_position_body[12]` / `foot_speed_body[12]`** | **present — the decisive compensation**: the robot's own FK of each leg, so the per-leg kinematic asymmetry survives without joints |
| body position / velocity / yaw_speed / body_height | present (on-board estimator, diagnostics only) |
| gait / mode fields | present (`mode` 3 = locomotion, `gait_type` 1) |
| time base | **Unix epoch UTC, common to all three streams** ⇒ exact 1:1 session↔Fixposition matching, no motion-correlation needed |
| foot inertial unit | 2 redundant IMUs on **one** board, 200 Hz, LH leg (operator), January only ⇒ validation source, never a mirror channel |
| Fixposition | geodetic/odometry/status parsed; fix quality strongly site-dependent (0.51 → 1.00) |

Consequence: the element is **foot_pos(12) + foot_vel(12) + foot_force(4) + IMU(6) = 34 in-Z channels, 28 leg-resolved**
— better than the planned "degraded footForce+IMU" fallback. Only the torque channel (and with it R⁺) is lost.

## 2. R1 — per-session table (11 own Go2 sessions)

| | result |
|---|---|
| naive H₀ | **rejects 11/11** (p ≈ 0.002; window-reject 0.67–0.98) — prediction 1 confirmed |
| H₀′ (pooled runs) | alarms 8/11 — prediction 2 **falsified as stated**, diagnosed in R5 |
| ν₀ | 7.2 – 21.9 |
| anti-symmetric energy | hind pair LH/RH in 6/11 (median share 0.52 vs 0.44); dominant family `foot_force` 11/11 |
| straight yield | 95–272 s per session, median speed 1.03–1.19 m/s (confirms the operator's ≈ 1 m/s) |

**R5 diagnosis (the payoff):** H₀′ alarms **8/11 when runs are pooled but 2/11 within a single run**; the ν-shape
classifier gives 8/11 `boundary-jump`, **0 `drift`**. The asymmetry is stationary inside a traverse and steps between
traverses ⇒ **calibrate ν₀ per run, not per session**. The signed L−R asymmetry is stable in **11/11** sessions
(LH−RH foot force negative, LF−RF positive) across three sites and two months — and the LH foot-IMU board is
**falsified** as its cause, since March carries no board yet shows the same or larger asymmetry.

## 3. R2 — cross-period reproducibility table

| quantity | January (8) | March (3) |
|---|---|---|
| ν₀ range | 7.2 – 21.7 | 9.9 – 21.9 |
| within-site sd | A 4.53, B 3.11 | C 6.38 |
| **between-date ratio mean(Mar)/mean(Jan)** | **1.18** | |

**Prediction 3 confirmed:** the two-month change of the asymmetry level is *smaller* than the session-to-session spread
within one site and day — the project's strongest reproducibility statement (Sprint 8's M1 evidence was single-day).
Cross-session H₀′ calibration does **not** transfer (0.32 Jan→Mar, 0.49 Mar→Jan): ν₀ is reproducible as a *level*, a
calibration is not portable.

Also from R4: on real outdoor trot the fixed 0.4 m/s foot-speed threshold discards **40.6 %** of nominal contact
measurements against **1.9 %** for the calibrated πᵢ gate — the real-robot confirmation of e16, closing the gap that
Leg-KILO's slow walk left open in Sprint 8. Honest null: gating does not improve ATE here (no severe natural slip).

## 4. Block B — nine items

| # | item | status |
|---|---|---|
| B1 | classical De Luca + χ² baseline | ✓ detection 0.77/0.96 vs R⁻ 0.71 **but FAR 0.15 / 0.50 vs 0.00**; nuisance 0.90–0.95 vs 0.00; classical wins on friction |
| B2 | robustness sweeps | ✓ phase ±10 % costs power not level; K_cal 60 enough; **block ≥ 2τ** |
| B3 | N2 written up | ✓ new Part 3, proofs; tag `theory-part3-v1` |
| B4 | N1-3 attempt | ✓ **closed as a proposition** (power lower bound + sample-size corollary, numerically verified) |
| B5 | 173247 diagnosis | ✓ **corrects Sprint 8**: between-window jump, not slow drift; needs ≥ 60 s runs to settle |
| B6 | Mini Cheetah block mode | ✗ **negative**: alarm drop confounded by window count; aggressive-gait limitation |
| B7 | joint sign tables | ✓ back-filled; **last theory TODO retired** |
| B8 | P1 Remark rewrite | ✓ leads with the falsification + rp003 anchor |
| B9 | MANIFEST packing bug | ✓ fixed **and** made unable to fail silently (post-build assertion, verified) |

## 5. Things only a robot day can still fix

1. **Controlled physical faults** — single-wheel μ patch / configurable bilateral injection (M1) and a payload-swap
   session on the Go2 (to separate payload placement from wear as the cause of the stable L−R asymmetry).
2. **Gate 4** — joint-level command access on M1 (residual channel + fault injection).
3. **efforts-semantics vendor confirmation** (current vs torque) on M1.
4. *(new, from B5)* **≥ 60 s continuous straight runs on M1** — the current 12–14 s runs cannot support a within-run H₀′
   control, so the 173247-type question stays open on that platform.
5. *(new, from Q1)* **A `LowState` recording on the Go2** if the joint/torque channel is ever wanted there — the
   high-level API cannot provide it, so R⁺/residual work on this robot needs a different recording setup.

Everything else formerly on this list is now closed in simulation, on public data, or on the two own corpora.
