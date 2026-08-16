# Sprint 8 report — M1 real data + leftover clearance + prediction experiments + public data harvest (2026-08-16)

All six planned blocks shipped: **D** (M1 hardware, headline), **L** (leftover fixes), **T2** (N1-2 two-layer),
**P** (predictions e15), **G** (πᵢ gating e16), **PUB** (public data e17). Packs rp025–rp030; tags `theory-part1-v1.1`,
`real-data-milestone-1`. 59 tests pass; `make theory` 0 error / 0 undefined / 0 overfull. Progress: `sprint8_progress.md`.

## 1. M1 real-data audit — the four `H:\m1_data` sessions of 2026-08-10 (one-line verdicts)

`/mnt/h` was not mounted (sudo needs a password); the data were staged read-only via **Windows robocopy**, the H:
originals hashed with `Get-FileHash`, and the ingest checksums verified equal to that list (source untouched).

| session | verdict |
|---|---|
| `m1_static_20260810_172037` (72 s) | **archive / audit** — standing, motors on, asymmetric vendor stand pose; rest-noise / IMU-bias / encoder-quantum reference only (no motion, no odometry). |
| `m1_walk_20260810_172847` (70.6 s, 30 m) | **rolling H₀′ + InEKF (small)** — wheeled rolling, 5 straight runs / 30.6 s; first segment of the continuous drive. |
| `m1_walk_20260810_173028` (129.9 s, 82.5 m) | **primary rolling session** — 12 straight runs / 83 s, long yaw loops; the best of the four. |
| `m1_walk_20260810_173247` (76 of 151 s readable, 49 m) | **rolling H₀′ + InEKF (truncated)** — db3 `_1` partially corrupt at the source, `_2/_3` 0-byte; the tolerant reader recovers 76 s. |

All "walk" sessions are **wheeled rolling, no leg gait** (like the legacy bags). Mapping VERIFIED (`io/m1_mapping.yaml`,
`unverified: false`): joint names/order, per-leg mirror signs (RF/RH = [+1,−1,−1,−1]; wheel forward sign), IMU frame
(x right, y back, z down; g units; `R_body_from_sensor`), odometry conventions. efforts semantics still vendor-open.

## 2. First real-robot figures (the deliverable)

| figure | what it shows |
|---|---|
| `results/e18_m1_real/e18-20260816/e18_real_h0_h0prime.png` | **First real-robot R⁻ H₀/H₀′** (M1, 3 sessions): naive H₀ rejects (real ε_dyn), sequential H₀′ e-process silent. |
| `…/e18_inekf_real.png` | **Rolling InEKF path recovery on real data**: rolling filters recover 0.99–1.00 of the driven path, fixed-foot ~0.03 (real-robot e10). |
| `results/m1_h_audit/audit_timeline_*.png`, `audit_odom_paths.png` | M1 session timelines + odometry paths (audit). |
| `results/e15_predictions/e15-20260816/e15c_slip_regimes.png` | **Blindness theorem as a slip classifier** (P3): unilateral slip → R⁻; bilateral → InEKF NIS. |
| `…/e15b_statistic_split.png` | **Statistic split** (P2): paired_energy blind to a zero-mean law difference, energy_distance consistent. |
| `…/e15a_chirality_ceiling.png` | **P1**: blindness robust to symmetric faults (no ceiling reached; A5-under-fault holds). |
| `results/e16_pi_gating/e16-20260816/e16_gating.png` | **πᵢ gating FAR guarantee**: fixed 0.4 m/s threshold false-rejects 15 % of nominal contacts, GeoFDI ≈ α. |
| `results/e17_public_realdata/e17-20260816/e17_legkilo_h0prime.png` | **Public real-robot R⁻ H₀/H₀′** (Leg-KILO Go1): naive H₀ rejects, H₀′ in band on 4/5 sequences. |
| `…/e17_minicheetah_terrain.png` | **Cross-terrain real-robot R⁻** (Mini Cheetah, 8 terrains): detection fires on every terrain. |
| `…/e17_street_h0prime.png`, `…/e17_legkilo_gating.png` | Street A1 R⁻; Leg-KILO real-data gating FAR. |

## 3. Leftover-clearance status

| item | status |
|---|---|
| `make_review_pack.sh` MANIFEST bug (rp020–024) | **fixed** — matches `MANIFEST*.md`; rp020–024 top-level MANIFEST back-filled. |
| H₀′-primary report notes (run_pipeline + Day-0 docs) | **done**. |
| weld e13a/e13b (RA-L split material) | **done** — plain DeLaN H₀ size 1.00 vs equivariant 0.00; H₀′ restores; `split_option.md`. |
| e03 mirrored-bilateral n=8 note | **done** (`liu_a1_audit.md`). |
| N2 formalisation registered as next theory item | **done** (`docs/sprints/theory_backlog.md`). |

## 4. Theory + predictions

N1-2 rewritten two-layer (`02_n1_theorems.tex`, tag `theory-part1-v1.1`): Layer I law-level blindness dichotomy with the
explicit **A5-under-fault** hypothesis + amplitude-ceiling remark; Layer II mean-level power; statistic-consistency
remark. e15 (Block P) tested the three forward-references and the P1/P2 anchors are back-filled. Honest outcomes: P1
blindness robust (ceiling not reached by symmetric faults); P2 statistic split decisive in the toy, directional in the
closed loop; P3 slip-classifier clean on Go2.

## 5. "Only things still missing" (next steps)

1. **Controlled physical faults on M1** — a single-wheel μ / configurable bilateral injection (for the clean e03-style
   blindness demonstration and the P1 ceiling on real hardware); the sim faults exist, the hardware injection does not.
2. **Gate 4** — joint-level command access on M1 (needed for the residual channel and fault injection); Day-0 §3.
3. **efforts-semantics vendor confirmation** (current vs torque) — blocks the M1 residual R⁻.
4. **N2 formalisation** — the two propositions in `theory_backlog.md` (next theory sprint).
5. **The five-piece write-up** — assemble the T-RO manuscript from the five-layer evidence stack (`outline.md` §8).

Secondary (owed, not blocking): the amplitude-ceiling needs a genuine-bifurcation regime to demonstrate; Mini Cheetah
cycle-level H₀′ needs a stationary gait / better flying-trot phase registration; the gating over-rejection differentiator
needs a faster real gait than Leg-KILO walking; a physical slip session per `hw_slip_protocol.md`.
