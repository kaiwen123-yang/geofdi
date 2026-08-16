# Sprint 10 report — manuscript v0 + closing experiments + onboard proxy benchmark (2026-08-16)

All three blocks shipped: **M0** (metadata finalised, three closing experiments), **W** (full manuscript v0),
**N** (onboard proxy benchmark). Packs rp034–rp036; tag `paper-draft-v0`. 65 tests pass; `make paper` builds 10 pages
with zero errors and zero undefined references; `make theory` remains clean.

## 1. Manuscript v0 — 10 pages, IEEEtran two-column

| § | section | page |
|---|---|---|
| I | Introduction (+ Fig. 1 concept figure) | 1 |
| II | Related Work and Position (+ Table I claims matrix) | 2 |
| III | Preliminaries | 3 |
| IV | Theory | 3 |
| V | Method (Algorithm 1 + Table II nine conditions; per-traverse rule; compute) | 4 |
| VI | Simulation Study | 5 |
| VII | External Benchmarks | 6 |
| VIII | Hardware Evaluation (+ placeholder box) | 6 |
| IX | Limitations (8 items) | 8 |
| X | Conclusion | 8 |
| App. | Proofs + numbering map | 9–10 |

Five headline pieces all present: working title (+2 alternates), Fig. 1, Algorithm 1 with **nine** correctness
conditions (C9 — side attribution needs a *signed* statistic — is the new one, and the text says we got it wrong once),
a full Introduction, and the claims matrix with every cell tagged by the experiment that produced it.

## 2. Placeholder box = the robot-day shopping list, one to one

| tag | experiment | what it fills |
|---|---|---|
| P1 | unilateral low-friction patch | hardware one-sided slip row (sim: R⁻ power 1.00) |
| P2 | bilateral low-friction patch | the blindness prediction on hardware (R⁻ at α, NIS elevated) |
| P3 | unilateral ballast | calibrates the detectable-magnitude axis |
| P4 | payload diagonal rotation | separates payload offset from wear as the cause of the stable LH−RH / LF−RF pattern |
| P5 | ≥ 60 s continuous straight runs (M1) | settles the residual-anomaly question on that platform |
| P6 | `LowState` recording (Go2) | unlocks the residual (R⁺) channel and the friction row on this robot |
| P7 | joint-level command access (M1) | controlled actuator-fault injection |

## 3. Bench (laptop proxy, one core, BLAS 1 thread, median of 3)

| robot | detector per cycle/block | budget | real-time factor | InEKF | sustainable |
|---|---:|---:|---:|---:|---:|
| Go2 `by2` | 3.38 ms/cycle | 443.7 ms | **131×** | 0.163 ms/sample | **6137 Hz** vs 249 Hz telemetry |
| M1 `…173028` | 2.25 ms/block | 1000 ms | **444×** | — | — |

Per-stage (Go2): phase 1.47 and H₀′ 1.75 ms dominate; the flip test itself is 0.06 ms at M = 512. NUC13 config and a
three-step protocol are ready for tomorrow's run into the same CSV schema.

## 4. Closing experiments

- **Site labels confirmed** exactly as inferred (xb=A, nmb=B, by=C) ⇒ **R2 numbers unchanged**; annotation-only reissue.
- **Split experiment**: neither pre-registered outcome — **localised non-stationarity**. In both sessions one half is
  clean (xb4 half B: e-process 0.54 against an alarm line of 20) and the other alarms. xb4's symmetric readouts are
  identical across the split (1 % / 3 %), so **the surface-switch explanation is not supported**; nmb3's foot force
  moves 17 % and the heavier half alarms. The deployment rule sharpens to *per homogeneous segment*.
- **Friction row**: raw-element R⁻ 0.00 → **analytic-residual R⁻ 1.00 at ×1.5, both at FAR 0.00**, versus the classical
  χ² needing FAR 0.50 to reach 0.85. The "classical wins on friction" reading was a comparison against the wrong
  channel.

## 5. Decisions still open before freeze

1. **Title** — three candidates in `paper/notes_title.md`; whether the name "GeoFDI" survives at all.
2. **Split or single manuscript** — `docs/paper/split_option.md` holds the content division; the Sprint 8 weld results
   are the RA-L half's evidence.
3. **Advisor authorship and affiliation** — both placeholders in `main.tex`.
4. Not a decision but a dependency: the abstract and introduction are written around the *absence* of injection
   experiments and will need re-tightening once P1–P7 land.

## 6. Pending on the user

`raw/m1/nominal-crossday/` still awaits the cloud-drive download; two emails pending; the NUC benchmark is tomorrow.
