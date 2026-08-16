# Pre-registration — public real-robot data harvest (Sprint 8 Block PUB, e17)

Committed BEFORE any e17 run (git timestamp = pre-registration timestamp). Datasets: (1) MIT Mini Cheetah contact
dataset (UMich-CURLY/deep-contact-estimator, 8 terrains + air gaits, 1000 Hz, q/dq/τ_est/IMU/foot pos-vel/contact
labels), to be downloaded and ingested to `raw/public/minicheetah-contact/`; (2) the already-ingested Cerberus Street A1
(`raw/public/street-a1`) and Leg-KILO Go1 (`raw/public/legkilo-go1`, 7 sequences). α = 0.05; R⁻ has zero trainable
parameters; deterministic seeds; the raw C2 element (model-free) unless a residual model is stated.

## PUB2 — Mini Cheetah (8 terrains)
1. **Cross-terrain R⁻ H₀′ FAR table** (the A3 environment-invariance claim on real hardware, the main table): on each of
   the 8 terrains, a straight-trot segment is cut, registered to cycles, and the R⁻ flip test (naive H₀ and the H₀′
   differenced/per-window test) is run. **Prediction:** naive H₀ may reject (real ε_dyn + block correlation, as on the M1
   hardware); the H₀′ per-window test is in band (FAR ≈ α) on every terrain — the asymmetry level is a robot property,
   invariant across terrain (A3). Falsified if the H₀′ FAR is out of band on a terrain whose gait is symmetric trot.
2. **Real-robot residual R⁻**: the dataset carries τ_est, so an analytic inverse-dynamics residual is available (model:
   MIT Mini Cheetah URDF if a usable one is found, else the model-free R⁻ on the raw element + the τ_est channel —
   choice recorded). Prediction: the residual R⁻ H₀′ is in band on nominal trot (as the analytic residual is on the sim).
3. **Air sequences = leg-in-air nominal**: the air trotting/walking/jumping/pronking gaits have no ground contact — the
   weld/leg-as-arm regime (Sprint 8 L3 world, RA-L material). Run the pipeline; prediction: R⁻ H₀′ in band (a mirror pair
   swinging in the air is Σ-symmetric).
4. (optional) K₄ demonstration on the front/hind symmetric groups if time allows.

## PUB3 — Street A1 / Leg-KILO Go1
1. **Straight-segment mining**: the phase estimator (`phase/estimator.py`) + a yaw-rate filter cut straight steady trot
   segments from the Street A1 bag and the Leg-KILO sequences (corridor/park/slope/grass have long straight runs). Real
   R⁻ H₀′ FAR one figure each (Street, legkilo). Prediction: naive H₀ may reject, H₀′ in band on the straight segments.
2. **Leg-KILO three-estimator gating comparison (real)**: the e16 pi-gating comparison (no gating / threshold / GeoFDI
   hard+soft) on a Leg-KILO sequence, using the dataset's ground truth (the `*_tum.txt` for corridor/indoor/park/running)
   as the reference; where the sequence has a slip-prone stretch (grass/slope), the gate should down-weight it. Reference
   priority: dataset ground truth; absent → `pip install kiss-icp` on `/points_raw` (VLP-16), config recorded. Prediction:
   GeoFDI-πᵢ per-event nominal FAR ≈ α where the fixed 0.4 m/s threshold false-rejects (as in the e16 sim).

## PUB4 — evidence stack
`docs/data_catalog.md` + the paper outline updated with the five-layer evidence structure (sim controlled / sim
prediction / M1 own hardware / public multi-platform / public multi-terrain).

## Registration
Seeds: e17 mining seed_base 84000. R⁻ settings as run_pipeline (N=64, window=10, M=512). No parameter is tuned after
seeing the data; a falsified prediction is reported as falsified. Reference topics and any KISS-ICP config are recorded
in the session meta.yaml / the audit.

---

## B6 addendum (Sprint 9) — block mode does **not** rescue H₀′ on the Mini Cheetah flying trot

Sprint 8 found the cycle-level H₀′ elevated on this dataset and attributed it to the 0.25 s flying trot being
non-stationary cycle-to-cycle plus phase-registration stress. B6 tested the obvious remedy: replace phase registration by
**rolling-style fixed-duration time blocks**, as the wheeled M1 mode does (`experiments/e17_public_realdata/block_mode.py`).

Construction (the part that needed care): for a trot the symmetry is σ\* = (mirror, shift by T/2), so the pure mirror is
*not* a symmetry of a time-aligned block — a block element must be **phase-free**. We use the exactly-equivariant
first-two-moment summary of each block: per channel `mean` (which carries the channel's mirror sign) and `std` (a
magnitude, sign +1), with `delta_theta = 0`. No phase estimate enters anywhere.

| terrain | cycle mode: K, H₀′ win-rej, alarm | block mode (L = 1 s): K, H₀′ win-rej, alarm, #windows |
|---|---|---|
| asphalt_road | 91, 0.00, — | 24, 0.00, —, 1 |
| concrete_difficult_slippery | 203, 0.23, — | 55, 0.00, —, 3 |
| forest | 261, 0.29, alarm | 66, 0.25, —, 4 |
| grass | 223, 0.64, alarm | 60, 0.50, alarm, 4 |
| middle_pebble | 135, 0.67, alarm | 35, 0.50, —, 2 |
| rock_road | 104, 0.29, alarm | 27, 1.00, —, 1 |
| sidewalk | 195, 1.00, alarm | 50, 1.00, alarm, 3 |
| small_pebble | 179, 0.50, alarm | 46, 0.00, —, 3 |
| **summary** | alarms **6/8**, median win-rej **0.397** | alarms **2/8**, median win-rej **0.375**, 1–4 windows |

**Verdict: no.** The sequential alarm count drops from 6/8 to 2/8, but that improvement is **confounded**: the straight-trot
yield (25–70 s per terrain) supports only 1–4 monitoring windows in block mode against 7–20 in cycle mode, and an
e-process over four windows simply has less opportunity to cross 1/α. The fair, window-count-free comparison is the
window-reject **rate**, and it barely moves (0.397 → 0.375, still far above α). Naive H₀ continues to reject on 8/8, so
the detection channel is unaffected either way.

**Conclusion, as the fallback in the sprint plan anticipated: this is an aggressive-gait limitation, not a registration
artefact.** On a 0.25 s flying trot with a flight phase the healthy asymmetry level is genuinely not stationary across the
record at the resolution these segments allow — neither phase registration nor time blocking makes H₀′ hold. The two
constructions that *do* hold on real hardware are the slower walking trots (Leg-KILO Go1, 4/5 in band) and the wheeled M1
rolling mode. Recorded as a stated scope limit of the H₀′ layer.
