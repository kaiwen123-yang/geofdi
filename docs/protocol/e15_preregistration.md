# Pre-registration — three prediction experiments for the N1-2 two-layer theorem (Sprint 8 Block P, e15)

Committed BEFORE any e15 run (this file's git timestamp is the pre-registration timestamp). The three experiments test the
three claims that the N1-2 restatement (theory Part 1, `02_n1_theorems.tex`, tag `theory-part1-v1.1`) added or forward-
referenced. Fixed harness: `go2_urdf_sym` (D005 world) and `m1_wheeled_sym`; α = 0.05; deterministic seeds (enumerate
index, never `hash(str)`); paired-energy and energy-distance flip tests as implemented (`detect/permutation.py`); R⁻ has
zero trainable parameters. Results back-fill the P1/P2 empirical-anchor placeholders in `02_n1_theorems.tex` and P3 goes
into the audit as the blindness-as-slip-classifier demonstration.

## P1 — chirality ceiling (amplitude ceiling of the blindness; Remark amplitude-ceiling → Layer I.a)
Stage a. `go2_urdf_sym` closed loop, **bilateral mirror-equal** actuator-gain fault (same κ on LF and RF, applied after a
calibration segment), κ ∈ {0.7, 0.5, 0.4, 0.3} (severity 1−κ ∈ {0.3,0.5,0.6,0.7}), R = 30 seeds each + a nominal (κ = 1)
control. Readouts per κ: (i) R⁻ paired-energy flip-test power (post-onset differenced test, FAR-controlled) and (ii) a
**chirality index** = the stable mirror residual of the realized gait descriptors (per-cycle left/right stance duty
factors and touch-down phases from the foot-contact flags; the A5′ ε_chir quantity), measured on the post-onset segment.
**Prediction:** κ = 0.7 (mild) is blind — R⁻ power ≈ α and chirality index ≈ the nominal floor (this is the already-proved
Layer I.a case). At some heavier κ the symmetric attractor of the faulted loop loses stability: the chirality index rises
off the floor and R⁻ power rises with it. Product: R⁻ power and chirality index vs 1−κ on one figure, with the ceiling
estimate = the smallest 1−κ at which the chirality index exceeds nominal + 3σ AND R⁻ power exceeds the band. Falsified for
Layer I.a if R⁻ rejects at κ = 0.7 with the chirality index still at the floor (blindness fails without a bifurcation), or
for the ceiling claim if the chirality index never rises across the swept range (no ceiling in range — reported, extend
the range in a follow-up, do not tune).

## P2 — statistic split (paired-mean blind, energy-distance visible; Remark statistic-consistency)
Stage b. `go2_urdf_sym`, **zero-mean** left-leg encoder-noise varial inflation: `encoder_noise_scale` on leg = LF,
magnitude = 1.0 (per-joint encoder-noise std ×2 ⇒ variance ×4), applied after the calibration segment; R = 50 seeds +
nominal. The fault has Π⁻μ = 0 (no mean change) but push_ρ P ≠ P (LF and its mirror RF now have different measurement-noise
laws). Readouts: post-onset differenced flip-test power under **paired_energy** and under **energy_distance** on the same
elements, plus the nominal size of both. **Prediction:** paired_energy power ≈ α (blind — Layer II.a: Π⁻μ = 0), energy_
distance power > α (visible — Layer I.b). Product: the two-statistic power comparison + a one-line deployment recommendation
written into the protocol (run both families, alarm on either). Falsified if paired_energy detects it above its nominal
size (Layer II characterization wrong) or energy_distance does not (Layer I.b converse wrong on this instance).

## P3 — slip regimes (blindness theorem as a slip classifier)
Stage c. Three contact regimes × two channels (R⁻ vs InEKF innovation/NIS), R = 30 each, with a nominal control:
- **Go2 unilateral slip** (`foot_friction`, leg = LF, magnitude = −0.5: LF foot friction ×0.5): predict R⁻ RESPONDS
  (one-sided → Π⁻ content).
- **Go2 uniform slip** (`foot_friction`, all legs, magnitude = −0.5): predict R⁻ SILENT (Σ-symmetric → blind, Layer I.a)
  while the InEKF innovation/NIS RESPONDS (the slip is a kinematic-model violation on both sides, seen by the estimator
  channel, not by the invariance channel).
- **M1 single-wheel vs both-wheel slip** (`wheel_friction`, leg = LF magnitude −0.5 vs legs LF+RF magnitude −0.5 on
  `m1_wheeled_sym`, rolling): the isomorphic prediction — single-wheel → R⁻ responds, both-wheel → R⁻ silent + the rolling-
  InEKF rolling-constraint residual responds.
Product: a regime × (R⁻, estimator) reading table — the blindness theorem used as a **slip classifier** (unilateral vs
bilateral slip separated by which channel fires). Falsified if uniform/bilateral slip triggers R⁻ above its nominal size,
or if neither channel responds to a bilateral slip (undetectable — would break the two-channel-necessity corollary).

## Registration
Seeds: P1 seed_base 80000 (+ 1000·round(κ·10)); P2 80500; P3 81000 (+ per-regime offset). Onset after K_cal calibration
cycles/blocks (as e06/e13 — never faulted-throughout, which would cancel the signature by centring). R⁻ uses the analytic/
raw element (no learned model). The chirality index and NIS use only measured quantities. Nothing is tuned after seeing
the data; a falsified prediction is reported as falsified.
