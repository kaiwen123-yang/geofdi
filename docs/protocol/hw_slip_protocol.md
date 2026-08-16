# Hardware slip experiment protocol (Sprint 8 Block G)

Purpose: a repeatable on-robot procedure to collect the three slip regimes that the blindness theorem classifies
(nominal / unilateral / bilateral slip) and to validate the πᵢ-gated estimator (`estimate/pi_gating.py`) against the
no-gating and literature-threshold baselines on real data. Everything goes through `scripts/ingest_session.sh`; each
session gets a pre-registration (template §5) committed before the run.

## 0. What the sim predicts (e15 P3, the design target)
On `go2_urdf_sym` and `m1_wheeled_sym` (Block P): **unilateral** slip → R⁻ responds (power 1.0), the estimator NIS is
quiet; **bilateral/uniform** slip → R⁻ silent (blind, Σ-symmetric), the estimator NIS responds. So the two channels
together classify the slip. On hardware we test the same table and, additionally, that πᵢ gating removes the estimator
corruption from a slipping foot with **zero nominal false-rejections** (the FAR guarantee), where the 0.4 m/s foot-speed
threshold false-rejects at touch-down/lift-off transients.

## 1. Materials
- **Low-friction patches**: 300 × 300 mm acrylic (PMMA) sheet and 1.5 mm PTFE (Teflon) sheet; PTFE adhesive tape
  (25 mm) for a repeatable single-foot patch. Coefficient of friction on the foot rubber: dry floor μ ≈ 0.8–1.0,
  acrylic μ ≈ 0.3, PTFE μ ≈ 0.05–0.1. A light water film on acrylic lowers μ further and is repeatable.
- **Fixture**: a taped lane (dry) with a removable patch zone (unilateral: patch only under the left track; bilateral:
  patch spans the whole width). Mark the patch entry so the slip onset time is known.
- **Go2 (outdoor/large indoor)**: Fixposition Vision-RTK2 (VRTK2) as the position reference — the QUADRIC-GINS VRTK2 bag
  format is already known to the repo (`legacy_aug_inventory.md`); record `/fixposition/odometry` (or the vendor topic)
  at ≥ 100 Hz with RTK fix, base station or NTRIP. Log the RTK fix status; drop non-fixed spans from the reference.
- **M1 (indoor)**: the vendor `/odom/slam_odom` (SLocalization) as the reference **if it is populated** — on the
  2026-08-10 corpus it was empty (`m1_h_data_audit.md` §3), so **enable and verify SLocalization before recording**;
  fallback reference = `pip install kiss-icp` on the recorded `/front_lidar` + `/rear_lidar` point clouds (record the
  KISS-ICP config and drift). `/odom/mc_odom` (wheel odometry) is NOT a slip reference — it is derived from the wheel
  rates and is blind to wheel slip by construction.

## 2. Recording (both robots)
Record everything the pipeline needs plus the reference: joint states (q/dq/effort, ≥ 200 Hz), IMU (≥ 200 Hz), command
(`/cmd_vel`), the reference pose topic, and the point clouds (M1, for the KISS-ICP fallback). Verify `/cmd_vel` is
actually published this time (it was empty on 2026-08-10) so straight segments are cut on the command, not the kinematic
fallback. Clock: use header stamps; check the header−receive skew as in the audit tooling.

## 3. Three-segment session design (per robot, per speed)
| segment | surface | what | reps | onset marker |
|---|---|---|---|---|
| N (nominal) | dry lane | straight rolling/trot, no patch | ≥ 10 × 20 s | — |
| U (unilateral) | left-track patch | straight over the patch; the LEFT feet/wheels slip | ≥ 10 × 20 s | patch-entry time (from the command / a floor mark seen in the lidar) |
| B (bilateral) | full-width patch | straight over the patch; BOTH sides slip | ≥ 10 × 20 s | patch-entry time |
Speeds: Go2 trot 0.3 / 0.5 m/s; M1 rolling 0.5 / 1.0 m/s. Keep the command constant across the patch (do not correct the
heading manually — a heading correction is a Σ-breaking command and confounds the blindness read).

## 4. Analysis (already implemented)
1. R⁻ H₀′ per segment via `run_pipeline.sh` (rolling/trot) — expect: N in band; U rejects (one-sided); B in band
   (blind) — the hardware version of the e15 P3 table.
2. Estimator comparison via `experiments/e16_pi_gating` on each session with the reference pose: no-gating / threshold
   (0.4 m/s + cov ×10) / GeoFDI-πᵢ (hard and soft). Metrics: position RMSE and end-point error vs the reference, NEES,
   per-leg gate-trigger timeline. Expect: on N, the threshold method has nonzero false-rejection (foot-speed spikes at
   lift-off) while GeoFDI-πᵢ has ≈ α per-event FAR (zero spurious drops); on U, both gates beat no-gating and GeoFDI is
   at least as good with fewer nominal false rejects.
3. Per-stance-event FAR (`detect/stance_event.py`) on the nominal segment must be ≈ α (the G gate).

## 5. Pre-registration template (commit before each recording day)
```
# Pre-registration — hardware slip session <robot> <date>
Data: raw/<robot>/slip/<session>.  Reference: <VRTK2 fix | SLocalization | KISS-ICP cfg>.
Predictions: N R⁻ in band; U R⁻ rejects + estimator NIS quiet; B R⁻ in band + estimator NIS responds.
Gating: threshold false-rejects on N (>0); GeoFDI-πᵢ per-event FAR ≈ α on N; both reduce U pose RMSE vs no-gating.
Falsification: U R⁻ in band (one-sided slip invisible) or B R⁻ rejects (bilateral slip visible to R⁻) on a world
satisfying A1–A5; GeoFDI-πᵢ FAR > 2α on nominal.  Fixed: α=0.05, 0.4 m/s threshold, cov×10, library = the N segment.
```
## 6. Safety / repeatability
Tether or spotter on the slip patch; start at the lowest speed; wipe/redry the lane between reps; log patch material and
any water film. The patch onset time and material go into `meta.yaml`.
