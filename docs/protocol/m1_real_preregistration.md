# Pre-registration — first GeoFDI runs on the M1 hardware sessions of 2026-08-10 (Sprint 8 Block D3)

Committed BEFORE any run on the real sessions (the git timestamp of this file is the pre-registration timestamp).
Data: the three ingested rolling sessions `raw/m1/nominal/m1_walk_20260810_{172847,173028,173247}` (payload fingerprints
`ab623e3e`, `2566ec71`, `4fc3c926`; audit `docs/protocol/m1_h_data_audit.md`). The standing session `raw/m1/audit/
m1_static_20260810_172037` is not a test session (no motion, asymmetric stand pose) and is used for noise floors only.

## D3.1 — R⁻ under H₀ / H₀′ (`scripts/run_pipeline.sh <session> --robot m1 --mode rolling`)

Fixed settings: N = 64, window = 10 blocks, α = 0.05, M = 512 permutations, seed 0, warm-up 6 s, segmentation =
`straight_mask_kinematic` (thresholds fixed in the audit §10; there is no command signal), data element = the manifest's
in-Z channels minus NaN channels (contacts, tau_cmd), block length **L = 1.0 s first** (protocol default), then the
pre-specified rule: if the lag-1 autocorrelation of the per-block anti-symmetric energy exceeds +0.10 at L = 1 s, the
run is repeated at **L = 2.0 s** and both are reported (no other L is tried; nothing else is tuned).

Predictions (per session; "in band" = the binomial 95 % band of the window rejection rate around α given the number of
windows, or p > α for single tests):
1. **Naive H₀ (whole-session flip test) may reject** — expected on ≥ 2 of the 3 sessions (p_paired_energy < 0.05):
   the healthy hardware loop is stably asymmetric (real ε_dyn: knee 0.02–0.05 rad, wheel rate ≈ 4 %, efforts 0.4–1.1 —
   audit §11) and the blocks are correlated at L = 1 s (Sprint 7 Block W: sim size 0.14 at L = 1, 0.07 at L = 2).
   A rejection here is the *predicted* hardware regime, not a failure of the method.
2. **H₀′ differenced test (first half vs second half) is in band**: p > 0.05 on all three sessions (the asymmetry level
   is stationary within a 70–130 s healthy drive).
3. **Per-window H₀′ (window vs the first-third calibration): no alarm within any session** (e-process max < 1/α = 20)
   and rejection rate in band; the ν₀ estimate is significantly positive (ν₀ / bootstrap-std > 2) on every session —
   the size of ν₀ is recorded for the theory intake (no prediction of its value; the sim rehearsal gave ν₀ ≈ 1.35 at
   K_cal = 15 which is not transferable).
4. Block correlation: lag-1 autocorrelation of the anti-symmetric block energy > 0 at L = 1 s (positive), smaller at L = 2 s.

Falsification: if the differenced H₀′ test rejects on ≥ 2 sessions, or an H₀′ e-process alarm fires within a healthy
session, the "stably asymmetric healthy loop" premise (Sprint 7 W / theory H₀′) fails on this hardware at these block
lengths — reported as such, no parameter is changed after seeing the data.

## D3.2 — rolling InEKF vs fixed-foot RIEKF vs ESKF on hardware (`experiments/e18_m1_real/`, stage `inekf`)

Inputs: central IMU rotated to the body frame and scaled (audit §6; specific force m/s², gyro rad/s), measured wheel
rates (rolling inputs u_i = r ω_i, r = 0.097 m from the audit §7), and — because the vendor joint-angle conventions do
not match the MJCF model (audit §5/§13: the MJCF forward kinematics with hardware angles places the front contact
points above the base; the Day-0 hand test has not been done) — a **declared constant contact geometry**: per-leg
body-frame contact points at the sim crouch footprint (x = +0.35 / −0.27 m front/hind, y = ±0.22 / ±0.24 m) and
z = −0.54 m (the vendor odometry base height). The legs are held in a crouch (joint std < 0.05 rad, audit §4), so the
true contact points move by < 2 cm; a constant offset of the contact geometry enters the filters only through body
rotation (R c ω) and is common to all filters compared. All four wheels are treated as in contact throughout.
Filters (identical noise settings, `experiments/e18_m1_real/config.yaml`): (a) rolling RIEKF (`RollingRIEKF`),
(b) fixed-foot RIEKF (`RIEKF`, the e10 baseline: contact points fixed in the world), (c) fixed-foot ESKF (`ESKF`),
(d) rolling ESKF (`RollingESKF`, same model as (a) in classical error coordinates — added for the parametrisation
comparison). Initialisation from the vendor odometry pose/velocity at t = 0 only.
Reference: `/odom/mc_odom` (vendor odometry, **not ground truth**; its yaw rate is 18–22 % below the gyros, audit §6),
therefore the pre-registered metrics are relative/short-horizon: (i) per straight run (audit §10 runs): displacement
error ‖Δp̂ − Δp_odom‖ over the run and its ratio to the run length; (ii) whole-session horizontal position error vs the
odometry after alignment at t = 0 (RMSE and end-point), reported with the yaw caveat; (iii) estimated path length vs
odometry path length; (iv) yaw drift vs the IMU-integrated yaw over straight runs. No NEES/NIS-vs-truth claim (no truth).
Predictions: fixed-foot RIEKF and fixed-foot ESKF **do not track** (they treat the rolling robot as near-stationary:
whole-session end-point error of the order of the path length, 30–80 m; per-run displacement error ≈ the run length),
the rolling RIEKF tracks (per-run displacement error < 10 % of the run length; whole-session RMSE ≪ path length), and
the rolling ESKF is close to the rolling RIEKF (the parametrisation effect is second order at these rotation rates).
Falsification: if the rolling RIEKF's per-run displacement error is not clearly below the fixed-foot filters' (< half),
the rolling-contact model is wrong for the real M1 — reported as such.

## D3.3 — equivariant DeLaN on hardware: **not run** (pre-declared)

The corpus holds 4.6 min of nominal rolling (153 s of straight rolling); the pre-registered minimum for training the
16-joint equivariant DeLaN is 20 min. Nothing is trained; the residual channel stays `off` in D3.1.

## D3.4 — quantities to carry into the audit / theory intake (no prediction, recorded as measured)

Efforts-semantics observations (audit §8), real ε_dyn candidates (audit §11), ν₀ and its bootstrap band per session,
the lag-1 block correlations at L = 1 / 2 s, and the InEKF per-run displacement errors.
