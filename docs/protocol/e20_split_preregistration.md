# Pre-registration — xb4 / nmb3 split experiment (Sprint 10 M0.2)

Committed BEFORE the run (this file's git timestamp is the pre-registration timestamp). Extends
`docs/protocol/go2_real_preregistration.md`; data and settings as in Block R (`experiments/e20_go2_quadric`, α = 0.05,
N = 64, 10-cycle windows, M = 512, seed 0, H₀′ calibration = first third of the element under test).

## Why these two sessions
Sprint 9 R5 established that H₀′ alarms are usually a *between-run* condition change: pooled over straight runs 8/11
sessions alarm, within a single run only 2/11. **`xb4` and `nmb3` are exactly those two.** They alarm inside one
continuous traverse (xb4: within-run window-reject 0.83, alarm at window 1; nmb3: 0.43, alarm at window 6), so they are
either (a) a *within-traverse condition change* — site A's rough→smooth tile line is the obvious candidate for xb4 — or
(b) a genuine residual anomaly of the machine.

## Splits (fixed here, before looking at any test result)
* **xb4 — spatial split by the RTK track.** The element is cut in two by the median RTK easting of the straight rows;
  each half is registered and tested on its own, and the two halves are also tested concatenated.
* **nmb3 — temporal split.** First half vs second half of the straight rows, same three tests.
* **Within-run control (both):** the session's longest single straight run is itself halved and each half tested, to
  separate "the alarm needs two runs" from "the alarm lives inside one run".

## Predictions
1. **If each half is in band and only the concatenation alarms**, the alarm is a *cross-segment* condition change, and
   the pre-registered P-A statement is corrected to: **"the site-A surface switch is a between-segment condition change,
   not a within-session nuisance the monitor must tolerate"** — i.e. the per-run calibration rule (R5) simply needs to be
   read as *per homogeneous segment*.
2. **If a single half still alarms**, it is a genuine within-traverse anomaly; that session's segment goes into the
   hardware section as a residual-anomaly example, and the deployment rule is *not* sufficient as stated.
3. Sub-prediction for xb4 specifically: if (1) holds, the two halves should also differ in a *symmetric* magnitude
   readout (Π⁺ energy / foot-force level / high-frequency IMU texture), since a surface change is bilaterally symmetric.

## Honest limitation stated in advance
**The "indoor half / outdoor half" labels are not independently verifiable from these recordings.** Checked before
running: for xb4 the GNSS fix-OK fraction is 0.81–0.85 in *every* candidate split (spatial by easting, spatial by
northing, temporal), the reported position variance differs only mildly (33 vs 36 m), and the high-frequency IMU texture
of the two halves is 1.90 vs 1.85 with foot-force spread 4.08 vs 4.19 — i.e. **no channel separates the halves strongly
enough to certify which is indoors.** The split is therefore reported as "spatial half A / half B (by RTK easting)" and
"first half / second half", and any surface-switch interpretation is explicitly labelled as an interpretation, not a
measurement. Falsification of prediction 3 is what would settle it, and a null there means the split is uninformative
about the surface line.
