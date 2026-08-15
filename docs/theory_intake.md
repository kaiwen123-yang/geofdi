# Theory intake log

Running list of what enters the theory notes (`theory/`), from which sprint, and what is still owed. Created in
Sprint 6 (2026-08-15) — the file was referenced by the sprint instructions but did not exist in the repo before.

## State before Sprint 6
- Part 0 (`00_notation.tex`, `01_assumptions.tex`, appendix counterexample): H0, A1–A5, ε-lemma with cycle-level
  defects, H0′, audit checklist. Tags `theory-part0-v1`, `-v1.1`, `-v1.2`.
- Part 1 (workstreams N1-1…N1-4: unfolded shift, exchangeability, e-processes, IPM route): **not present** in the repo
  (the Sprint 3/5 theory track never landed here; no `02_*.tex`, no Sprint 5 commits, no `run_pipeline.sh`).

## Sprint 6 (2026-08-15) — Part 1 core stand-in + Part 2 (tag `theory-part2-v1`)
- `02_part1_core.tex` — **stand-in**: Theorem N1-1 (exactness of the Hemerik–Goeman flip test under H0 + independence)
  and Theorem N1-2 (isotypic detectability: Π⁻μ = 0 invisible, Π⁻μ ≠ 0 consistent, general alternatives iff
  ρ-invariant law), both with proofs. Labels `thm:n1-1`, `thm:n1-2` are the ones Part 2 cites; the full Part 1 must keep
  them. Marked with an inline `\todo`.
- `03_equivariant_residuals.tex` — Part 2:
  - Def. equivariant nominal model, defect δ_f (sup) and δ_f^{(q)} (quantile; the measured one), residual data element
    R_k, ρ_R action, Π± split; Remark: model-free channel = f̂ ≡ 0; Remark: causal generators (observer memory).
  - Prop. N3-1 residual inherits H0 (proof) → N1-1/N1-2 verbatim on R_k; Cor. unified two-channel structure (R⁻ on
    Π⁻r, R⁺ on Π⁺r; corrects the raw/residual asymmetry); Cor. non-equivariant model contaminates H0: ε̄_model as fifth
    ε-term, ε̄_model ≤ c·δ_f with c the TV-modulus (Gaussian case c ≤ √N/(s√2π)); quantile version ≤ N(1−q) + c δ_f^{(q)}.
    Falsification: A1 fails → H0′ on the residual; anchor e13b.
  - Prop. N3-2 power gain: SNR_r/SNR_y ≥ min_c Var(Π⁻y)_c/Var(Π⁻r)_c ≥ 1 under (a) fault–model commutation (commanded
    torque; gain/bias/friction) and (b) variance removal; (c) conditional general form; Remark on when the residual is
    weaker (closed-loop response, faults outside the torque balance, channel count). Anchor e13a.
  - §4 learned model: Lemma calibration centring (systematic error out of β_op; first-order removal of contamination for
    R⁻, second-order/K/K_cal caveats); Prop. equivariant DeLaN by mirror weight sharing (exact C₂ equivariance, SPD/
    Lagrangian structure inherited, two templates for front/hind; single template = ablation); Remark exact zero in
    floating point (S, E signed permutations); Remark why no soft PINN penalty.
  - §5 ε-budget table with the ε̄_model row; TikZ three-channel architecture (learned vs zero-parameter components).
  - §6 anchors table — back-filled from e13 (run e13-20260815-2334) and Block Q; all four rows **confirmed** (e13d n/a).
  - Lemma centring revised after e13b: (iii) naive mean-profile centring gives size ≈ 1 (factor (K_cal+K)/(K_cal+1) shift vs O((d_rN)^-1/2) spread); (iv) the H0′ route is differencing monitored vs calibration cycles (exact, half SNR).
- New verified bib entries: lutter2019delan, cohen2016group, weiler2019general, finzi2021practical, deluca2003actuator,
  yu2015daviskahan, vovk2021evalues (publisher/PMLR/OpenReview pages checked 2026-08-15).

## Owed / open
- Full Part 1 (N1-1 general form with the unfolded shift and exchangeability conditions, N1-3 e-processes,
  N1-4 IPM refinement of Lemma lem:eps and of ε̄_model — the TV modulus c is the same object as Part 0 Remark rem:linear's
  metric change).
- Part 0 M1 back-fill (joint count/order/signs from `docs/protocol/legacy_aug_inventory.md` §4a): still `\todo`.
- e13d (wheeled M1 residual channel) needs the M1 world (Sprint 5 Block W, absent).
- Second-order (variance) asymmetry of non-equivariant models after centring: quantify (Lemma centring (iii)).

## Sprint 7 Block 0 (2026-08-16) — intake for Block T

**From the Liu et al. paper (real-robot section):** the A1 hardware runs show motor aging and IMU drift, a persistent
lateral deviation during in-place stepping with zero command, and the authors state that their threshold-based
detector misses the degradation because it is "too subtle to fall below the detection threshold". Reading for the
theory: (i) a stable nonzero asymmetry of a healthy-but-aged robot is exactly the H₀′ scenario (A2/A5 failing on
hardware: nominal asymmetry level ν₀ > 0 to be calibrated, faults as changes of asymmetry); (ii) "subtle degradation"
is the low-SNR regime of Block P where the residual R⁻ (Part 2, N3-2) and the sequential layer are supposed to win over
threshold rules; (iii) the released dataset is Gazebo simulation (Part 0 audit table: e03 is an external *simulation*
benchmark, hardware anchors stay with our own Go2/M1 sessions).

**Sprint 6 leftovers (Block T to absorb):**
1. **Corollary N3-3 candidate — equivariant model error is Π⁺-only.** For an equivariant nominal model the systematic
   (gait-locked) model error m̄(θ) is itself an equivariant function of the nominal trajectory; on a Σ-symmetric nominal
   trajectory it therefore satisfies m̄ = ρ_R m̄, i.e. Π⁻m̄ = 0 pointwise in phase: the mirror pairing cancels it
   exactly and only the fluctuating part of the error remains in Π⁻r. Consequence: R⁻ on the residual is insensitive
   to the model's *quality* (only to its equivariance), while R⁺ / the DK certificate pay for it through β_op.
   Evidence (e13a): Rminus_res_eq ≈ Rminus_res_an at every grid cell (min-detectable bias 0.10 / gain 0.02 / friction
   ×1.5 for both) whereas Rplus_res_eq_full ≪ Rplus_res_an_full (min-detectable gain 0.05 vs 0.02, bias 0.20 vs 0.10);
   the DeLaN residual's Π⁻ variance is 3× the analytic one and its power at the smallest cells is lower — the residual
   *fluctuation* still costs, the systematic error does not.
2. **Contamination saturates on the best plain model.** e13b: the plain DeLaN with δ_f^{(0.95)} = 0.67 N·m (δ_f^{(0.5)}
   0.21, val RMSE 0.42) already gives size 1.00 for a 10-cycle residual flip test; there is no "small enough" δ_f for an
   energy statistic over d_r N coordinates → equivariance is a *necessary* condition for the residual R⁻ (Remark to add
   to Cor. contamination).
3. **Centring trap** (Lemma centring (iii)/(iv) already in Part 2 v1 text after e13b): subtracting the calibration mean
   profile and re-running the exact flip test has size ≈ 1 for every model at K ≥ 60 (factor (K_cal+K)/(K_cal+1) shift vs
   O((d_r N)^{-1/2}) spread); the valid H₀′ construction differences monitored vs calibration cycles. → protocol_params
   entry (Block 0 item 10).
