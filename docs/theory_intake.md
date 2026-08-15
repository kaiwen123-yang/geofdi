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
  - §6 anchors table (pending markers, back-filled after Block R).
- New verified bib entries: lutter2019delan, cohen2016group, weiler2019general, finzi2021practical, deluca2003actuator,
  yu2015daviskahan, vovk2021evalues (publisher/PMLR/OpenReview pages checked 2026-08-15).

## Owed / open
- Full Part 1 (N1-1 general form with the unfolded shift and exchangeability conditions, N1-3 e-processes,
  N1-4 IPM refinement of Lemma lem:eps and of ε̄_model — the TV modulus c is the same object as Part 0 Remark rem:linear's
  metric change).
- Part 0 M1 back-fill (joint count/order/signs from `docs/protocol/legacy_aug_inventory.md` §4a): still `\todo`.
- e13d (wheeled M1 residual channel) needs the M1 world (Sprint 5 Block W, absent).
- Second-order (variance) asymmetry of non-equivariant models after centring: quantify (Lemma centring (iii)).
