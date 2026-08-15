# Figure and table plan (T-RO manuscript)

Status: ✓ exists (path given), ◐ exists but to be re-rendered/merged for the paper, ✗ to be produced. All results
live under `$GEOFDI_DATA_ROOT/results/<exp>/<run_id>/`; the run ids below are the ones in the review packs.

## Main figures (≤ 10)

| # | figure | content | data source (run id) | status |
|---|---|---|---|---|
| F1 | Concept + architecture | mirror pairing on the trot (Σ = {(e,0),(g_s,½)}); three-channel architecture (Part 2 Fig. fig:architecture) | theory TikZ + a Go2 mirror-pair schematic | ◐ (architecture ✓ in theory PDF; schematic ✗) |
| F2 | Exactness & nuisance invariance | e01a size QQ / KS under H₀; e04b nuisance × channel FAR bars (payload sym/asym, drift, speed, slope single/merged) | e01 `s1-20260815-1422`, e04 `s2-20260815-1652` | ◐ (both exist; merge into one two-panel figure) |
| F3 | Wrapped vs unfolded noise boundary | e04d size vs actuator noise, wrapped vs unfolded (rem:wrap) | e04 `s2-20260815-1652` e04d_noise_sweep.png | ✓ |
| F4 | Power matrix + sequential layer | e04a delay vs magnitude (R⁻ e-process, e-CUSUM, R⁺); e05b blind cells (HFE gain / KFE bias) with rplus_resid | e04 `s2-…`, e05 `e05-20260815-1859` | ◐ (merge) |
| F5 | Isotypic prediction, raw and residual | e04c (raw Z) and e13c (equivariant residual): power per channel × 3 groups + antisymmetric share | e04 `s2-…` e04c_isotypic.png; e13 `e13-20260815-2334` e13c_isotypic_isolation.png | ◐ (e13c ✓ after the run; merge) |
| F6 | Residual vs raw R⁻ at low SNR | e13a power curves (gain / bias / friction × HFE, KFE): raw R⁻, analytic-residual R⁻, equivariant-DeLaN R⁻, Mahalanobis ref. | e13 `e13-20260815-2334` e13a_power_curves.png | ✓ (after the run) |
| F7 | Contamination and size vs δ_f | e13b size of the residual flip test vs δ_f^{(0.95)} (plain ladder vs equivariant; naive centring; H₀′ differenced), K = 60 / 200; inset: Block Q δ_f ladder | e13 e13b_size_vs_defect.png; `results/delan_ladder/q-20260815/delta_f_ladder.png` | ✓ (after the run) |
| F8 | Isolation | e04e ranking (swing conditioning) + e05c payload-vs-fault plane + e06 DK certificate vs confusion (accuracy vs β²/β²_thr) + e13c confusion (equivariant rows) | e04 e04e_isolation_ranking.png; e05 e05c_joint_reading.png; e06 `e06-20260815-2003` e06iii_isolation_vs_dk.png; e13c | ◐ (merge, choose 3 panels) |
| F9 | Baselines at unified FAR | e07: detection rate / delay per detector on the e04a grid + nuisance rows (GRU seen/unseen type, AE, Mahalanobis, ours) | e07 `e07-20260815-final` | ✓ (re-render as a compact heat table) |
| F10 | InEKF CFAR channel (or appendix) | e02 NIS bins / noise stratification / fault-signature geometry | e02 `s3-20260815-1734` | ◐ (choose one panel; rest to appendix) |
| — | Hardware | Go2 trot Gate 1/1b + nominal FAR; M1 rolling R⁻ size/power (e13d design) | ⌂ | ✗ |

## Tables

| table | content | source | status |
|---|---|---|---|
| T1 nuisance table | nuisance × channel FAR (per test / alarm fraction, binomial band) incl. residual R⁺ under drift | e04b `e04b_nuisance_far.csv`; e05a `e05a_far.csv` (+ K_cal 200) | ✓ (merge) |
| T2 blind-spot table | Σ-invariant faults (bilateral equal) vs single / unequal: R⁻ window rejection in band, R⁺ detects — raw and residual | e04c `e04c_isotypic_power.csv`; e13c `e13c_isotypic_power.csv` | ✓ (after e13) |
| T3 baseline table | unified-FAR protocol: det100/det20/median delay per detector; nuisance rows; naive-threshold FAR column | e07 tables | ✓ |
| T4 minimal detectable magnitude | smallest severity with det100 ≥ 0.9 per (fault type, joint, detector): raw R⁻ / analytic residual / equivariant residual / Mahalanobis / R⁺ | e13a `e13a_min_detectable.csv` | ✓ (after e13) |
| T5 e03 external table | Liu A1 dataset: four classes × detectors at equal FAR | e03 | ✗ |
| T6 ε-budget / falsification table | A1–A5 + ε̄_model rows with signals and degradation paths | theory Tables tab:degradation, tab:degradation2 | ✓ |
| T7 DeLaN ladders | val RMSE / β̂ / δ_f^{(0.95)} plain vs equivariant, four sample sizes (+ seed replicates, weld) | `results/delan_ladder/q-20260815/ladder.csv` | ✓ |
| T8 variance decomposition | Var(Π⁻ τ_cmd) vs Var(Π⁻ r) per torque row; SNR ratio per grid cell | e13a `e13a_variance_decomposition.csv`, `e13a_snr.csv` | ✓ (after e13) |
| T9 isolation confusion (three variants) | raw+track / analytic rows / equivariant rows accuracy + confusion | e13c `e13c_confusion_*.csv` | ✓ (after e13) |
| T10 M1 model audit / hardware manifest | joint conventions, wheel DoFs, sign table (Part 0 back-fill) | docs/protocol/m1_model_audit.md, legacy_aug_inventory.md | ◐ |

## Rules
- Long edge ≤ 1600 px for pack figures; the paper versions are re-rendered vector (PDF) from the CSVs.
- Every figure's data source is a CSV in a review pack; no figure is produced from a run that is not in a pack.

## Regeneration (Block F — `scripts/make_paper_figures.py`)

One-shot factory: `GEOFDI_DATA_ROOT=... python scripts/make_paper_figures.py [--check] [--only T1,F6b]`.
Writes `$GEOFDI_DATA_ROOT/results/paper/{tables/*.csv, figures/*.pdf, paper_tables.tex, coverage.md}`. The run-id map
lives in the script (`RUN`) and matches the review-pack run ids above; `--check` verifies every source CSV exists.

- Tables **T1–T5, T7–T9 generated** from the pack CSVs (paper-ready CSV + a booktabs `paper_tables.tex`); T9b adds the
  e09 three-channel isolation accuracy. **T6, T10 are prose/theory tables** (registered, authored in LaTeX/markdown).
- Figures **F4b (ARL–delay), F6b (low-SNR power), F10b (N2 signatures) regenerated as vector PDF** from CSV; the
  multi-panel merges (F2, F3, F5–F10) are **registered** with their source run PNG — the manuscript composes the vector
  versions by hand from the same CSVs. New Sprint-7 panels registered: **F11 (e09 confusion), F12 (e11 complementarity)**.
- Coverage table (every id → status) is regenerated at `results/paper/coverage.md` on each run.
