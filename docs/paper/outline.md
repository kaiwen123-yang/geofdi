# GeoFDI — paper outline (T-RO long paper, single manuscript; not split)

Working title: **Symmetry as the Nominal Model: Distribution-Free Fault Detection and Isolation for Legged Robots
from Morphological Symmetry, with Equivariant Residuals**

Conventions used below: `eNN` = experiment id under `experiments/`, `rpNNN` = review pack in `review/outbox`,
`Part k` = theory notes `theory/sections/`. Status tags: ✓ done (sim), ◐ partial, ✗ pending, ⌂ hardware placeholder.
The section list follows the sprint instruction; each section names the experiments and packs it draws on.

---

## Abstract (one paragraph)

A healthy, morphologically symmetric legged robot executing a symmetric gait produces phase-registered per-cycle data
whose law is invariant under the gait's spatio-temporal symmetry group Σ ⊂ G × S¹. We turn this invariance into a
fault-detection null hypothesis H₀ that needs no fault model and no training: an exact, distribution-free permutation
(sign-flip) test on mirror-paired cycles (R⁻ channel), made anytime-valid by e-processes, detects faults as symmetry
breaking; its structural blind spot — Σ-invariant faults (bilateral, synchronized) — is characterized by the isotypic
decomposition and covered by a magnitude channel (R⁺). We then make the nominal model part of the same picture: an
equivariant nominal model (analytic momentum observer or a Deep Lagrangian Network made exactly equivariant by mirror
weight sharing) passes H₀ from the data to the residual, so R⁻ and R⁺ become the sign and trivial isotypic components
of one residual; a non-equivariant learned model contaminates H₀ by a fifth ε-term controlled by its equivariance
defect, and the residual channel dominates the raw channel in power under an explicit SNR condition. Isolation uses a
three-channel reading (R⁻ pair–joint ranking, joint residual rows, floating-base rows) plus a Davis–Kahan isolability
certificate. Simulation (Go2, MuJoCo, URDF-exact world) verifies exactness, nuisance invariance, the isotypic
predictions, low-SNR power, and the contamination/size claim; [hardware: Go2 trot, M1 wheeled-legged — placeholder].

## 1. Introduction

- Problem: proprioceptive FDI on legged robots without fault models, without fault data, with a false-alarm guarantee
  that survives nuisances (speed, payload, terrain, drift).
- Gap: model-based residual methods need an accurate nominal model and are nuisance-sensitive; learning-based methods
  need fault data and offer no FAR guarantee; symmetry has been used as an inductive bias for *models*
  (Ordoñez-Apraez et al.) but not as the *null hypothesis of a test*.
- Four contributions:
  1. H₀ as distributional Σ-invariance of the per-cycle data element; A1–A5 with ε-variants, additive ε-robustness of
     the level, H₀′ recalibrated null (Part 0; rp000–rp004).
  2. Exact flip test + e-process/e-CUSUM sequential layer; isotypic characterization of detectable vs invisible faults
     (Part 1 core; e01, e04; rp003, rp005).
  3. Equivariant nominal models: residual inherits H₀ (N3-1), ε̄_model for non-equivariant learners, power gain N3-2,
     equivariant DeLaN by weight sharing (Part 2; e13, Block Q; rp011–rp013).
  4. Three-channel isolation with the DK isolability certificate and a unified-FAR baseline comparison (e05, e06, e07;
     rp008, rp009).

## 2. Related work (three paragraphs)

1. **Legged FDI and FTC.** Liu et al. RA-L 2025 (A1 joint partial failure dataset; GRU classifier), FT-Net style
   fault-tolerant policies, RL-based fault-tolerant locomotion; joint-level actuator fault detection with learned
   classifiers. Contrast: they need fault data / fault classes; we test a symmetry null. (e03 head-to-head is the
   external anchor; e07 baselines under one FAR protocol.)
2. **Symmetry in robotics and detection.** Morphological symmetries (Ordoñez-Apraez et al. RSS 2023, IJRR 2025) as
   algebraic structure on data and models; equivariant networks (Cohen–Welling, Weiler–Cesa, EMLP); equivariance /
   transformation-based conformal OOD detection (iDECODe, CODiT [verify citations]); gait symmetry theory (Golubitsky
   et al., Collins–Stewart); biomechanical gait symmetry indices (Robinson SI, Sadeghi review [verify]). Contrast: we
   make the group action the null hypothesis of an exact test on cycle elements, with an ε-budget and a sequential
   layer.
3. **Model-based residuals and learned dynamics.** Generalized-momentum observers (De Luca–Mattone; Haddadin, De Luca,
   Albu-Schäffer survey), physically consistent GP momentum observers (Evangelisti–Hirche), Deep Lagrangian Networks
   (Lutter, Ritter, Peters), contact-aided InEKF (Hartley et al.; Barrau–Bonnabel). Contrast: our residual is read by
   an invariance test on its sign component; the nominal model must be equivariant, and we show what happens when it
   is not.

## 3. Preliminaries (Part 0 §1–2)

Groups and representations (ρ_X, ρ_U, ρ_Y, ρ_R; polar/axial rules; joint signs), gait phase and the per-cycle data
element Z_k, the spatio-temporal group Σ (trot: (g_s, ½)), mirror-pairing rule, wrapped vs unfolded elements
(Remark rem:wrap; e04d), assumptions A1–A5 with ε-variants and audits.

## 4. Theory (Parts 0/1/2) — theorem list and dependency graph

| id | statement | part | anchor |
|---|---|---|---|
| Def H₀, H₁, H₀′ | distributional Σ-invariance; structured alternatives; asymmetry-change null | 0 | e01a/e01c (rp003) |
| Lemma ε-robust level | d_TV(Law ρZ, Law Z) ≤ ε̄_tot; level ≤ α + ½ε̄_tot | 0 | e01c, conj:quadratic |
| A5 / bifurcation remark | unique symmetric attractor needed; chiral trot at kp=60 | 0 | S1 Finding 1 (rp003) |
| Thm N1-1 | exactness of the HG flip test under H₀ | 1 (core) | e01a, e04d |
| Thm N1-2 | isotypic detectability: Π⁻μ ≠ 0 visible, Σ-invariant faults invisible | 1 (core) | e04c (rp005) |
| Prop N3-1 | residual inherits H₀; N1-1/N1-2 verbatim on R_k; Cor two-channel; Cor contamination (ε̄_model ≤ c δ_f) | 2 | e13b, e13c |
| Prop N3-2 | SNR_r/SNR_y ≥ 1 under commutation + variance removal | 2 | e13a |
| Lemma centring | β_op independent of systematic error; naive centring breaks the flip test; H₀′ differenced test exact | 2 | e06 (rp008), e13b |
| Prop equivariant DeLaN | weight sharing → δ_f = 0, SPD/Lagrangian inherited | 2 | Block Q (rp012) |
| DK certificate (N3) | isolable iff gap > c_DK β_op | (Sprint 4 notes) | e06 (rp008) |

Dependency graph: A1–A5 → H₀ (Def) → N1-1 (exact test) → sequential layer (Ville) ; H₀ + isotypic split → N1-2 →
two-channel design ; H₀ + equivariant f̂ → N3-1 → residual R⁻ / R⁺ ; N3-1 + defect → ε̄_model → ε-lemma (5 terms) ;
N3-2 (power) uses N1-2(ii) ; centring lemma → DK certificate + H₀′ on residuals ; equivariant DeLaN → N3-1 applies to
the learned model.

## 5. Method

- **Three-channel architecture** (Part 2 Fig. fig:architecture): telemetry → phase registration → nominal model f̂
  (analytic Pinocchio observer under A1, or equivariant DeLaN) → residual element R_k → (i) R⁻ flip test on Π⁻R_k,
  (ii) joint rows: conformal magnitude on Π⁺R_k + per-leg deviation, (iii) floating-base rows (payload); model-free
  path f̂ ≡ 0. Only f̂ is learned; test/calibration/sequential/isolation have zero trainable parameters.
- **Equivariant DeLaN by mirror weight sharing** (`geofdi.dynamics.delan_equiv`): one Lagrangian template per leg
  pair, mirrored leg = S f₀(Sq, Sq̇, Sq̈, Ea); δ_f = 0 exactly (also in floating point); Block Q ladder (rp012).
- **Sequential layer**: window p → e = p^{-1/2}/2 → e-process (Ville) and FAR-calibrated e-CUSUM (h from nominal
  windows); protocol_params.md (window discreteness: 5-cycle windows have 16 sign patterns).
- **Isolation rules**: pair–joint from the nominal-scaled R⁻ projection energy with swing conditioning (e04e), left/right
  from the joint residual rows (e05b/e13c), payload from the base rows (e05c Finding 2), DK certificate for type (e06).
- **Nuisance handling**: out-and-back paired blocks (A3), calibration discipline (nominal cycles only), H₀′ differenced
  test when the model is not equivariant (Part 2 Lemma centring (iv)).

## 6. Simulation study (Go2, MuJoCo 3.11, go2_urdf_sym world; D004/D005)

Four main figures (see figure_plan.md) + low-SNR + residual comparison:
- **Fig A** exactness & nuisance table: e01a size QQ (rp003), e04b nuisance × channel FAR table (rp005), e04d
  wrapped-vs-unfolded noise boundary.
- **Fig B** power matrix & sequential layer: e04a delay vs magnitude, e-CUSUM vs e-process (rp005); e05b blind cells
  (rp008).
- **Fig C** isotypic prediction: e04c three groups (raw) and e13c (residual): power per channel + anti share.
- **Fig D** equivariant residual: e13a residual vs raw R⁻ power curves + minimal detectable magnitude table + variance
  decomposition; e13b size vs δ_f (contamination, centring, H₀′ differenced); Block Q δ_f ladder.
- Isolation: e04e ranking, e05c payload vs fault, e06 DK certificate vs confusion (analytic / DeLaN / weld), e13c
  three-variant confusion.
- Baselines: e07 unified-FAR table (GRU seen/unseen, AE, Mahalanobis) (rp009).
- Estimation channel: e02 InEKF CFAR NIS bins / stratification / fault-signature geometry (rp006) — one figure or
  appendix.

## 7. External benchmark — Liu et al. A1 fault dataset (e03) ✗

Four-class table (nominal + the dataset's fault classes; sim-vs-hardware provenance unresolved, README rate mismatch,
GRU baseline to be re-implemented — docs/protocol/liu_a1_audit.md). Reading planned: R⁻ window rejection / e-CUSUM
delay per class vs the GRU at equal FAR; the dataset has no phase signal → phase estimation from joint kinematics is a
prerequisite (workstream N1/N3 phase estimator).

## 8. The five-layer evidence stack (Sprint 8)

The claims are supported at five levels of increasing externality; each layer answers a distinct objection.

**L1 — Controlled simulation (Go2, MuJoCo, go2_urdf_sym).** Exact-level flip test, isotypic power, residual
inheritance, sequential layer, three-channel isolation (§6; e01/e04/e08/e09/e11/e13; rp003–rp024). The world is
mirror-exact by construction, so this layer isolates the *statistical* claims from modelling error.

**L2 — Pre-registered predictions (e15, Block P; rp028).** The N1-2 two-layer theorem's own predictions, pre-registered
before the runs: (P1) blindness is *robust* to symmetric gain faults (R⁻ ≈ α at every κ; the amplitude ceiling needs a
genuine symmetry-breaking bifurcation, not reached by symmetric degradation); (P2) the statistic split — paired_energy is
blind to a zero-mean law difference (pinned at α), energy_distance is consistent (power → 1 with the variance ratio),
decisive in the exact-hypothesis toy; (P3) the blindness theorem **as a slip classifier** — one-sided slip fires R⁻,
bilateral slip is R⁻-silent but lights the InEKF NIS (Go2 clean: unilateral R⁻ 1.0 / NIS quiet, uniform R⁻ 0.0 / NIS 1.4).

**L3 — Own hardware, M1 wheeled-legged (Block D; rp025).** First self-collected corpus (2026-08-10, four rosbag2
sessions; `m1_h_data_audit.md`). Verified channel map (names, per-leg mirror signs, IMU frame). First real-robot R⁻: the
naive H₀ flip test rejects on all three rolling sessions (the stably-asymmetric healthy loop, real ε_dyn) while the
**sequential H₀′ monitor stays silent** — the H₀′-is-primary story confirmed on hardware. Rolling InEKF recovers the
driven path (0.99–1.00 of the odometry length) where fixed-foot filters recover ~0.03 (the e10 result on real data).

**L4 — Public multi-platform (Block PUB, e17; rp030).** Leg-KILO Go1 (RA-L 2024) and Cerberus Street A1 (ICRA 2023),
straight-trot mined: the naive H₀ flip test rejects on every sequence (real mirror asymmetry) and the **H₀′ per-window
test is in band (FAR ≈ α) on 4/5 Leg-KILO sequences** (Go1 walking trot; Street A1's 260 m outdoor walk is partly
non-stationary, H₀′ win-reject 0.29). The πᵢ-gating FAR check reproduces on real Go1 (both the 0.4 m/s threshold and the
calibrated gate ≈ α on this slow trot — the threshold's over-rejection needs the faster gait of the e16 sim).

**L5 — Public multi-terrain (MIT Mini Cheetah contact dataset, CoRL 2021; e17).** The A3 breadth: the R⁻ detection
channel fires across **all 8 terrains** (naive H₀ p ≈ 0.002 everywhere) — the asymmetry is detected regardless of
environment. Honest caveat: on this *flying* trot (0.25 s period, flight phase) the cycle-level H₀′ is itself elevated
(non-stationary cycle-to-cycle + phase-registration stress), unlike the cleaner walking trots of L3/L4 — cycle-level H₀′
needs a well-registered, stationary gait. The air gaits are the leg-in-air (weld) nominal, an RA-L bridge to the L3 §L
weld result.

**Still open (the lab):** controlled physical faults (payload, foot-friction pads, single-wheel μ), Gate 4 (joint-level
command access for fault injection), efforts-semantics vendor confirmation — see `docs/protocol/hw_slip_protocol.md`.

## 9. Limitations (self-listed, seven)

1. Σ-invariant faults are invisible to R⁻ by construction; the magnitude channel that covers them is not nuisance-
   invariant (drift inflates it: e04b/e05a) — the guarantee is asymmetric across the two channels.
2. Exactness of the wrapped per-cycle element needs reversible or measurement-noise-dominated fluctuations (e04d
   boundary at actuator noise 0.1 N·m); the unfolded element is exact but costs 3 cycles per element.
3. Cross-cycle exchangeability is assumed, not tested; slow drifts (temperature) need blocking/differencing.
4. The ε-budget is total-variation based and linear in T_c (loose); the IPM refinement (N1-4) is owed.
5. Spontaneous symmetry breaking (chiral attractor) makes H₀ false with A1–A4 intact — Gate 1b must be run.
6. The equivariant DeLaN inherits the leg-Lagrangian approximation (trunk angular effects in β̂; contact term needs
   measured or estimated wrenches); the analytic observer needs the model and contact wrenches (sim oracle now).
7. Hardware validation and the external benchmark are pending; the sim world is mirror-exact by construction (the
   URDF world with real chirality is the A1′ rehearsal, e01/Block G).

## 10. Conclusion

Symmetry as the nominal model: what a healthy robot must satisfy is used as the test's null; the learned model, when
present, is made to respect the same symmetry so that the test can inherit it. Future: full Part 1 (unfolded shift,
IPM), phase estimation without controller access, hardware.

---

### Section ↔ experiment / pack index
| section | experiments | packs |
|---|---|---|
| 3–4 | — (theory) | rp000, rp001, rp002, rp004, rp011 |
| 5 | Block Q, protocol docs | rp012 |
| 6 Fig A | e01a, e01b, e04b, e04d, Block G | rp003, rp005, rp007 |
| 6 Fig B | e04a, e05b | rp005, rp008 |
| 6 Fig C | e04c, e13c | rp005, rp013 |
| 6 Fig D | e13a, e13b, Block Q | rp012, rp013 |
| 6 isolation | e04e, e05c, e06, e13c | rp005, rp008, rp013 |
| 6 baselines | e07 | rp009 |
| 6 estimation | e02 | rp006 |
| 7 | e03 | (pending) |
| 8 | hardware | (pending); M1 model audit rp010 |
