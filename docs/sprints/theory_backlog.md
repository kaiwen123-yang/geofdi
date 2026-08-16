# Theory backlog — deferred theory-sprint items (registered from the working sprints)

One entry per deferred theorem/write-up that is not being done in the current sprint. Add the sprint that deferred it,
the claim to formalise, the empirical anchor that already exists, and the target `.tex`.

## N2 formalisation — sparse-correction consistency + signature = observable projection (registered Sprint 8, Block L5)

**Status: NEXT theory-sprint item (not done in Sprint 8).** Sprint 7 Block N2 built and validated the augmented invariant
estimators empirically (`inekf/rinekf_bias.py`, `inekf/inekf_rolling.py`; e10) but the results are not yet stated as
theorems in Part 2. Two propositions to write into `theory/parts/…` (the Part 2 estimator section) with proofs:

1. **Sparse-correction consistency.** For the bias-augmented right-invariant EKF, a fault that enters as a sparse
   additive term on a subset of measurement channels (e.g. a single encoder bias b_enc,j, or a gyro bias b_g) is
   consistently reconstructed by the corresponding augmented state, up to the observability of that channel under the
   gait. Formalise the observability condition (which augmented states are recoverable in flat trot / rolling) and the
   consistency (the innovation-driven estimate → the true injected bias as K→∞ on the observable subspace). Empirical
   anchor (e10, Sprint 7): encoder-bias reconstruction cos 0.73 / b̂ 0.085 for a 0.05 injection; gyro-bias only partial
   (pitch 0.010, yaw InEKF-unobservable). Sign facts already fixed: H_benc = +R J, A[d_i,R]=0 for rolling (group-affine),
   mirror-equivariance signs b_g→−E b_g, b_a→E b_a, b_enc→S12 b_enc[PERM] (see [[sprint7-outcomes]]).

2. **Signature = observable projection of the fault onto the innovation.** The fault's InEKF innovation signature is the
   projection of the fault direction onto the observable subspace, transported by the adjoint (the "innovation-direction
   match to the analytic adjoint prediction" measured in e10). State it as: the expected innovation direction under fault
   f equals Π_obs Ad_X H f, and give the mirror-equivariance of the signature (why a bilaterally-symmetric fault produces
   a Σ-invariant signature — links to N1-2 blindness). Empirical anchor: e10 slip mirror-covariance cos 0.993.

Target: Part 2 estimator section; cross-reference N1-2 (blindness) and N3 (isolability). Falsification: an observable
augmented state whose estimate does NOT converge to the injection, or a signature direction that does not match the
adjoint prediction beyond the e10 noise floor. Do together with the Part 2 recompile.

## (add further deferred theory items below as sprints defer them)
