# N2 memo — rolling-contact InEKF for the wheeled M1 (Sprint 7 Block N2)

## Problem
The contact-aided RI-EKF (Hartley et al. 2020, `geofdi.inekf.rinekf`) models each stance foot as a **fixed** world
point, d_i = const, with the kinematic measurement y = R^T(d_i − p) = h(q). For the wheeled M1 the contact point is not
fixed: the wheel rolls, so the contact point translates along the ground at the wheel's rolling velocity. A fixed-point
foot state would fight the rolling and corrupt the base estimate. This memo specifies the rolling-contact model, argues
it keeps the group-affine (log-linear) structure of the InEKF, and defines the measurement used by `inekf_rolling`.

## Rolling-contact kinematics
Let leg i have wheel radius r, wheel spin rate ω_wheel,i (about the leg's lateral axis, +y in the leg frame), and let
the contact point be the lowest point of the wheel. Under pure rolling without lateral slip the contact point's WORLD
velocity is
    ḋ_i = R u_i,      u_i = v_body,i + r ω_wheel,i (x̂_leg × ẑ) ,
i.e. the body-frame velocity of the wheel centre (from the leg kinematics: v_body,i = J_leg(q) q̇ + ω_b × p_wheel,i)
plus the rolling contribution r·ω_wheel along the wheel's forward direction, with **zero lateral component** (no side
slip) and **zero vertical component** (the wheel stays on the ground). In the world frame ḋ_i = R u_i, so the contact
state is no longer constant: d_i+ = d_i + R u_i Δt.

## Group-affine preservation
The RI-EKF's log-linearity relies on the state kinematics being **group-affine** (Barrau–Bonnabel 2017): the deterministic
dynamics of X = [R v p d_1 … d_N] ∈ SE_{2+N}(3) must satisfy f(X) = X f(I) + f(I) X − X f(I) X ... — concretely, each
block's rate must be a linear function of the state consistent with the group action. The fixed-foot model has ḋ_i = 0
(trivially group-affine). The rolling model adds ḋ_i = R u_i.
- If u_i is treated as a **known input** (computed from the measured wheel rate and leg kinematics, exactly as the IMU
  ω, a are known inputs to the R and v blocks), then ḋ_i = R u_i has the **same form** as the position kinematics
  ṗ = v = R (R^T v): a world-frame rate given by R times a body-frame input. This is the group-affine template of the
  velocity/position blocks, so the augmented dynamics remain group-affine and the error ξ_d,i still satisfies a
  linear ODE dξ_d,i/dt = (terms in ξ_R through the [u_i]_× coupling) + Ad noise. The propagation Jacobian gains a block
    A[d_i, R] = −R [u_i]_×      (the world contact-rate depends on the attitude error through R),
  analogous to A[v, R] = [g]_× and A[p, v] = I; A^k is still nilpotent (the added block does not create a cycle), so
  Φ = Exp(AΔt) is still a finite polynomial. **Conclusion: the rolling contact is group-affine when u_i is a known input;
  the InEKF's exact log-linear covariance propagation is preserved.**
- The lateral-no-slip and vertical-no-penetration constraints are enforced as **pseudo-measurements** (below), not in
  the dynamics, so they do not affect group-affineness.

## Measurement model
Two measurements per rolling contact, both right-invariant (functions of X^{-1} times a fixed vector):
1. **Foot-position kinematics** (as in the fixed-foot filter): y = h(q) = R^T(d_i − p), innovation z = R̂ h(q) + p̂ − d̂_i,
   H = [0 0 −I … I(i) …]. This ties the base to the (now moving) contact state.
2. **Non-holonomic pseudo-measurement** (rolling constraint): the contact point's body-frame velocity has no lateral
   (ŷ_leg) and no vertical (ẑ) component beyond r·ω_wheel: the measured wheel/leg kinematics predict u_i, and the filter's
   implied contact velocity ḋ_i must match R u_i. Implemented as a velocity residual on the two constrained axes
   (lateral, vertical) with a small covariance; the forward axis is left free (it carries r·ω_wheel, which is the odometry).
The odometry information enters through ḋ_i = R u_i in propagation (the contact state moves at the measured rolling rate),
so the base velocity v is observable from the wheel odometry the way a fixed foot makes it observable from zero-velocity.

## Fault relevance (why N2 cares)
- A single-wheel encoder or motor fault changes ω_wheel,i and hence u_i: the innovation of contact i acquires a steady
  forward-axis offset ≈ r·Δω_wheel (a rolling-odometry residual), while a leg-joint (ABAD/HIP/KNEE) fault shows in the
  foot-position innovation. The adjoint structure (RI-EKF innovations are R-covariant) means a mirror-leg fault maps to
  the mirror innovation — the same Σ-equivariance the detection channel uses, now on the estimator residual.
- The rolling contact makes the base velocity observable during straight driving, so the InEKF NIS is a usable residual
  channel for the rolling mode (Σ = G) even though there is no gait phase.

## Implementation (`geofdi.inekf.inekf_rolling`)
`RollingRIEKF` extends `RIEKF`: `propagate` advances each contact state by R u_i Δt and adds the A[d_i, R] = −R[u_i]_×
Jacobian block; `set_rolling_inputs(u_body_per_leg)` supplies u_i each step from the wheel rate and leg Jacobian;
`correct` adds the lateral/vertical rolling pseudo-measurement. Equivariance unit test (S3 style): mirrored inputs give
the mirrored state to 1e-9. NIS smoke on `m1_wheeled_sym`: InEKF vs ESKF per-bin FAR (realistic regime).

## Status
Memo + `inekf_rolling` + NIS smoke this sprint (e10). The full outdoor validation (legacy M1 bags, `raw/m1/legacy-aug`)
is a Day-0+ item — those bags are wheeled driving and are the natural first rolling-InEKF test set.
