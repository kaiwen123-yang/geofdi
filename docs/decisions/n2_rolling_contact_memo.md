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
  ω, a are known inputs to the R and v blocks), then the d_i column rate R u_i = X · [u_i; 0] has the navigation-equation
  form Barrau–Bonnabel (2017) prove **group-affine**. The consequence is the defining InEKF property: the right-invariant
  error dynamics are trajectory- AND input-independent. Working the RI error ξ_{d_i} = d̂_i − R̂R^T d_i through with the
  SAME measured ω, u_i in both truth and estimate gives (derived, and checked numerically in
  scratchpad/roll_check.py to 1e-8):
    dξ_{d_i}/dt = 0,   i.e.  A[d_i, R] = 0.
  The cancellation is exact because, unlike the velocity block v̇ = R a + **g** whose world-CONSTANT gravity term
  produces the A[v, R] = [g]_× coupling, the moving-contact rate R u_i has **no world-constant part** — u_i is a pure
  body input rotated by the shared R, and R̂R^T is constant under the shared ω, so the R̂ u_i terms cancel.
  **Correction to an earlier draft of this memo, which claimed A[d_i, R] = −R[u_i]_×: that is the *non-invariant* (plain-EKF,
  world-position parameterisation) Jacobian for a moving landmark; in the RIEKF the group-affine structure annihilates it.**
  So the propagation matrix A is UNCHANGED from the fixed-foot filter; the rolling contact changes only
  (a) the **mean** propagation d̂_i+ = d̂_i + R̂ u_i Δt, and
  (b) the d_i **process noise**: the wheel-rate/leg-encoder uncertainty in u_i enters the d_i block as a body-frame
      covariance Cov(u_i), anisotropic in the wheel frame — large on the forward (rolling-odometry) axis σ_roll,
      small on the lateral/vertical (no-slip / no-penetration) axes σ_slip — mapped by the same Ad_X̂ as every other
      block. **Conclusion: the rolling contact is group-affine; the InEKF's exact log-linear covariance propagation is
      preserved with the SAME A, and the no-slip constraint is realised as a low-σ_slip prior on the lateral/vertical
      contact motion rather than a separate pseudo-measurement.**
- The lateral-no-slip and vertical-no-penetration constraints are enforced as **pseudo-measurements** (below), not in
  the dynamics, so they do not affect group-affineness.

## Measurement model
Two measurements per rolling contact, both right-invariant (functions of X^{-1} times a fixed vector):
1. **Foot-position kinematics** (as in the fixed-foot filter): y = h(q) = R^T(d_i − p), innovation z = R̂ h(q) + p̂ − d̂_i,
   H = [0 0 −I … I(i) …]. This ties the base to the (now moving) contact state.
2. **Non-holonomic no-slip constraint** (rolling): the contact point moves in the world at exactly R u_i with no
   lateral or vertical drift. Rather than a separate pseudo-measurement, this is realised as the anisotropic d_i
   process noise of point (b) above — small σ_slip on the lateral/vertical wheel axes (the contact cannot slide sideways
   or sink) and larger σ_roll on the forward axis (the rolling-odometry uncertainty). Because the InEKF error is
   group-affine, this soft constraint costs no extra measurement and preserves the exact covariance recursion.
The odometry information enters through ḋ_i = R u_i in propagation (the contact state moves at the measured rolling rate),
so the base velocity v is observable from the wheel odometry the way a fixed foot makes it observable from zero-velocity:
the foot-position innovation forces the base to keep pace with the known-moving contact.

## Fault relevance (why N2 cares)
- A single-wheel encoder or motor fault changes ω_wheel,i and hence u_i: the innovation of contact i acquires a steady
  forward-axis offset ≈ r·Δω_wheel (a rolling-odometry residual), while a leg-joint (ABAD/HIP/KNEE) fault shows in the
  foot-position innovation. The adjoint structure (RI-EKF innovations are R-covariant) means a mirror-leg fault maps to
  the mirror innovation — the same Σ-equivariance the detection channel uses, now on the estimator residual.
- The rolling contact makes the base velocity observable during straight driving, so the InEKF NIS is a usable residual
  channel for the rolling mode (Σ = G) even though there is no gait phase.

## Implementation (`geofdi.inekf.inekf_rolling`)
`RollingRIEKF` extends `RIEKF`: `propagate` advances each contact state by R̂ u_i Δt (mean) with the SAME A as the base
filter (A[d_i, R] = 0, verified) and an anisotropic wheel-frame d_i process noise (σ_roll forward, σ_slip lateral/
vertical); `set_rolling_inputs(u_body_per_leg, wheel_frame_per_leg)` supplies u_i and the wheel axes each step (u_i =
[r·ω_wheel, 0, 0] in the wheel frame for straight rolling); `correct` is inherited (the foot-position kinematic
measurement), the no-slip being folded into σ_slip. NIS smoke on `m1_wheeled_sym`: RollingRIEKF (correct model) vs a
fixed-foot RIEKF (wrong model for rolling — the stationary-foot assumption is violated, so its NIS inflates) and ESKF.

## Status
Memo + `inekf_rolling` + NIS smoke this sprint (e10). The full outdoor validation (legacy M1 bags, `raw/m1/legacy-aug`)
is a Day-0+ item — those bags are wheeled driving and are the natural first rolling-InEKF test set.
