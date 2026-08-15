"""Mirror equivariance of the bias-augmented contact InEKF (Sprint 7 Block N2, geofdi.inekf.rinekf_bias.RIEKFBias).

Directly drives propagate / add_contact / remove_contact / correct with a designed IMU + contact + encoder-Jacobian
stream and its exact sagittal mirror, and checks the WHOLE augmented state transforms by the C2 action to ~1e-9:
    R -> E R E,  v -> E v,  p -> E p,  d_i -> E d_{sigma(i)},
    b_g -> -E b_g (gyro bias is a pseudovector),  b_a -> E b_a (accel bias is polar),
    b_enc -> S12 . b_enc[PERM] (encoder bias mirrors like the joint vector),  and NIS is identical.
No finite differences and no simulator: the leg Jacobians are supplied, so the only error is floating-point (~1e-12),
which makes this a clean algebraic certificate that the bias augmentation preserves the mirror symmetry the FDI test
relies on. (test_inekf.py::test_filter_is_mirror_equivariant is the sim-driven analogue for the bias-free filter.)"""
import numpy as np

from geofdi.inekf.rinekf_bias import RIEKFBias

E = np.diag([1.0, -1.0, 1.0])                              # sagittal reflection y -> -y
SIG = {0: 1, 1: 0, 2: 3, 3: 2}                             # LF<->RF, LH<->RH  (feet 0..3 = LF,RF,LH,RH)
DSGN = np.diag([-1.0, 1.0, 1.0])                           # per-leg joint signs HAA -1, HFE +1, KFE +1
PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])   # 12-dim joint permutation under the mirror
S12 = np.tile(np.array([-1.0, 1.0, 1.0]), 4)              # signs for the permuted joints
H0 = {0: np.array([0.19, 0.14, -0.30]), 1: np.array([0.19, -0.14, -0.30]),
      2: np.array([-0.19, 0.14, -0.30]), 3: np.array([-0.19, -0.14, -0.30])}


def _mirror_benc(b):                                       # b_enc' = S12 * b_enc[PERM]
    return S12 * np.asarray(b)[PERM]


def _exp_so3(phi):
    from geofdi.inekf.liegroups import exp_so3
    return exp_so3(phi)


def _build_stream(seed=7, n=180):
    """A nominal (gyro, accel, contact-set, foot h, leg-Jacobian) stream A and its exact mirror B."""
    rng = np.random.default_rng(seed)
    dt = 0.005
    # contact schedule: two diagonals alternating (trot) then all-four, exercising add/remove with the bias block
    def cset(k):
        if k < 60:
            return (0, 3)
        if k < 120:
            return (1, 2)
        return (0, 1, 2, 3)
    A, B = [], []
    Jbank = {leg: rng.normal(0, 0.3, (3, 3)) + np.eye(3) for leg in range(4)}
    for k in range(n):
        gyro = np.array([0.05 * np.sin(0.1 * k), 0.03 * np.cos(0.07 * k), 0.02]) + rng.normal(0, 0.01, 3)
        accel = np.array([0.1 * np.sin(0.05 * k), 0.08 * np.cos(0.06 * k), 9.81]) + rng.normal(0, 0.02, 3)
        feet = cset(k)
        measA, measB = [], []
        for leg in feet:
            wobble = rng.normal(0, 0.002, 3)
            hA = H0[leg] + wobble
            cov = (2e-3 ** 2) * (Jbank[leg] @ Jbank[leg].T) + 1e-8 * np.eye(3)
            measA.append((leg, hA, cov, Jbank[leg], np.array([3 * leg, 3 * leg + 1, 3 * leg + 2])))
            # mirror image of this foot
            legm = SIG[leg]
            hB = E @ hA
            covB = E @ cov @ E.T
            Jm = E @ Jbank[leg] @ DSGN
            measB.append((legm, hB, covB, Jm, np.array([3 * legm, 3 * legm + 1, 3 * legm + 2])))
        A.append((gyro, accel, tuple(feet), measA))
        B.append((-E @ gyro, E @ accel, tuple(SIG[f] for f in feet), measB))
    return dt, A, B


def _run(stream, dt, R0, v0, p0, bg0, ba0):
    f = RIEKFBias(R0, v0, p0, n_enc=12, bg0=bg0, ba0=ba0,
                  P0_diag=(1e-4, 1e-4, 1e-4), P0_bias=(1e-6, 1e-5), P0_benc=1e-6,
                  sigma_gyro=0.01, sigma_accel=0.1, sigma_contact=2e-3, sigma_kin_floor=2e-3)
    prev = set()
    nis = []
    for (gyro, accel, feet, meas) in stream:
        f.propagate(gyro, accel, dt)
        cur = set(feet)
        for leg in cur - prev:
            h, cov = next((h, c) for (lg, h, c, J, ei) in meas if lg == leg)
            f.add_contact(leg, h, cov)
        for leg in prev - cur:
            f.remove_contact(leg)
        prev = cur
        measurements = [(lg, h, c) for (lg, h, c, J, ei) in meas]
        leg_jac = {lg: (J, ei) for (lg, h, c, J, ei) in meas}
        rec = f.correct(measurements, leg_jac=leg_jac)
        if rec is not None:
            nis.append(rec["nis"])
    return f, np.array(nis)


def test_bias_augmented_filter_is_mirror_equivariant():
    dt, SA, SB = _build_stream()
    rng = np.random.default_rng(1)
    R0 = _exp_so3(rng.normal(0, 0.05, 3)); v0 = rng.normal(0, 0.02, 3); p0 = np.array([0.0, 0.0, 0.30])
    bg0 = np.array([0.010, -0.004, 0.002]); ba0 = np.array([0.03, -0.02, 0.05])
    fA, nisA = _run(SA, dt, R0, v0, p0, bg0, ba0)
    fB, nisB = _run(SB, dt, E @ R0 @ E, E @ v0, E @ p0, -E @ bg0, E @ ba0)

    assert np.abs(fB.R - E @ fA.R @ E).max() < 1e-9
    assert np.abs(fB.v - E @ fA.v).max() < 1e-9
    assert np.abs(fB.p - E @ fA.p).max() < 1e-9
    assert np.abs(fB.bg - (-E @ fA.bg)).max() < 1e-9, (fB.bg, -E @ fA.bg)
    assert np.abs(fB.ba - E @ fA.ba).max() < 1e-9, (fB.ba, E @ fA.ba)
    assert np.abs(fB.benc - _mirror_benc(fA.benc)).max() < 1e-9, (fB.benc, _mirror_benc(fA.benc))
    for i, dA in fA.d.items():
        assert np.abs(fB.d[SIG[i]] - E @ dA).max() < 1e-9
    assert len(nisA) == len(nisB)
    assert np.abs(nisA - nisB).max() < 1e-7 * (1 + np.abs(nisA).max())


def test_encoder_bias_is_observable_only_with_leg_jacobian():
    """Without the leg-Jacobian argument the encoder-bias block gets no measurement information: its marginal
    covariance only grows (random walk), whereas with the Jacobian the contacts shrink it. A one-line guard that the
    H_benc wiring is actually doing something."""
    dt, SA, _ = _build_stream(seed=3, n=120)
    R0 = np.eye(3); v0 = np.zeros(3); p0 = np.array([0.0, 0.0, 0.30])

    def run(with_jac):
        f = RIEKFBias(R0, v0, p0, n_enc=12, P0_benc=1e-4, sigma_benc_rw=1e-5, sigma_gyro=0.01, sigma_accel=0.1)
        prev = set()
        for (gyro, accel, feet, meas) in SA:
            f.propagate(gyro, accel, dt)
            cur = set(feet)
            for leg in cur - prev:
                h, cov = next((h, c) for (lg, h, c, J, ei) in meas if lg == leg)
                f.add_contact(leg, h, cov)
            for leg in prev - cur:
                f.remove_contact(leg)
            prev = cur
            measurements = [(lg, h, c) for (lg, h, c, J, ei) in meas]
            leg_jac = {lg: (J, ei) for (lg, h, c, J, ei) in meas} if with_jac else None
            f.correct(measurements, leg_jac=leg_jac)
        bi = f._benc_idx()
        return np.trace(f.P[bi, bi])

    tr_jac = run(True); tr_nojac = run(False)
    assert tr_jac < tr_nojac                                # information reduces the encoder-bias covariance
