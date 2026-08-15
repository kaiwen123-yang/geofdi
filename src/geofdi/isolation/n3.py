"""N3 — isolability certificate for joint-space residual signatures (Block L1).

Signature dictionary. A residual generator (analytic momentum observer or a learned nominal model) yields the joint
residual r(t) in R^12; per gait cycle it is phase-registered into a profile R in R^{12 x N} and flattened to
x = vec(R) in R^{12N}. For joint j the three fault types have the analytic first-order signatures
    gain     (kappa):  s = -(1 - kappa) * tau_j(theta) e_j     (the fault removes a fraction of the commanded torque)
    bias     (b):      s =  b * 1(theta) e_j                    (constant torque offset)
    friction (dmu):    s = -dmu * sign(dq_j(theta)) e_j         (Coulomb friction increase; viscous part analogous)
where tau_j(theta), sign(dq_j(theta)) are the NOMINAL phase profiles (mean over nominal cycles) — the dictionary uses
nominal data only. Each class c = (type, joint) is the 1-D subspace S_c = span(s_c) of R^{12N} (rank-1 dictionary; a
rank-k version stacks the top-k principal profiles across nominal cycles / configuration bins).

Geometry. Between joints all classes are orthogonal (e_j vs e_k). Within a joint the angle between gain and bias is
arccos(|mean tau_j| / rms tau_j): a joint whose torque keeps its sign over the cycle makes a gain fault look like a
bias — the certificate below quantifies exactly this.

Certificate (Davis–Kahan, Yu–Wang–Samworth 2015 variant, Theorem 2 with k = 1). Under fault c with magnitude m the
per-cycle profile is x_k = m s_c + e_k where e_k is the residual error (nominal model error + noise); the population
uncentered second moment is Sigma = m^2 s_c s_c^T + Sigma_e with ||Sigma_e||_op = beta_op^2. The top eigenvector v of
Sigma satisfies
    sin theta(v, s_c) <= 2 ||Sigma_e||_op / gap,     gap = lambda_1 - lambda_2 (m^2 s_c s_c^T) = m^2 ||s_c||^2
(YWS: ||sin Theta||_F <= 2 min(sqrt(k) ||E||_op, ||E||_F) / (lambda_k - lambda_{k+1}); k = 1 gives the constant 2). The
nearest-subspace classifier applied to v (or, equivalently, to the mean profile) is correct if v is closer to S_c than
to every other class, which the triangle inequality guarantees whenever
    theta_min(c, c') > 2 * arcsin( min(1, 2 beta_op^2 / (m^2 ||s_c||^2)) )      for all c' != c.
So class c is DK-certified isolable at magnitude m iff  m^2 ||s_c||^2 * sin(theta_min(c, .)/2) > 2 beta_op^2 , i.e.
gap > c_DK * beta with c_DK = 2 / sin(theta_min/2) in units of beta_op^2. beta_op^2 is measured on NOMINAL cycles as the
top eigenvalue of the residual second-moment matrix in the profile space (analytic model: the observer floor + noise;
DeLaN: its model error).

Classifier. Nearest 1-D subspace of the empirical top eigenvector of the post-onset second-moment matrix (uncentered
PCA of the residual profiles), over the 36 classes; the confusion matrix is compared with the certificate.

Centering. Profiles are taken RELATIVE to the calibrated nominal residual profile (mean over the calibration cycles):
the nominal residual generator has a deterministic gait-locked floor pattern (observer floor / learned-model bias)
which is known after calibration and is not part of the fault signature; beta_op^2 is then the top eigenvalue of the
centred nominal second moment (the fluctuating part of the model error), and the fault signature is the CHANGE of the
mean profile — the same H0'-style reading as the S1/S2 change tests.
"""
from __future__ import annotations

import numpy as np

TYPES = ("gain", "bias", "friction")


def build_dictionary(tau_cycles: np.ndarray, dq_cycles: np.ndarray, joint_names: list[str]) -> dict:
    """tau_cycles, dq_cycles: nominal registered cycles (K, 12, N). Returns {(type, joint): unit vector in R^{12N}} and
    the raw (unnormalised) profiles used to scale magnitudes."""
    K, nj, N = tau_cycles.shape
    tau_bar = tau_cycles.mean(0); sgn_bar = np.sign(dq_cycles).mean(0)          # (12, N)
    D = {}; raw = {}
    for j, name in enumerate(joint_names):
        for typ in TYPES:
            prof = np.zeros((nj, N))
            if typ == "gain":
                prof[j] = -tau_bar[j]                     # per unit (1 - kappa)
            elif typ == "bias":
                prof[j] = 1.0                             # per unit b
            else:
                prof[j] = -sgn_bar[j]                     # per unit dmu (Coulomb)
            v = prof.ravel(); raw[(typ, name)] = v.copy()
            D[(typ, name)] = v / (np.linalg.norm(v) + 1e-15)
    return {"unit": D, "raw": raw, "N": N, "nj": nj}


def principal_angle_matrix(dictionary: dict) -> tuple[list, np.ndarray]:
    keys = list(dictionary["unit"].keys()); V = np.array([dictionary["unit"][k] for k in keys])
    C = np.clip(np.abs(V @ V.T), 0, 1); A = np.degrees(np.arccos(C)); np.fill_diagonal(A, 0.0)
    return keys, A


def beta_op2(nominal_profiles: np.ndarray) -> float:
    """Top eigenvalue of the uncentered second-moment matrix of nominal residual profiles (K, 12N)."""
    X = np.asarray(nominal_profiles); S = X.T @ X / len(X)
    return float(np.linalg.eigvalsh(S)[-1])


def dk_certificate(dictionary: dict, cls, magnitude: float, beta2: float, keys=None, A=None) -> dict:
    """Is class `cls` DK-certified isolable at fault magnitude `magnitude` given beta_op^2 = beta2?"""
    if keys is None or A is None:
        keys, A = principal_angle_matrix(dictionary)
    i = keys.index(cls); theta_min = float(np.min(np.delete(A[i], i)))
    gap = magnitude ** 2 * float(np.linalg.norm(dictionary["raw"][cls]) ** 2)
    sin_pert = min(1.0, 2 * beta2 / gap) if gap > 0 else 1.0
    pert_deg = float(np.degrees(np.arcsin(sin_pert)))
    ok = 2 * pert_deg < theta_min
    beta2_threshold = gap * np.sin(np.radians(theta_min / 2)) / 2      # certificate flips here
    return {"class": cls, "magnitude": magnitude, "theta_min_deg": theta_min, "nearest": keys[int(np.argmin(np.where(np.arange(len(keys)) == i, np.inf, A[i])))],
            "gap": gap, "beta2": beta2, "dk_perturbation_deg": pert_deg, "certified": bool(ok), "beta2_threshold": float(beta2_threshold)}


def top_direction(profiles: np.ndarray) -> np.ndarray:
    """Top eigenvector of the uncentered second-moment matrix of (K, 12N) profiles (sign fixed by the mean)."""
    X = np.asarray(profiles); m = X.mean(0)
    # power-iteration friendly: use SVD of X (K x 12N)
    U, s, Vt = np.linalg.svd(X, full_matrices=False); v = Vt[0]
    return v if v @ m >= 0 else -v


def nearest_class(v: np.ndarray, dictionary: dict):
    keys = list(dictionary["unit"].keys()); V = np.array([dictionary["unit"][k] for k in keys])
    c = np.abs(V @ v) / (np.linalg.norm(v) + 1e-15)
    order = np.argsort(-c)
    return keys[order[0]], float(c[order[0]]), keys[order[1]], float(c[order[1]])
