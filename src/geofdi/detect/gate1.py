"""Gate 1 (Assumption A2 audit) estimator — the mirrored-command distributional gap (Sprint 7 Block W4 rehearsal).

Part 0: compare the commanded torques at phase theta + 1/2 with the rho_U-transformed commands at phase theta, on
matched mirrored states; report the estimated gap with a confidence interval. Implemented estimator (per mirror pair
of torque channels): the state-matched mean mirrored-command difference
    eps_hat_j = | mean_k mean_theta [ tau_leg,j(theta) - s_j * tau_partner,j(theta + 1/2) ] |     (trot: cycles registered
by the kinematic phase; rolling: fixed blocks, shift 0), studentized by its cycle-to-cycle bootstrap std. State matching:
in the trot the phase IS the state (the periodic orbit); in rolling the legs are static. The unconditional version
(pooled over phase) is the second column. Units: N m per joint pair; the aggregate eps_hat_ctrl = max over joints.

Rehearsal (sim only): inject a controller asymmetry (kp gain 1+delta on one joint / wheel-rate gain) and compare
eps_hat with the TRUE injected command offset, obtained by re-evaluating the symmetric controller on the recorded states:
    delta_tau_true(t) = tau_cmd(t) - tau_sym(q_meas(t), dq_meas(t), t)  -> eps_true_j = |mean delta_tau_true_j|.
Gate: |eps_hat - eps_true| / eps_true < 0.3 on the injected joint.
"""
from __future__ import annotations

import numpy as np

from ..detect.permutation import pooled_scale


def gate1_estimate(Z: np.ndarray, rep, names: list[str], torque_group: str = "tau_cmd", n_boot: int = 200,
                   rng: np.random.Generator | None = None) -> dict:
    """Z (K, d, N) registered cycles/blocks; rep the C2Rep; returns per-channel mirrored-command gaps for the torque
    channels: mean difference (N m), bootstrap std, |mean|/std, and the pooled-scale standardized version."""
    rng = np.random.default_rng() if rng is None else rng
    Zs = rep.apply("s", Z); D = Z - Zs                                # D[k, c, :] = tau_c(theta) - s tau_partner(theta+1/2)
    K = Z.shape[0]; out = {}; agg = 0.0; agg_name = None
    idx = [i for i, n in enumerate(names) if n.startswith(torque_group + "_")]
    for i in idx:
        m = D[:, i, :].mean()                                          # mean over cycles and phase
        boots = np.array([D[rng.integers(0, K, K), i, :].mean() for _ in range(n_boot)])
        per_cycle = D[:, i, :].mean(axis=1)
        out[names[i]] = {"eps_hat": float(abs(m)), "signed_mean": float(m), "boot_std": float(boots.std()), "z": float(abs(m) / (boots.std() + 1e-12)),
                         "cycle_std": float(per_cycle.std())}
        if abs(m) > agg:
            agg, agg_name = abs(m), names[i]
    return {"per_channel": out, "eps_hat_ctrl": float(agg), "argmax_channel": agg_name, "K": int(K)}
