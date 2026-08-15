"""Three-channel isolation readout (Sprint 7 Block I).

Reading vector for one monitored segment:
  (1) R^-  : the invariance-test state (alarm / silent) and, if alarmed, the ranked (pair, joint) from the R^- projection
             energy on the residual (or raw) element with swing-phase conditioning and the nominal calibration scale;
  (2) joint residual rows : per-(leg, joint) mean shift of the momentum / DeLaN residual relative to the calibrated
             nominal profile -> which joint, and left/right within the R^- pair by the |mean shift| of THAT joint's row
             (Block-I fix: the whole-leg residual ENERGY score decreases on a friction-faulted leg, so max(signed energy
             deviation) picks the wrong leg; the |row mean shift| is monotone in the fault for gain, bias and friction);
  (3) floating-base rows : mean shift of the 6 base momentum-residual rows -> payload (f_z) / lateral offset (m_x).

decide(reading, rules) applies the pre-registered decision rules (docs/protocol/e09_preregistration.md) and returns a
class label with a confidence and the raw reading. Nothing is trained.
"""
from __future__ import annotations

import numpy as np

from ..detect.monitors import SWING, channel_projection_energy, rank_groups  # noqa: F401
from ..detect.permutation import pooled_scale
from ..groups.c2 import C2Rep

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
PAIR_OF = {"LF": "F", "RF": "F", "LH": "H", "RH": "H"}
PAIR_LEGS = {"F": ("LF", "RF"), "H": ("LH", "RH")}


def rminus_state(Z, rep, K_cal, names, swing=True, groups=("q", "dq", "tau_cmd", "tau_meas")):
    """R^- projection-energy ranking of (pair, joint) on the calibration-scaled antisymmetric mean shift."""
    cal, post = Z[:K_cal], Z[K_cal:]
    e = channel_projection_energy(post, rep, names, swing_condition=swing, groups=groups, Z_cal=cal)
    ranked = rank_groups(e["per_pair"])
    (pair, joint), energy = ranked[0]
    return {"pair": pair, "joint": joint, "pair_energy": float(energy), "ranked_pairs": [(f"{p}-{j}", float(v)) for (p, j), v in ranked]}


def joint_row_shift(Zr, K_cal):
    """(12,) standardized mean shift of each joint residual row (post-onset minus calibration mean, / calibration std)."""
    mu = Zr[:K_cal].mean(axis=(0, 2)); sd = Zr[:K_cal].std(axis=(0, 2)) + 1e-12
    return (Zr[K_cal:].mean(axis=(0, 2)) - mu) / sd


def base_row_shift(Zb, K_cal):
    """(6,) standardized mean shift of the base momentum-residual rows (fx, fy, fz, mx, my, mz)."""
    mu = Zb[:K_cal].mean(axis=(0, 2)); sd = Zb[:K_cal].std(axis=(0, 2)) + 1e-12
    raw = Zb[K_cal:].mean(axis=(0, 2)) - Zb[:K_cal].mean(axis=(0, 2))
    return raw, (Zb[K_cal:].mean(axis=(0, 2)) - mu) / sd


def leg_energy_share(joint_shift):
    """Fraction of the standardized joint-shift energy carried by the most affected leg (0.25 even, 1 single leg)."""
    per_leg = np.array([np.sum(joint_shift[3 * i:3 * i + 3] ** 2) for i in range(4)])
    return per_leg / (per_leg.sum() + 1e-12)


def resolve_left_right(pair, joint_shift):
    """Within the R^- pair, the leg with the larger |residual mean shift| of the pair's joint rows (Block-I fix)."""
    legs = PAIR_LEGS[pair]; ji = None
    # pick the joint with the largest |shift| within the pair, then compare its two legs
    mags = {j: max(abs(joint_shift[3 * LEGS.index(l) + JOINTS.index(j)]) for l in legs) for j in JOINTS}
    joint = max(mags, key=mags.get)
    vals = {l: abs(joint_shift[3 * LEGS.index(l) + JOINTS.index(joint)]) for l in legs}
    leg = max(vals, key=vals.get)
    return leg, joint, {l: float(v) for l, v in vals.items()}


def readout(Z, Zr, Zb, rep_raw, rep_res, K_cal, chans, res_names, use_residual_for_rminus=True, base_z_thresh=3.0):
    """Assemble the three-channel reading. `Z` raw cycles, `Zr` residual joint cycles (12), `Zb` base residual cycles (6);
    rep_raw / rep_res the C2 reps; res_names the residual channel names."""
    rminus = rminus_state(Zr if use_residual_for_rminus else Z, rep_res if use_residual_for_rminus else rep_raw, K_cal,
                          res_names if use_residual_for_rminus else chans, groups=("res",) if use_residual_for_rminus else ("q", "dq", "tau_cmd", "tau_meas"))
    js = joint_row_shift(Zr, K_cal); base_raw, base_z = base_row_shift(Zb, K_cal)
    share = leg_energy_share(js)
    leg, joint, lr = resolve_left_right(rminus["pair"], js)
    # pair-level shift energy (F, H)
    pair_energy = {"F": float(np.sum(js[:3] ** 2) + np.sum(js[3:6] ** 2)), "H": float(np.sum(js[6:9] ** 2) + np.sum(js[9:12] ** 2))}
    # (pair, joint) cell energies: bilateral concentrates in one cell (both legs, one joint); drift spreads over cells
    cell = {}
    for p, (la, lb) in PAIR_LEGS.items():
        for j in JOINTS:
            cell[(p, j)] = js[3 * LEGS.index(la) + JOINTS.index(j)] ** 2 + js[3 * LEGS.index(lb) + JOINTS.index(j)] ** 2
    tot = sum(cell.values()) + 1e-12; dom_share = max(cell.values()) / tot
    return {"rminus": rminus, "joint_shift": js, "base_shift_raw": base_raw, "base_shift_z": base_z, "leg_share": share,
            "resolved_leg": leg, "resolved_joint": joint, "left_right": lr, "max_leg_share": float(share.max()),
            "base_fz_z": float(base_z[2]), "base_mx_z": float(base_z[3]), "base_fz_raw": float(base_raw[2]), "base_mx_raw": float(base_raw[3]),
            "pair_energy": pair_energy, "dominant_cell_share": float(dom_share)}


def decide(reading, rminus_alarmed, base_fz_thresh=2.0, base_mx_thresh=0.1, share_thresh=0.5, signal_thresh=1.0, concentration_thresh=0.7):
    """Decision rules (e09). Payload uses the RAW base momentum-residual shift in physical units (a 1 kg payload = Δm·g ≈
    9.8 N on f_z and Δm·g·offset on m_x, e05c) — the standardized z-score under-reads it because the base f_z row has a
    large nominal variance (that was the pre-registered-threshold miss: 1 kg gave z ≈ 2.1 < 3). `signal_thresh` is the
    smallest standardized joint-row shift that counts as a signal (quantifies 'quiet'); `pair_ratio` = the weaker mirror
    pair's shift energy relative to the stronger below which the signal is confined to one pair (bilateral) vs spread
    over both (drift). Hierarchy: payload (base) -> nominal -> R- silent (drift vs bilateral by pair spread) -> R-
    alarmed (single / pair). Returns (label, confidence, why)."""
    fzr, mxr = reading["base_fz_raw"], reading["base_mx_raw"]; share = reading["max_leg_share"]
    signal = float(np.max(np.abs(reading["joint_shift"]))); dom = reading["dominant_cell_share"]
    if abs(fzr) >= base_fz_thresh and abs(mxr) >= base_mx_thresh:
        return "payload_lateral", "base fz & mx (raw)", f"base fz={fzr:.1f} N, mx={mxr:.2f} N m"
    if abs(fzr) >= base_fz_thresh and abs(mxr) < base_mx_thresh:
        return "payload_symmetric", "base fz only (raw)", f"base fz={fzr:.1f} N, mx={mxr:.2f} N m"
    if not rminus_alarmed and signal < signal_thresh:
        return "nominal", "nothing shifted", f"max joint shift {signal:.2f} < {signal_thresh}"
    if not rminus_alarmed:
        # R- silent, something shifted: concentrated in one (pair, joint) cell = mirror-symmetric bilateral fault; spread
        # over cells (multiple joints, both pairs) = a common-mode change (symmetric torque/friction drift)
        if dom >= concentration_thresh:
            return "bilateral_mirror", "R- silent, one (pair,joint) cell", f"dominant cell share {dom:.2f}, share {share:.2f}"
        return "symmetric_drift_or_bilateral", "R- silent, spread over cells", f"dominant cell share {dom:.2f}, share {share:.2f}"
    # R- alarmed
    if share >= share_thresh:
        return f"single_leg:{reading['resolved_leg']}-{reading['resolved_joint']}", "R- + one-leg joint row", f"leg share {share:.2f}, {reading['left_right']}"
    return f"pair:{reading['rminus']['pair']}-{reading['rminus']['joint']}", "R- + spread joint rows", f"leg share {share:.2f} (left/right low-confidence)"
