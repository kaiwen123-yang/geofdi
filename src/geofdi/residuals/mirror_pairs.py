"""The RESIDUAL data element and its C2 representation rho_R (theory Part 2, Definition def:residual-element).

r = y - f_hat(x, u) is a torque-valued signal on the 12 joints (+ optionally the 6 floating-base rows of the momentum
observer: body-frame force f (polar, E) and moment m (axial, -E)). Phase-registered per cycle it becomes the residual
data element R_k in R^{d_r x N}; under an equivariant nominal model it inherits H0 (Proposition prop:residual-h0), so the
SAME Hemerik-Goeman flip test acts on it through this representation. Signs/partners are those of the torque channels
(residual rows are conjugate to joint coordinates: rho_R = rho_U on the joint block), read through the telemetry
manifest conventions (JOINT_SIGN, IMU_ACC_SIGN, IMU_GYRO_SIGN) so that no second sign table exists.

    man = residual_manifest(include_base=False)      # channels res_<leg>_<joint> (+ res_base_f*/m*)
    rep = C2Rep(man)                                   # rep.apply("s", R) on (K, d_r, N) residual cycles
    RES_COLS, BASE_COLS                                # column names used by detect.rplus / e05
"""
from __future__ import annotations

from ..groups.c2 import C2Rep
from ..sim.telemetry import IMU_ACC_SIGN, IMU_GYRO_SIGN, JOINT_SIGN, JOINTS, LEGS, MIRROR_LEG

RES_COLS = [f"res_{l}_{j}" for l in LEGS for j in JOINTS]
BASE_COLS = ["res_base_fx", "res_base_fy", "res_base_fz", "res_base_mx", "res_base_my", "res_base_mz"]


def residual_manifest(include_base: bool = False) -> dict:
    ch = []
    for leg in LEGS:
        for j in JOINTS:
            ch.append({"name": f"res_{leg}_{j}", "group": "res", "leg": leg, "joint": j, "kind": "scalar-signed",
                       "partner": f"res_{MIRROR_LEG[leg]}_{j}", "sign": JOINT_SIGN[j], "in_Z": True})
    if include_base:
        for ax in "xyz":     # base force: polar vector in the body frame -> E; base moment: axial -> -E
            ch.append({"name": f"res_base_f{ax}", "group": "res_base", "leg": None, "joint": None, "kind": "polar",
                       "partner": f"res_base_f{ax}", "sign": IMU_ACC_SIGN[ax], "in_Z": True})
        for ax in "xyz":
            ch.append({"name": f"res_base_m{ax}", "group": "res_base", "leg": None, "joint": None, "kind": "axial",
                       "partner": f"res_base_m{ax}", "sign": IMU_GYRO_SIGN[ax], "in_Z": True})
    return {"schema": "geofdi-residual-element-v1", "leg_order": list(LEGS), "joint_order": list(JOINTS),
            "gait_group": {"G": "C2 sagittal reflection", "Sigma": "{(e,0),(g_s,1/2)}", "delta_theta": 0.5}, "channels": ch}


def residual_rep(include_base: bool = False) -> C2Rep:
    return C2Rep(residual_manifest(include_base))


def isotypic_split(R, rep: C2Rep):
    """Pi+ R, Pi- R for cycles R (K, d, N): the trivial / sign isotypic components under rho(sigma_*)
    (mirror + half-period shift), Pi+- = (I +- rho)/2. Pi- carries what the R- test sees, Pi+ what magnitude channels see."""
    Rs = rep.apply("s", R)
    return 0.5 * (R + Rs), 0.5 * (R - Rs)
