#!/usr/bin/env python3
"""Block W1 smoke report for the wheeled M1 worlds: rolling 0.5/1.0/2.0 m/s x 60 s on both worlds (stability, contact
fraction, wheel-load symmetry, drift), stepping mode 30 s, mirror-sim residuals (t01: sym world exact, original world =
eps_dyn candidate). Writes results/m1_wheeled_smoke/<run_id>/smoke.json + smoke.md.

    python scripts/m1_wheeled_smoke.py [--run-id ID]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tests"))
from geofdi.sim.env_m1 import SimConfigM1, rollout_m1                # noqa: E402
from geofdi.sim.telemetry_m1 import LEGS                              # noqa: E402
from test_m1_wheeled import _mirror_error, _pair                      # noqa: E402


def euler(s):
    w, x, y, z = s.base_qw.to_numpy(), s.base_qx.to_numpy(), s.base_qy.to_numpy(), s.base_qz.to_numpy()
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))); pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S")); a = ap.parse_args()
    out = REPO / "results" / "m1_wheeled_smoke" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    res = {"rolling": [], "stepping": [], "t01": {}}
    for model in ("m1_wheeled_sym", "m1_wheeled"):
        for sp in (0.5, 1.0, 2.0):
            df, man = rollout_m1(SimConfigM1(model=model, speed=sp, duration_s=60.0, seed=1, controller={"ramp_s": 3.0}))
            s = df[df.t > 6.0]; roll, pitch, yaw = euler(s); fz = s[[f"fc_z_{l}" for l in LEGS]]
            res["rolling"].append({"model": model, "speed": sp, "vx_mean": float(s.base_vx.mean()), "vx_std": float(s.base_vx.std()), "z_mean": float(s.base_z.mean()),
                                   "roll_mean_deg": float(roll.mean()), "roll_std_deg": float(roll.std()), "pitch_mean_deg": float(pitch.mean()), "pitch_std_deg": float(pitch.std()),
                                   "yaw_drift_deg": float(yaw[-1] - yaw[0]), "y_drift_m": float(s.base_y.iloc[-1] - s.base_y.iloc[0]),
                                   "contact_fraction": s[[f"c_{l}" for l in LEGS]].mean().round(4).tolist(), "fz_mean_N": fz.mean().round(2).tolist(), "fz_std_N": fz.std().round(2).tolist(),
                                   "wheel_dq_mean": s[[f"dq_{l}_WHEEL" for l in LEGS]].mean().round(3).tolist(), "fell": bool(df.base_z.min() < 0.25)})
            print(res["rolling"][-1], flush=True)
    for model in ("m1_wheeled_sym",):
        df, man = rollout_m1(SimConfigM1(model=model, mode="stepping", speed=0.0, duration_s=30.0, seed=1))
        s = df[df.t > 5.0]; roll, pitch, yaw = euler(s)
        res["stepping"].append({"model": model, "z_min": float(s.base_z.min()), "z_max": float(s.base_z.max()), "roll_std_deg": float(roll.std()), "roll_max_deg": float(np.abs(roll).max()),
                                "pitch_mean_deg": float(pitch.mean()), "pitch_std_deg": float(pitch.std()), "x_drift_m": float(df.base_x.iloc[-1]), "y_drift_m": float(df.base_y.iloc[-1]),
                                "contact_fraction": s[[f"c_{l}" for l in LEGS]].mean().round(3).tolist(), "fell": bool(df.base_z.min() < 0.25),
                                "params": "stance (0,0.8,-1.5), lift 0.30/0.10, kp 200, kd 6, period 0.5 s, wheels position-held (kp 20, kv 2)"})
        print(res["stepping"][-1], flush=True)
    for model in ("m1_wheeled_sym", "m1_wheeled"):
        A, B, man = _pair(model, duration=6.0); err, worst = _mirror_error(A, B, man)
        res["t01"][model] = {"max_mirror_residual": err, "worst_channels": worst}
        print(model, "t01 residual", err, flush=True)
    (out / "smoke.json").write_text(json.dumps(res, indent=1))
    lines = ["# wheeled M1 smoke (Block W1)", "", "| model | speed | vx | roll mean±std | pitch mean±std | yaw drift/54s | y drift | contact fraction | fz mean [N] | fz std |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rolling"]:
        lines.append(f"| {r['model']} | {r['speed']} | {r['vx_mean']:.3f} | {r['roll_mean_deg']:.2f}±{r['roll_std_deg']:.2f} | {r['pitch_mean_deg']:.2f}±{r['pitch_std_deg']:.2f} | {r['yaw_drift_deg']:.2f}° | {r['y_drift_m']:.2f} m | {r['contact_fraction']} | {r['fz_mean_N']} | {r['fz_std_N']} |")
    lines += ["", "stepping: " + json.dumps(res["stepping"]), "", "t01: " + json.dumps({k: v["max_mirror_residual"] for k, v in res["t01"].items()})]
    (out / "smoke.md").write_text("\n".join(lines) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
