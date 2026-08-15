"""Kinematic gait-phase estimator (Sprint 7 Block W3): phase from joint kinematics alone (no controller clock).

    theta_hat, info = estimate_phase(df, joint="KFE", contact_cols=None, fs=200.0)

Method. The trot's diagonal structure gives a clean antisymmetric gait signal
    s(t) = q_LF + q_RH - q_RF - q_LH          (knee angles by default; thigh angles for datasets without a knee signal)
whose fundamental is the gait frequency f0 (estimated from the spectrum of s, searched in [0.5, 4] Hz). s is band-passed
around f0 (zero-phase Butterworth, +-50 % of f0) and the analytic signal (Hilbert transform) gives the instantaneous
phase phi(t); theta_hat = ((phi - phi0) / 2pi) mod 1. The offset phi0 aligns theta = 0 with the LF touchdown: if contact
flags are given, phi0 = circular mean of phi at the LF stance-onset events (event correction; also usable with
current/torque-based contact proxies); otherwise phi0 = the offset at which the *template* trot has theta = 0 at the
knee-flexion cycle start — calibrated once on the simulator (info['phi0_source']). Errors are reported in turns
(circular); the Go2 simulator truth phase (controller clock) is the reference for the < 5 % of a period gate.

    circular_error(theta_hat, theta_true) -> per-sample |Delta theta| in turns (0..0.5)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

LEGS = ("LF", "RF", "LH", "RH")


def gait_signal(df, joint: str = "KFE", legs=LEGS) -> np.ndarray:
    q = {l: df[f"q_{l}_{joint}"].to_numpy() for l in legs}
    return q["LF"] + q["RH"] - q["RF"] - q["LH"]


def dominant_frequency(s: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 4.0) -> float:
    s = s - np.mean(s); n = len(s)
    w = np.hanning(n); S = np.abs(np.fft.rfft(s * w)) ** 2; f = np.fft.rfftfreq(n, d=1.0 / fs)
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(S[m])])


def estimate_phase(df, joint: str = "KFE", contact_cols=None, fs: float | None = None, band: float = 0.5,
                   phi0_default: float | None = None, t_col: str = "t", method: str = "linear", window_cycles: float = 10.0) -> tuple[np.ndarray, dict]:
    """method: 'linear' (default) — the unwrapped Hilbert phase is replaced by a piecewise-linear phase clock (local
    least-squares fit over `window_cycles` cycles): constant rate within the window, so the phase map has no
    within-cycle warping (a warping that is not exactly half-period-equivariant registers the two halves of the record
    slightly differently and shows up in the H0' differenced test — W3 finding); 'hilbert' — raw instantaneous phase."""
    t = df[t_col].to_numpy(); fs = float(fs or 1.0 / np.median(np.diff(t)))
    s = gait_signal(df, joint)
    f0 = dominant_frequency(s, fs)
    lo, hi = max(0.05, f0 * (1 - band)), min(0.49 * fs, f0 * (1 + band))
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    sf = filtfilt(b, a, s - s.mean())
    phi_raw = np.unwrap(np.angle(hilbert(sf)))
    if method == "linear":
        # local linear fit of the unwrapped phase: rate b(t) and offset from a sliding window of `window_cycles` periods
        n = len(t); w = int(round(window_cycles / f0 * fs)); w = max(min(w, n), 8)
        phi = np.empty(n)
        # global fit as the fallback / for short records
        A = np.vstack([np.ones(n), t - t[0]]).T; coef, *_ = np.linalg.lstsq(A, phi_raw, rcond=None); phi_g = A @ coef
        if w >= n:
            phi = phi_g
        else:
            # centred moving windows (edges use the nearest full window)
            half = w // 2
            for i in range(n):
                a0 = min(max(i - half, 0), n - w); b0 = a0 + w
                tt = t[a0:b0] - t[a0]; pp = phi_raw[a0:b0]
                Aw = np.vstack([np.ones(w), tt]).T; c, *_ = np.linalg.lstsq(Aw, pp, rcond=None)
                phi[i] = c[0] + c[1] * (t[i] - t[a0])
        info_extra = {"method": "linear", "window_cycles": window_cycles, "rate_hz_global": float(coef[1] / (2 * np.pi))}
    else:
        phi = phi_raw; info_extra = {"method": "hilbert"}
    info = {"f0_hz": f0, "period_s": 1.0 / f0, "fs": fs, "band_hz": (lo, hi), **info_extra}
    if contact_cols:
        c = df[contact_cols[0]].to_numpy()                       # LF contact flag: stance onset = rising edge
        on = np.where((c[1:] > 0.5) & (c[:-1] <= 0.5))[0] + 1
        if len(on) >= 3:
            ph = np.mod(phi[on], 2 * np.pi)
            phi0 = float(np.angle(np.mean(np.exp(1j * ph)))); info["phi0_source"] = f"LF stance-onset events (n={len(on)})"
        else:
            phi0 = phi0_default if phi0_default is not None else 0.0; info["phi0_source"] = "default (too few contact events)"
    else:
        phi0 = phi0_default if phi0_default is not None else PHI0_TEMPLATE; info["phi0_source"] = "template calibration"
    theta = np.mod((phi - phi0) / (2 * np.pi), 1.0)
    info["phi0"] = float(np.mod(phi0, 2 * np.pi))
    return theta, info


# offset of the analytic-signal phase of s = q_LF + q_RH - q_RF - q_LH (knee) at controller phase 0 for the S0/D004 trot
# template (lift on the knee: LF swings in [0.5, 1) with a sin^2 bump): calibrated on go2_urdf_sym (see W3 report).
PHI0_TEMPLATE = -1.4977      # calibrate_phi0 on go2_urdf_sym trot in place, seed 1 (2026-08-16)


def circular_error(theta_hat: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    d = np.mod(theta_hat - theta_true + 0.5, 1.0) - 0.5
    return np.abs(d)


def calibrate_phi0(df, theta_true: np.ndarray, joint: str = "KFE", fs: float | None = None) -> float:
    """Circular-mean offset between the analytic phase of the gait signal and the true controller phase (used once, on
    the simulator, to set PHI0_TEMPLATE)."""
    theta, info = estimate_phase(df, joint=joint, fs=fs, phi0_default=0.0)
    d = np.mod(theta - theta_true, 1.0) * 2 * np.pi
    return float(np.angle(np.mean(np.exp(1j * d))))
