"""C2 = {e, g_s} sagittal-reflection representation on the telemetry data element, built from a channel manifest.

The manifest (geofdi.sim.telemetry.build_manifest / assets/channels_m1.yaml) lists, per channel, its mirror
partner and sign: (rho(g_s) Z)[partner] = sign * Z[channel]. Kinds: 'scalar-signed' (joint quantities: partner
= same joint of the mirror leg, sign = joint sign, HAA -1 / HFE +1 / KFE +1 in the uniform-axis convention),
'polar' (accelerometer: a -> E a with E = diag(1,-1,1)), 'axial' (gyro: w -> -E w, theory eq:imu-action),
'scalar-magnitude' (contacts, temperatures: pure leg permutation). The gait element sigma_* = (g_s, 1/2)
additionally shifts the phase by half a period: (rho(sigma) Z)(theta) = rho(g_s) Z(theta + 1/2).

    rep = C2Rep(manifest)               # rep.P is the explicit d x d signed permutation matrix
    Zm = rep.apply("s", Z)              # Z: (..., d, N) phase-registered cycles; "e" is the identity
    rep.mirror_only(Z), rep.shift(Z, 0.5)
"""
from __future__ import annotations

import numpy as np
import yaml


class C2Rep:
    def __init__(self, manifest: dict, gyro_sign_bug: bool = False):
        """gyro_sign_bug=True deliberately uses the WRONG (polar, +E) transformation for the gyro — the
        negative control of tests/test_representations.py; never use in analysis."""
        chans = [c for c in manifest["channels"] if c.get("in_Z", True)]
        self.names = [c["name"] for c in chans]
        self.index = {n: i for i, n in enumerate(self.names)}
        d = len(self.names)
        P = np.zeros((d, d))
        for c in chans:
            i = self.index[c["name"]]
            j = self.index[c["partner"]]
            sign = float(c["sign"])
            if gyro_sign_bug and c.get("kind") == "axial":
                sign = -sign          # +E instead of -E on axial channels
            P[j, i] = sign
        self.P = P
        self.d = d
        self.dtheta = float(manifest.get("gait_group", {}).get("delta_theta", 0.5))
        self.kinds = {c["name"]: c.get("kind") for c in chans}
        self.groups = {c["name"]: c.get("group") for c in chans}
        assert np.allclose(P @ P, np.eye(d)), "rho(g_s) must be an involution"

    @classmethod
    def from_yaml(cls, path, **kw) -> C2Rep:
        return cls(yaml.safe_load(open(path)), **kw)

    def mirror_only(self, Z: np.ndarray) -> np.ndarray:
        """Apply rho(g_s) along the channel axis (axis -2 of (..., d, N), or axis -1 of (..., d))."""
        Z = np.asarray(Z)
        if Z.shape[-1] == self.d and (Z.ndim == 1 or Z.shape[-2] != self.d):
            return Z @ self.P.T
        return np.einsum("ij,...jn->...in", self.P, Z)

    def shift(self, Z: np.ndarray, dtheta: float | None = None) -> np.ndarray:
        """Phase shift theta -> theta + dtheta on the last axis (N-point grid): Z'(theta) = Z(theta + dtheta)."""
        dt = self.dtheta if dtheta is None else dtheta
        N = Z.shape[-1]
        k = round(dt * N)
        if abs(dt * N - k) > 1e-9:
            raise ValueError(f"phase shift {dt} is not a multiple of the grid step 1/{N}")
        return np.roll(Z, -k, axis=-1)

    def apply(self, sigma: str, Z: np.ndarray) -> np.ndarray:
        """sigma in {'e', 's'}: identity or the gait element (g_s, 1/2) on cycles Z of shape (..., d, N)."""
        if sigma == "e":
            return np.asarray(Z).copy()
        if sigma == "s":
            return self.mirror_only(self.shift(Z))
        raise ValueError(sigma)

    def channel_indices(self, group: str | None = None, kind: str | None = None) -> np.ndarray:
        idx = [i for i, n in enumerate(self.names)
               if (group is None or self.groups[n] == group) and (kind is None or self.kinds[n] == kind)]
        return np.array(idx, dtype=int)
