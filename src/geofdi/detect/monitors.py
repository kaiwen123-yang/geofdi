"""Online monitors built from the S1 tests: R^- (mirror) and R^+ (magnitude) channels, e-process / e-CUSUM alarms,
per-channel isolation energies with swing-phase conditioning, and the strict 'unfolded' mirror test.

Calibration discipline (e04): every monitor is calibrated on NOMINAL cycles only (the pre-onset segment); fault
cycles are only evaluated. Nothing in here looks at the fault labels.

R^- channel  : Hemerik–Goeman mirror test per window of `window` cycles (statistic paired_energy or
               energy_distance) -> p_w -> e_w = p_w^{-1/2}/2 -> running product E_t (Ville alarm at 1/alpha) and
               e-CUSUM S_t = max(0, S_{t-1} + log e_w) with a threshold h calibrated on nominal windows.
R^+ channel  : per-cycle magnitude score (default: sum over legs of the phase-binned L2 tracking error
               ||q - q_ref||, a mirror-INVARIANT quantity), conformal p per cycle against the calibration
               cycles' scores, then the same e-process / e-CUSUM.
Isolation    : per (leg, joint) group, the standardized R^- projection energy ||mean_k D_k[c,:]||^2 summed over
               the group's channels; optional swing-phase conditioning restricts the phase grid of each leg's
               channels to that leg's swing segment (partner is in swing at theta+1/2 as well).
Unfolded test: elements built from disjoint 3-cycle chunks: X_j = cycle 3j, Y_j = rho_g[B_{3j+1} | A_{3j+2}] (the
               true half-period shift + mirror), sign flips on D_j = X_j - Y_j (Remark rem:wrap strict version).
"""
from __future__ import annotations

import numpy as np

from .evalue import p_to_e
from .permutation import hg_permutation_tests, pooled_scale, random_flips

LEGS = ("LF", "RF", "LH", "RH")
JOINTS = ("HAA", "HFE", "KFE")
# swing segments of the template gait (duty 0.5): LF swings in [0.5,1), LH = LF(theta+1/2) -> [0,0.5),
# RF = S*LF(theta+1/2) -> [0,0.5), RH = S*LH(theta+1/2) -> [0.5,1)
SWING = {"LF": (0.5, 1.0), "RF": (0.0, 0.5), "LH": (0.0, 0.5), "RH": (0.5, 1.0)}


# ------------------------------------------------------------------------------------------ alarms
def eprocess_alarm(pvals, alpha: float, start: int = 0):
    """Running product of e-values from window `start` on; returns (E array, first alarm window index or None)."""
    e = p_to_e(np.asarray(pvals, dtype=float))
    E = np.cumprod(e[start:])
    hits = np.where(E >= 1.0 / alpha)[0]
    return E, (int(hits[0]) + start if len(hits) else None)


def ecusum(pvals, h: float, start: int = 0):
    """Page/e-CUSUM: S_t = max(0, S_{t-1} + log e_t); alarm when S_t >= h. Returns (S array, alarm index or None)."""
    le = np.log(p_to_e(np.asarray(pvals, dtype=float)))
    S = np.zeros(len(le)); s = 0.0
    for i in range(start, len(le)):
        s = max(0.0, s + le[i]); S[i] = s
    hits = np.where(S >= h)[0]
    hits = hits[hits >= start]
    return S, (int(hits[0]) if len(hits) else None)


def calibrate_ecusum_threshold(nominal_pval_runs, horizon: int, far: float = 0.05, n_boot: int = 2000,
                               rng: np.random.Generator | None = None) -> float:
    """h such that P(max_{t<=horizon} S_t >= h) ~= far on nominal windows: block-bootstrap sequences of length
    `horizon` from the pooled nominal window p-values (each run's sequence is used as a block source)."""
    rng = np.random.default_rng() if rng is None else rng
    runs = [np.asarray(r, dtype=float) for r in nominal_pval_runs if len(r) > 0]
    maxes = []
    for _ in range(n_boot):
        seq = []
        while len(seq) < horizon:
            r = runs[rng.integers(len(runs))]
            L = min(len(r), horizon - len(seq)); a = rng.integers(0, len(r) - L + 1)
            seq.extend(r[a:a + L])
        S, _ = ecusum(seq, h=np.inf)
        maxes.append(S.max())
    return float(np.quantile(maxes, 1 - far))


# ------------------------------------------------------------------------------------------ R^- channel
class MirrorMonitor:
    def __init__(self, rep, window: int = 5, M: int = 512, statistic: str = "paired_energy", alpha: float = 0.05,
                 block_len: int = 1):
        self.rep, self.window, self.M, self.statistic, self.alpha, self.block_len = rep, window, M, statistic, alpha, block_len

    def window_pvalues(self, Z: np.ndarray, seed: int = 0) -> np.ndarray:
        K = Z.shape[0]; nw = K // self.window; ps = np.empty(nw)
        for w in range(nw):
            rng = np.random.default_rng([seed, w])
            res = hg_permutation_tests(Z[w * self.window:(w + 1) * self.window], self.rep, statistics=(self.statistic,),
                                       M=self.M, rng=rng, block_len=self.block_len)
            ps[w] = res[self.statistic]["p"]
        return ps


# ------------------------------------------------------------------------------------------ R^+ channel
def tracking_scores(Z: np.ndarray, names: list[str], qref: np.ndarray, per_leg: bool = False) -> np.ndarray:
    """Per-cycle magnitude score from phase-registered q (in Z) and q_ref (K,12,N): sqrt(mean over joints & phase of
    (q - q_ref)^2) per leg; per_leg -> (K,4), else summed over legs (K,) — a mirror-invariant R^+ score."""
    idx = {n: i for i, n in enumerate(names)}
    K = Z.shape[0]; s = np.zeros((K, 4))
    for li, leg in enumerate(LEGS):
        cols = [idx[f"q_{leg}_{j}"] for j in JOINTS]
        err = Z[:, cols, :] - qref[:, 3 * li:3 * li + 3, :]
        s[:, li] = np.sqrt((err ** 2).mean(axis=(1, 2)))
    return s if per_leg else s.sum(axis=1)


def conformal_pvalues(cal_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """One-sided conformal p per score: (1 + #{cal >= s}) / (n_cal + 1)."""
    cal = np.sort(np.asarray(cal_scores)); n = len(cal)
    ge = n - np.searchsorted(cal, np.asarray(scores), side="left")
    return (1.0 + ge) / (n + 1.0)


# ------------------------------------------------------------------------------------------ isolation
def channel_projection_energy(Z: np.ndarray, rep, names: list[str], swing_condition: bool = False,
                              groups=("q", "dq", "tau_cmd", "tau_meas"), Z_cal: np.ndarray | None = None) -> dict:
    """||mean_k D_k[c, :]||^2 per channel on standardized D = Z - rho Z, aggregated per (leg, joint) and per
    (pair, joint) over the given channel groups; with swing_condition only the phase samples in the channel's leg
    swing segment count. Scale: if Z_cal is given, each channel's D is divided by its NOMINAL residual std over the
    calibration cycles (energy relative to the null fluctuation — the isolation statistic); otherwise by the pooled
    within-cycle std (the flip-invariant scale of the detection test, which over-weights low-variance channels).
    Note: D_c and D_partner(c) have identical energy by construction, so R^- ranks (pair, joint), not left/right."""
    Zs = rep.apply("s", Z)
    D = Z - Zs
    if Z_cal is not None:
        Dc = Z_cal - rep.apply("s", Z_cal)
        sc = Dc.transpose(1, 0, 2).reshape(Dc.shape[1], -1).std(axis=1)[:, None] + 1e-9
    else:
        sc = pooled_scale(Z, Zs)
    D = D / sc                                                    # (K, d, N)
    N = Z.shape[-1]; theta = (np.arange(N) + 0.5) / N
    per_channel, per_group, per_pair = {}, {}, {}
    pair_of = {"LF": "F", "RF": "F", "LH": "H", "RH": "H"}
    for i, n in enumerate(names):
        g = rep.groups.get(n)
        parts = n.split("_")
        if g in groups and len(parts) >= 3:
            leg, joint = parts[-2], parts[-1]
        else:
            continue
        mask = np.ones(N, dtype=bool)
        if swing_condition:
            a, b = SWING[leg]; mask = (theta >= a) & (theta < b)
        e = float((D[:, i, mask].mean(axis=0) ** 2).sum())
        per_channel[n] = e; per_group[(leg, joint)] = per_group.get((leg, joint), 0.0) + e
        per_pair[(pair_of[leg], joint)] = per_pair.get((pair_of[leg], joint), 0.0) + e
    return {"per_group": per_group, "per_pair": per_pair, "per_channel": per_channel}


def leg_magnitude_deviation(scores_cal: np.ndarray, scores_post: np.ndarray) -> np.ndarray:
    """R^+ left/right resolution: per-leg deviation of the post-onset tracking score from calibration, studentized by
    the calibration std POOLED over the mirror pair (both legs have the same law under H0; per-leg sample stds from
    60 cycles are too noisy). Columns follow LEGS = (LF, RF, LH, RH)."""
    mu = scores_cal.mean(axis=0)
    sd = np.sqrt(0.5 * (scores_cal.var(axis=0) + scores_cal.var(axis=0)[[1, 0, 3, 2]])) + 1e-9
    return (scores_post.mean(axis=0) - mu) / sd


def rank_groups(per_group: dict) -> list:
    return sorted(per_group.items(), key=lambda kv: -kv[1])


# ------------------------------------------------------------------------------------------ unfolded (strict) test
class UnfoldedPairs:
    """Build (X_j, Y_j) from disjoint 3-cycle chunks and evaluate flip statistics on D_j = X_j - Y_j."""

    def __init__(self, Z: np.ndarray, rep, standardize: bool = True):
        K, d, N = Z.shape; h = N // 2
        J = K // 3
        X = np.stack([Z[3 * j] for j in range(J)])
        Y = np.stack([rep.mirror_only(np.concatenate([Z[3 * j + 1][:, h:], Z[3 * j + 2][:, :h]], axis=-1)) for j in range(J)])
        scale = pooled_scale(X, Y) if standardize else 1.0
        self.J = J
        self.X = (X / scale).reshape(J, -1); self.Y = (Y / scale).reshape(J, -1)
        Dm = self.X - self.Y; self.G = Dm @ Dm.T
        self._Dist = None

    def paired_energy(self, flips):
        f = np.asarray(flips, dtype=float); return np.einsum("mk,kl,ml->m", f, self.G, f) / (self.J ** 2)

    def energy_distance(self, flips):
        if self._Dist is None:
            P = np.concatenate([self.X, self.Y]); sq = (P * P).sum(1)
            self._Dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))
        Dm = self._Dist; J = self.J; f = np.asarray(flips); out = np.empty(f.shape[0]); ar = np.arange(J)
        for m in range(f.shape[0]):
            a = np.where(f[m] > 0, ar, J + ar); b = np.where(f[m] > 0, J + ar, ar)
            out[m] = 2 * Dm[np.ix_(a, b)].mean() - Dm[np.ix_(a, a)].sum() / (J * (J - 1)) - Dm[np.ix_(b, b)].sum() / (J * (J - 1))
        return out


def unfolded_permutation_tests(Z: np.ndarray, rep, statistics=("paired_energy", "energy_distance"), M: int = 512,
                               rng: np.random.Generator | None = None) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    up = UnfoldedPairs(Z, rep)
    flips = random_flips(up.J, M - 1, rng)
    out = {}
    for s in statistics:
        fn = getattr(up, s)
        obs = fn(np.ones((1, up.J)))[0]; null = fn(flips)
        out[s] = {"p": (1.0 + np.sum(null >= obs)) / M, "obs": float(obs), "n_elements": up.J}
    return out
