"""Per-stance-event statistics for the estimation channel (Sprint 8 Block G).

The R- test conditions on the gait cycle; the estimation gate conditions on the STANCE EVENT: every foot touch-down opens
a test window that lasts while the foot is in contact. For each such window a per-leg anomaly SCORE is accumulated (the
whitened kinematic contact-constraint residual: the InEKF per-foot innovation, or the foot's estimated world-frame
velocity), turned into a conformal p-value against a NOMINAL event library (calibration touch-down events), and combined
across the window into an e-value. The e-values feed a per-leg time-varying weight (estimate/pi_gating). FAR is counted
PER EVENT (a nominal event whose conformal p <= alpha is a false alarm). Wheeled version (M1): the "stance event" is the
continuous rolling contact and the score is the whitened rolling-constraint residual ||d_dot_hat - R u|| (same interface).

    lib = EventLibrary()                       # nominal score library, per leg
    lib.add(leg, score)                        # during a nominal run
    p = lib.conformal_p(leg, score)            # one-sided conformal p of a monitoring score
    ev = StanceEventTracker(n_legs=4, alpha=0.05, kind="sqrt")
    ev.update(leg, in_contact, score, lib)     # per control step -> maintains a per-leg e-process over stance events
    w = ev.weights()                           # per-leg gating weight built from the current e-values
"""
from __future__ import annotations

import numpy as np

from .evalue import p_to_e
from .monitors import conformal_pvalues


class EventLibrary:
    """Per-leg library of nominal stance-event scores; conformal p-values are computed against it."""

    def __init__(self, n_legs: int = 4):
        self.scores: list[list[float]] = [[] for _ in range(n_legs)]

    def add(self, leg: int, score: float) -> None:
        if np.isfinite(score):
            self.scores[leg].append(float(score))

    def finalize(self) -> "EventLibrary":
        self.arr = [np.sort(np.asarray(s)) if s else np.array([0.0]) for s in self.scores]
        return self

    def conformal_p(self, leg: int, score: float) -> float:
        cal = getattr(self, "arr", None)
        cal = cal[leg] if cal is not None else np.sort(np.asarray(self.scores[leg] or [0.0]))
        return float(conformal_pvalues(cal, np.array([score]))[0])


class StanceEventTracker:
    """Per-leg e-process over stance EVENTS. A stance event = a maximal contiguous in-contact window. The event's score
    is the mean of its per-step whitened residuals; on the event's close its conformal p (against the library) becomes an
    e-value e = p_to_e(p) and multiplies the leg's running e-process. `weights()` maps the current per-leg e-value to a
    gating weight (see estimate/pi_gating). Per-event because a per-step test would over-count the correlated stance
    samples; FAR is then controlled per event."""

    def __init__(self, n_legs: int = 4, alpha: float = 0.05, kind: str = "sqrt", kappa: float = 0.5, decay: float = 1.0):
        self.n = n_legs; self.alpha = alpha; self.kind = kind; self.kappa = kappa; self.decay = decay
        self.E = np.ones(n_legs)                 # running e-process per leg (product of event e-values)
        self.e_cur = np.ones(n_legs)             # e-value of the CURRENT open event (updated live from the running mean)
        self._acc = [[] for _ in range(n_legs)]  # scores of the currently-open event
        self._in = np.zeros(n_legs, bool)
        self.events: list[dict] = []             # closed-event log (leg, t0, t1, score, p, e)

    def update(self, leg: int, in_contact: bool, score: float, lib: EventLibrary, t: float | None = None) -> None:
        if in_contact:
            if np.isfinite(score):
                self._acc[leg].append(float(score))
            self._in[leg] = True
            if self._acc[leg]:
                p = lib.conformal_p(leg, float(np.mean(self._acc[leg])))
                self.e_cur[leg] = float(p_to_e(np.array([p]), self.kind, self.kappa)[0])   # live e for the open event
        elif self._in[leg]:                       # event just closed
            if self._acc[leg]:
                sc = float(np.mean(self._acc[leg])); p = lib.conformal_p(leg, sc); e = float(p_to_e(np.array([p]), self.kind, self.kappa)[0])
                self.E[leg] = self.E[leg] ** self.decay * e; self.events.append({"leg": leg, "t": t, "score": sc, "p": p, "e": e})
            self._acc[leg] = []; self._in[leg] = False; self.e_cur[leg] = 1.0

    def current_e(self) -> np.ndarray:
        """Per-leg e-value in force NOW: the open event's live e (if in contact) times the closed-event e-process."""
        return self.E * np.where(self._in, self.e_cur, 1.0)

    def weights(self, mode: str = "soft") -> np.ndarray:
        """Per-leg gating weight from the current e-value. hard: 0 if e >= 1/alpha else 1. soft: 1/(1+e) in (0,1]."""
        e = self.current_e()
        if mode == "hard":
            return (e < 1.0 / self.alpha).astype(float)
        return 1.0 / (1.0 + np.maximum(e - 1.0, 0.0))
