"""Equivariant DeLaN by mirror weight sharing (Sprint 6, Block Q; theory Part 2, Proposition prop:equiv-delan).

The plain DeLaN (delan.py) trains one independent Lagrangian net per leg, so nothing forces
    f_RF(q, dq, ddq, a) == S f_LF(S q, S dq, S ddq, E a)          (S = joint signs, E = diag(1,-1,1))
and the learned nominal model has an equivariance defect delta_f > 0: its residual is NOT symmetric in law under H0
(the learned asymmetry contaminates the residual data element, Part 2 Corollary cor:contamination).

Here a leg pair shares ONE template net f_0; the mirrored leg is DEFINED through the template,
    f_sigma(z) := rho_Y(sigma)^{-1} f_0(rho_X(sigma) z, rho_U(sigma) u)      (Part 2, eq. eq:mirror-template)
i.e. transform the inputs into the template's leg, evaluate, transform the output back. All signs come from the channel
manifest through groups.c2 (JOINT signs of the q channels, polar signs of imu_a) — no second sign table exists here.
Consequences (proved in Part 2): (i) exact C2 equivariance, delta_f = 0 up to floating point; (ii) M_sigma(q) =
S M_0(Sq) S is congruent to an SPD matrix -> SPD; the mirrored Lagrangian L_sigma(q,dq;a) = L_0(Sq,Sdq;Ea) is a
Lagrangian, so the Euler-Lagrange structure of the template is inherited exactly; (iii) the fixed passive terms
b dq + f tanh(dq/0.05) are equivariant because S is diagonal +-1 and tanh is odd.

Templates: n_templates = 2 (front pair LF/RF -> template "F", hind pair LH/RH -> "H"; the Go2/M1 have no front-hind
symmetry, so C2 only relates left and right) or n_templates = 1 (all four legs on one template: LH uses the template
UNTRANSFORMED — an ablation that additionally assumes identical front and hind leg subsystems, which is not a
morphological symmetry of the robot).

    quad = EquivariantDeLaN.build(n_templates=2, device="cuda")
    train_equivariant(quad, {leg: {"train": {...}, "val": {...}}}, epochs=40, ...)
    quad.predict("RF", q, dq, ddq, a)                    # same numpy API as DeLaNQuadruped.predict
    equivariance_defect(model, q, dq, ddq, a)             # per-sample ||f_RF(z) - S f_LF(Sz)|| for ANY model
    load_delan(path)                                      # dispatch: plain or equivariant from meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..groups.c2 import C2Rep
from ..sim.telemetry import JOINTS, LEGS, build_manifest
from .delan import LEG_ORDER, DeLaNQuadruped, LegDeLaN, beta_hat, train_leg

TEMPLATE_MAPS = {
    2: {"LF": ("F", False), "RF": ("F", True), "LH": ("H", False), "RH": ("H", True)},
    1: {"LF": ("A", False), "RF": ("A", True), "LH": ("A", False), "RH": ("A", True)},
}


def mirror_maps(manifest: dict | None = None) -> dict:
    """Signs and partners of the sagittal reflection, read from the channel manifest via groups.c2 (single source of
    truth): partner[leg], S[leg] (3,) joint signs (order JOINTS), E (3,) accelerometer (polar) signs, Eg (3,) gyro
    (axial) signs. (rho(g_s) Z)[partner] = sign * Z[channel]."""
    man = build_manifest() if manifest is None else manifest
    rep = C2Rep(man)
    by_name = {c["name"]: c for c in man["channels"]}
    partner, S = {}, {}
    for leg in LEGS:
        c0 = by_name[f"q_{leg}_{JOINTS[0]}"]
        partner[leg] = c0["partner"].split("_")[1]
        s = []
        for j in JOINTS:
            c = by_name[f"q_{leg}_{j}"]
            i, k = rep.index[c["name"]], rep.index[c["partner"]]
            s.append(float(rep.P[k, i]))
        S[leg] = np.array(s)
    E = np.array([float(rep.P[rep.index[f"imu_a_{ax}"], rep.index[f"imu_a_{ax}"]]) for ax in "xyz"])
    Eg = np.array([float(rep.P[rep.index[f"imu_w_{ax}"], rep.index[f"imu_w_{ax}"]]) for ax in "xyz"])
    return {"partner": partner, "S": S, "E": E, "Eg": Eg}


def _to_dev(x, ref):
    return torch.as_tensor(np.asarray(x), dtype=ref.dtype, device=ref.device)


class EquivariantDeLaN:
    """Mirror-shared DeLaN: templates {key: LegDeLaN}, leg_map {leg: (key, mirrored)}, maps from mirror_maps()."""

    def __init__(self, templates: dict, leg_map: dict, maps: dict, meta: dict):
        self.templates = templates; self.leg_map = leg_map; self.maps = maps; self.meta = meta
        self.equivariant = True

    # -- construction / (de)serialisation ------------------------------------------------------------
    @staticmethod
    def build(n_templates: int = 2, hidden: int = 128, depth: int = 3, eps: float = 1e-3, damping: float = 0.01,
              frictionloss: float = 0.2, device: str = "cpu", manifest: dict | None = None) -> EquivariantDeLaN:
        if n_templates not in TEMPLATE_MAPS:
            raise ValueError("n_templates must be 1 or 2")
        leg_map = TEMPLATE_MAPS[n_templates]
        keys = sorted({k for k, _ in leg_map.values()})
        templates = {k: LegDeLaN(3, hidden, depth, eps, damping, frictionloss).to(device) for k in keys}
        meta = {"equivariant": True, "n_templates": n_templates, "hidden": hidden, "depth": depth, "eps": eps,
                "damping": damping, "frictionloss": frictionloss, "leg_map": {l: list(v) for l, v in leg_map.items()}}
        return EquivariantDeLaN(templates, dict(leg_map), mirror_maps(manifest), meta)

    def save(self, path: Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        for k, net in self.templates.items():
            torch.save(net.state_dict(), path / f"template_{k}.pt")
        (path / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @staticmethod
    def load(path: Path, device: str = "cpu") -> EquivariantDeLaN:
        path = Path(path); meta = json.loads((path / "meta.json").read_text())
        leg_map = {l: (v[0], bool(v[1])) for l, v in meta["leg_map"].items()}
        keys = sorted({k for k, _ in leg_map.values()}); templates = {}
        for k in keys:
            net = LegDeLaN(3, meta["hidden"], meta["depth"], meta["eps"], meta["damping"], meta["frictionloss"]).to(device)
            net.load_state_dict(torch.load(path / f"template_{k}.pt", map_location=device)); net.eval(); templates[k] = net
        return EquivariantDeLaN(templates, leg_map, mirror_maps(), meta)

    @property
    def nets(self) -> dict:
        """Per-leg view (template objects; mirrored legs share the object) — for code that iterates .nets.items()."""
        return {leg: self.templates[self.leg_map[leg][0]] for leg in LEG_ORDER}

    def parameters(self):
        for net in self.templates.values():
            yield from net.parameters()

    # -- the mirror map --------------------------------------------------------------------------------
    def to_template(self, leg: str, q, dq, ddq, a):
        """Inputs of `leg` expressed in the template's leg: (Sq, Sdq, Sddq, Ea) if the leg is the mirrored one."""
        _, mirrored = self.leg_map[leg]
        if not mirrored:
            return q, dq, ddq, a
        S = _to_dev(self.maps["S"][leg], q); E = _to_dev(self.maps["E"], a)
        return q * S, dq * S, ddq * S, a * E

    def from_template(self, leg: str, tau):
        _, mirrored = self.leg_map[leg]
        if not mirrored:
            return tau
        return tau * _to_dev(self.maps["S"][leg], tau)          # rho_Y(sigma)^{-1} = S (involution)

    def forward_leg(self, leg: str, q, dq, ddq, a):
        """torch: tau_hat (B, 3) of `leg` through its template."""
        key, _ = self.leg_map[leg]
        tq, tdq, tddq, ta = self.to_template(leg, q, dq, ddq, a)
        return self.from_template(leg, self.templates[key](tq, tdq, tddq, ta))

    def mass_matrix(self, leg: str, q):
        """torch: M_leg(q) = S M_0(S q) S for a mirrored leg (congruence with an orthogonal S -> SPD)."""
        key, mirrored = self.leg_map[leg]
        if not mirrored:
            return self.templates[key].mass_matrix(q)
        S = _to_dev(self.maps["S"][leg], q)
        M0 = self.templates[key].mass_matrix(q * S)
        return S[None, :, None] * M0 * S[None, None, :]

    def predict(self, leg: str, q, dq, ddq, a, batch: int = 8192) -> np.ndarray:
        """numpy in / numpy out: predicted joint torques (T, 3) for one leg (same API as DeLaNQuadruped.predict)."""
        key, _ = self.leg_map[leg]; net = self.templates[key]
        p = next(net.parameters()); dev, dt = p.device, p.dtype; out = []
        for i in range(0, len(q), batch):
            tq = torch.as_tensor(q[i:i + batch], dtype=dt, device=dev); tdq = torch.as_tensor(dq[i:i + batch], dtype=dt, device=dev)
            tddq = torch.as_tensor(ddq[i:i + batch], dtype=dt, device=dev); ta = torch.as_tensor(a[i:i + batch], dtype=dt, device=dev)
            with torch.enable_grad():
                tau = self.forward_leg(leg, tq, tdq, tddq, ta)
            out.append(tau.detach().cpu().numpy())
        return np.concatenate(out)


def load_delan(path: Path, device: str = "cpu"):
    """Load a plain DeLaNQuadruped or an EquivariantDeLaN depending on meta.json."""
    meta = json.loads((Path(path) / "meta.json").read_text())
    if meta.get("equivariant", False):
        return EquivariantDeLaN.load(path, device=device)
    return DeLaNQuadruped.load(path, device=device)


# ------------------------------------------------------------------------------------ equivariance defect
def _leg_pairs(maps: dict) -> list:
    seen, pairs = set(), []
    for leg in LEG_ORDER:
        p = maps["partner"][leg]
        if leg not in seen and p not in seen:
            pairs.append((leg, p)); seen.update({leg, p})
    return pairs


def equivariance_defect(model, q, dq, ddq, a, pairs=None, maps: dict | None = None, batch: int = 8192) -> dict:
    """Per-sample equivariance defect of any per-leg model with a predict(leg, q, dq, ddq, a) API:
        delta_f(z) = || f_partner(rho z) - S f_leg(z) ||_2,   rho z = (S q, S dq, S ddq, E a),
    evaluated on the given samples (q, dq, ddq, a) taken as inputs of `leg`. Returns {pair: (n,) array} plus the pooled
    quantiles ('q95', 'q50', 'max' over all pairs) — the delta_f^{(0.95)} of Part 2 when the samples are nominal
    validation data; the sup version is not computable, the quantile version is what is reported."""
    maps = mirror_maps() if maps is None else maps
    pairs = _leg_pairs(maps) if pairs is None else pairs
    out = {}
    allv = []
    for leg, partner in pairs:
        S = maps["S"][leg]; E = maps["E"]
        f_leg = model.predict(leg, q, dq, ddq, a, batch=batch)
        f_par = model.predict(partner, q * S, dq * S, ddq * S, a * E, batch=batch)
        d = np.linalg.norm(f_par - S * f_leg, axis=1)
        out[f"{leg}-{partner}"] = d; allv.append(d)
    allv = np.concatenate(allv)
    out["q95"] = float(np.quantile(allv, 0.95)); out["q50"] = float(np.quantile(allv, 0.5)); out["max"] = float(allv.max())
    out["rms"] = float(np.sqrt(np.mean(allv ** 2)))
    return out


# ------------------------------------------------------------------------------------ training
def _mirror_leg_data(d: dict, S: np.ndarray, E: np.ndarray) -> dict:
    """Map a mirrored leg's arrays into template coordinates: (Sq, Sdq, Sddq, Ea, Sy)."""
    return {"q": d["q"] * S, "dq": d["dq"] * S, "ddq": d["ddq"] * S, "a": d["a"] * E, "y": d["y"] * S}


def template_datasets(quad: EquivariantDeLaN, data_by_leg: dict) -> dict:
    """{template_key: {"train": {...}, "val": {...}}} — the legs sharing a template are pooled after mapping the
    mirrored ones through rho (weight sharing == training the template on the pair's mirrored union)."""
    out = {}
    for leg in LEG_ORDER:
        if leg not in data_by_leg:
            continue
        key, mirrored = quad.leg_map[leg]
        S = quad.maps["S"][leg]; E = quad.maps["E"]
        parts = out.setdefault(key, {"train": [], "val": []})
        for split in ("train", "val"):
            d = data_by_leg[leg][split]
            parts[split].append(_mirror_leg_data(d, S, E) if mirrored else d)
    for key, parts in out.items():
        for split in ("train", "val"):
            parts[split] = {k: np.concatenate([p[k] for p in parts[split]]).astype(np.float32) for k in ("q", "dq", "ddq", "a", "y")}
    return out


def train_equivariant(quad: EquivariantDeLaN, data_by_leg: dict, epochs: int = 40, batch: int = 8192, lr: float = 2e-3,
                      device: str = "cpu", weight_decay: float = 0.0, seed: int = 0, log=print, n_bins: int = 6,
                      quantile: float = 0.95) -> dict:
    """Train every template on its pooled (mirrored-union) data; report per-template histories and PER-LEG validation
    RMSE / beta_hat on each leg's own validation arrays (comparable with the plain per-leg reports)."""
    tds = template_datasets(quad, data_by_leg)
    report = {"templates": {}, "legs": {}}
    for key, d in tds.items():
        log(f"  [equiv] template {key}: train {len(d['train']['q'])} val {len(d['val']['q'])}")
        hist, res = train_leg(quad.templates[key], d, epochs=epochs, batch=batch, lr=lr, device=device,
                              weight_decay=weight_decay, seed=seed, log=log)
        report["templates"][key] = {"history": hist, "final_val_mse": hist[-1]["val_mse"],
                                    "final_val_rmse_per_joint": hist[-1]["val_rmse_per_joint"], "n_train": int(len(d["train"]["q"]))}
    for leg in LEG_ORDER:
        if leg not in data_by_leg:
            continue
        v = data_by_leg[leg]["val"]
        pred = quad.predict(leg, v["q"], v["dq"], v["ddq"], v["a"])
        res = pred - v["y"]
        report["legs"][leg] = {"final_val_rmse_per_joint": np.sqrt((res ** 2).mean(0)).tolist(), "final_val_mse": float((res ** 2).mean()),
                               "beta_hat": beta_hat(v["q"], res, n_bins=n_bins, quantile=quantile), "n_val": int(len(v["q"]))}
    return report
