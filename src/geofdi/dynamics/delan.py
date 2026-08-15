"""Deep Lagrangian Networks (Lutter, Ritter & Peters, ICLR 2019) as the LEARNED nominal model of a Go2 leg.

Structure (per leg, n = 3 joints):  L(q) lower-triangular with softplus(+eps) diagonal  ->  M(q) = L L^T + eps I  (PD),
V(q; a) potential,  g = dV/dq,  C(q, dq) dq from the Christoffel form (dM/dq via autograd),
tau_hat = M(q) ddq + C(q, dq) dq + g(q; a) + b dq + f sign(dq)  (passive terms are FIXED nominal constants, not learned).

Moving base. A leg on a floating trunk obeys the fixed-base leg Lagrangian in the field of the trunk's SPECIFIC FORCE
(d'Alembert): the potential is V(q; a) = -sum_i m_i a^T p_i(q) with a the specific force in the base frame, which is
exactly what the trunk IMU measures. So the network's potential takes (q, a_imu) — a is an input parameter of V, not
a differentiated coordinate. Effects of the trunk's angular velocity/acceleration on the leg (Coriolis / Euler
terms) are NOT representable and end up in the model-error budget beta_hat (measured on the validation set).
The learned residual is  r = tau_cmd + J^T f_c - tau_hat  with the known contact wrench (as in the analytic observer;
M ddq + h = tau + J^T f).

TRAINING DATA: NOMINAL rollouts only (go2_urdf_sym, several speeds / terrains, no faults, no nuisances). The model
never sees fault data — the R+ channel it feeds stays a nominal-model residual; only the model is learned.

Weights live under $GEOFDI_DATA_ROOT/models/delan/<tag>/ (repo: config + this code); training runs on the GPU when
available.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LEG_ORDER = ("LF", "RF", "LH", "RH")


class LegDeLaN(nn.Module):
    """One leg: q (3), dq (3), ddq (3), a (3, specific force in the base frame) -> tau_hat (3)."""

    def __init__(self, n: int = 3, hidden: int = 128, depth: int = 3, eps: float = 1e-3, damping: float = 0.01,
                 frictionloss: float = 0.2, learn_passive: bool = False):
        super().__init__()
        self.n = n; self.eps = eps
        act = nn.Softplus
        def mlp(nin, nout):
            layers = []; d = nin
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), act()]; d = hidden
            layers.append(nn.Linear(d, nout)); return nn.Sequential(*layers)
        self.n_tril = n * (n + 1) // 2
        self.net_L = mlp(n, self.n_tril)                 # off-diagonal + raw diagonal of L(q)
        self.net_V = mlp(n + 3, 1)                       # V(q; a)
        self.register_buffer("b", torch.full((n,), float(damping)))
        self.register_buffer("f", torch.full((n,), float(frictionloss)))
        self.learn_passive = learn_passive
        if learn_passive:
            self.b = nn.Parameter(torch.full((n,), float(damping))); self.f = nn.Parameter(torch.full((n,), float(frictionloss)))
        idx = torch.tril_indices(n, n); self.register_buffer("tril_i", idx[0]); self.register_buffer("tril_j", idx[1])
        self.register_buffer("diag_mask", (idx[0] == idx[1]))
        # input normalisation (set from data)
        self.register_buffer("q_mu", torch.zeros(n)); self.register_buffer("q_sd", torch.ones(n))
        self.register_buffer("a_mu", torch.zeros(3)); self.register_buffer("a_sd", torch.ones(3))

    def mass_matrix(self, q):
        z = (q - self.q_mu) / self.q_sd
        l = self.net_L(z)
        l = torch.where(self.diag_mask, nn.functional.softplus(l) + self.eps, l)
        L = torch.zeros(q.shape[0], self.n, self.n, device=q.device, dtype=q.dtype)
        L[:, self.tril_i, self.tril_j] = l
        return L @ L.transpose(1, 2) + self.eps * torch.eye(self.n, device=q.device, dtype=q.dtype)

    def potential(self, q, a):
        z = torch.cat([(q - self.q_mu) / self.q_sd, (a - self.a_mu) / self.a_sd], dim=1)
        return self.net_V(z)[:, 0]

    def forward(self, q, dq, ddq, a):
        """tau_hat (B, n). Uses autograd for dM/dq and dV/dq (create_graph so that training can backprop through)."""
        q = q.requires_grad_(True)
        M = self.mass_matrix(q)                                    # (B, n, n)
        # Christoffel: C dq = 0.5 * (dM/dt dq)*2 - 0.5 * d/dq (dq^T M dq) ... use the standard form:
        # (C dq)_i = sum_jk Gamma_ijk dq_j dq_k, Gamma_ijk = 0.5 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        # -> C dq = dM/dt dq - 0.5 * grad_q (dq^T M dq),  with dM/dt = sum_k dM/dq_k dq_k
        Mdq = torch.einsum("bij,bj->bi", M, dq)                    # (B, n)
        # dM/dt dq: derivative of (M dq) w.r.t. q along dq, holding dq fixed
        Mdq_dot = torch.zeros_like(Mdq)
        for i in range(self.n):
            gi = torch.autograd.grad(Mdq[:, i].sum(), q, create_graph=True)[0]      # d(M dq)_i / dq  (B, n)
            Mdq_dot[:, i] = (gi * dq).sum(1)
        kin = 0.5 * (dq * Mdq).sum(1)
        dkin = torch.autograd.grad(kin.sum(), q, create_graph=True)[0]               # 0.5 grad_q (dq^T M dq)
        Cdq = Mdq_dot - dkin
        V = self.potential(q, a)
        g = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        tau = torch.einsum("bij,bj->bi", M, ddq) + Cdq + g + self.b * dq + self.f * torch.tanh(dq / 0.05)
        return tau


class DeLaNQuadruped:
    """Four LegDeLaN nets (one per leg, in LEG_ORDER) with a common (de)serialisation and a numpy residual API."""

    def __init__(self, nets: dict, meta: dict):
        self.nets = nets; self.meta = meta

    @staticmethod
    def build(hidden=128, depth=3, eps=1e-3, damping=0.01, frictionloss=0.2, device="cpu"):
        nets = {leg: LegDeLaN(3, hidden, depth, eps, damping, frictionloss).to(device) for leg in LEG_ORDER}
        return DeLaNQuadruped(nets, {"hidden": hidden, "depth": depth, "eps": eps, "damping": damping, "frictionloss": frictionloss})

    def save(self, path: Path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        for leg, net in self.nets.items():
            torch.save(net.state_dict(), path / f"{leg}.pt")
        (path / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @staticmethod
    def load(path: Path, device="cpu"):
        path = Path(path); meta = json.loads((path / "meta.json").read_text())
        nets = {}
        for leg in LEG_ORDER:
            net = LegDeLaN(3, meta["hidden"], meta["depth"], meta["eps"], meta["damping"], meta["frictionloss"]).to(device)
            net.load_state_dict(torch.load(path / f"{leg}.pt", map_location=device)); net.eval(); nets[leg] = net
        return DeLaNQuadruped(nets, meta)

    def predict(self, leg: str, q, dq, ddq, a, batch: int = 8192) -> np.ndarray:
        """numpy in / numpy out: predicted joint torques (T, 3) for one leg."""
        net = self.nets[leg]; p = next(net.parameters()); dev, dt = p.device, p.dtype; out = []
        for i in range(0, len(q), batch):
            tq = torch.as_tensor(q[i:i + batch], dtype=dt, device=dev); tdq = torch.as_tensor(dq[i:i + batch], dtype=dt, device=dev)
            tddq = torch.as_tensor(ddq[i:i + batch], dtype=dt, device=dev); ta = torch.as_tensor(a[i:i + batch], dtype=dt, device=dev)
            with torch.enable_grad():
                tau = net(tq, tdq, tddq, ta)
            out.append(tau.detach().cpu().numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------------------------ data + training
def leg_arrays(df, leg: str, dt: float = 0.005, sg_window: int = 7):
    """(q, dq, ddq, a_imu, tau_cmd, Jt f_c) for one leg from a telemetry frame. ddq by Savitzky-Golay derivative of the
    measured joint velocity (window sg_window, order 2). The contact term needs the kinematic Jacobian: it is provided by
    geofdi.dynamics.pin_model (leg-block of J^T f at the recorded contact point) — see contact_torque()."""
    from scipy.signal import savgol_filter
    q = df[[f"q_{leg}_{j}" for j in ("HAA", "HFE", "KFE")]].to_numpy(); dq = df[[f"dq_{leg}_{j}" for j in ("HAA", "HFE", "KFE")]].to_numpy()
    ddq = savgol_filter(dq, sg_window, 2, deriv=1, delta=dt, axis=0)
    a = df[["imu_a_x", "imu_a_y", "imu_a_z"]].to_numpy()
    tau = df[[f"tau_cmd_{leg}_{j}" for j in ("HAA", "HFE", "KFE")]].to_numpy()
    return q, dq, ddq, a, tau


def contact_torque(df, dyn, leg_index: int) -> np.ndarray:
    """(T, 3) joint torques of one leg due to the recorded contact wrench: J_p^T f + J_w^T tau at the recorded point."""
    from .momentum_observer import MomentumObserver
    from ..sim.telemetry import LEGS
    obs = MomentumObserver(dyn, 0.005, 10.0)
    L = LEGS[leg_index]
    q = df[[f"q_{l}_{j}" for l in LEGS for j in ("HAA", "HFE", "KFE")]].to_numpy()
    pos = df[["base_x", "base_y", "base_z"]].to_numpy(); quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy()
    fc = df[[f"fc_{a}_{L}" for a in "xyz"]].to_numpy(); cp = df[[f"cp_{a}_{L}" for a in "xyz"]].to_numpy(); tc = df[[f"tc_{a}_{L}" for a in "xyz"]].to_numpy()
    out = np.zeros((len(df), 3))
    for k in range(len(df)):
        if np.linalg.norm(fc[k]) == 0:
            continue
        qpos = np.concatenate([pos[k], quat[k], q[k]])
        pts = np.zeros((4, 3)); pts[leg_index] = cp[k]
        J, Jw = obs._point_jacobians(qpos, pts)
        tg = J[leg_index].T @ fc[k] + Jw[leg_index].T @ tc[k]
        out[k] = tg[6 + 3 * leg_index: 9 + 3 * leg_index]
    return out


def contact_torques_all(df, dyn) -> np.ndarray:
    """(T, 12) joint torques of all four legs due to the recorded contact wrenches (one Jacobian pass per row)."""
    from .momentum_observer import MomentumObserver
    from ..sim.telemetry import LEGS
    obs = MomentumObserver(dyn, 0.005, 10.0)
    q = df[[f"q_{l}_{j}" for l in LEGS for j in ("HAA", "HFE", "KFE")]].to_numpy()
    pos = df[["base_x", "base_y", "base_z"]].to_numpy(); quat = df[["base_qw", "base_qx", "base_qy", "base_qz"]].to_numpy()
    fc = np.stack([df[[f"fc_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1)
    cp = np.stack([df[[f"cp_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1)
    tc = np.stack([df[[f"tc_{a}_{l}" for a in "xyz"]].to_numpy() for l in LEGS], axis=1)
    out = np.zeros((len(df), 12)); act = np.linalg.norm(fc, axis=2) > 0
    for k in range(len(df)):
        if not act[k].any():
            continue
        qpos = np.concatenate([pos[k], quat[k], q[k]])
        J, Jw = obs._point_jacobians(qpos, cp[k])
        tg = np.zeros(18)
        for i in range(4):
            if act[k, i]:
                tg += J[i].T @ fc[k, i] + Jw[i].T @ tc[k, i]
        out[k] = tg[6:]
    return out


def delan_residuals(df, quad: "DeLaNQuadruped", jt_all: np.ndarray, dt: float = 0.005) -> np.ndarray:
    """(T, 12) learned-model residual r = tau_cmd + J^T f_c - DeLaN(q, dq, ddq, a) for all legs."""
    from ..sim.telemetry import LEGS
    out = np.zeros((len(df), 12))
    for li, leg in enumerate(LEGS):
        q, dq, ddq, a, tau = leg_arrays(df, leg, dt=dt)
        out[:, 3 * li:3 * li + 3] = tau + jt_all[:, 3 * li:3 * li + 3] - quad.predict(leg, q, dq, ddq, a)
    return out


def train_leg(net: LegDeLaN, data: dict, epochs: int = 30, batch: int = 4096, lr: float = 1e-3, device="cpu",
              weight_decay: float = 0.0, log=print, seed: int = 0):
    """data: dict of numpy arrays q, dq, ddq, a, y (target torque = tau_cmd + J^T f_c, the leg-inertial part) split into train/val.
    Returns history (list of dicts) and the validation residual array."""
    torch.manual_seed(seed)
    net.to(device)
    tr = {k: torch.as_tensor(data["train"][k], dtype=torch.float32, device=device) for k in ("q", "dq", "ddq", "a", "y")}
    va = {k: torch.as_tensor(data["val"][k], dtype=torch.float32, device=device) for k in ("q", "dq", "ddq", "a", "y")}
    net.q_mu.copy_(tr["q"].mean(0)); net.q_sd.copy_(tr["q"].std(0) + 1e-6); net.a_mu.copy_(tr["a"].mean(0)); net.a_sd.copy_(tr["a"].std(0) + 1e-6)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = tr["q"].shape[0]; hist = []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=device); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            tau = net(tr["q"][idx], tr["dq"][idx], tr["ddq"][idx], tr["a"][idx])
            loss = ((tau - tr["y"][idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        sched.step()
        net.eval(); res = []
        for i in range(0, va["q"].shape[0], 16384):
            tau = net(va["q"][i:i + 16384], va["dq"][i:i + 16384], va["ddq"][i:i + 16384], va["a"][i:i + 16384])
            res.append((tau - va["y"][i:i + 16384]).detach())
        res = torch.cat(res); vl = float((res ** 2).mean())
        hist.append({"epoch": ep, "train_mse": tot / n, "val_mse": vl, "val_rmse_per_joint": res.pow(2).mean(0).sqrt().cpu().tolist()})
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"    epoch {ep:3d} train {tot / n:.4f} val {vl:.4f} rmse/joint {np.round(hist[-1]['val_rmse_per_joint'], 3)}")
    return hist, res.cpu().numpy()


def beta_hat(q: np.ndarray, res: np.ndarray, n_bins: int = 6, quantile: float = 0.95) -> dict:
    """Empirical model-error bound: per-configuration-bin (HFE x KFE grid) quantile of ||residual||_2 (validation set)."""
    e = np.linalg.norm(res, axis=1)
    b1 = np.quantile(q[:, 1], np.linspace(0, 1, n_bins + 1)); b2 = np.quantile(q[:, 2], np.linspace(0, 1, n_bins + 1))
    i1 = np.clip(np.searchsorted(b1, q[:, 1], side="right") - 1, 0, n_bins - 1); i2 = np.clip(np.searchsorted(b2, q[:, 2], side="right") - 1, 0, n_bins - 1)
    grid = np.full((n_bins, n_bins), np.nan)
    for a in range(n_bins):
        for b in range(n_bins):
            m = (i1 == a) & (i2 == b)
            if m.sum() > 20:
                grid[a, b] = np.quantile(e[m], quantile)
    return {"global_q": float(np.quantile(e, quantile)), "global_rms": float(np.sqrt((e ** 2).mean())), "grid": grid.tolist(),
            "edges_hfe": b1.tolist(), "edges_kfe": b2.tolist(), "quantile": quantile}
