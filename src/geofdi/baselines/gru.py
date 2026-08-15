"""GRU baselines — the supervised, TRAINED-ON-FAULT-DATA method (its own protocol).

Two modes:
- classification (e07): windows of `win` steps of the standardized data element Z (d channels) -> P(fault); 2 x 64 GRU
  + FC, window 50 = 0.25 s at 200 Hz (placeholder architecture; the split by fault MAGNITUDE / TYPE is the point).
- `regression_eta` (Liu et al., RA-L 2025, GRUFD, Table I; Sprint 7 Block 0): input 57 = body angles (3) + body rates
  (3) + q (12) + q_des (12) + dq (12) + dq_des (12) + body command (3), hidden 256, output 12 = torque retention rates
  eta_hat (MSE loss), 100 epochs, batch 32, lr 1e-4, inference at 50 Hz, low-pass on eta_hat, joint faulty iff
  eta_hat_j < 0.7 (Algorithm 1). Number of GRU layers is not stated in the paper: 1 layer, marked "to verify".
  Sequence handling: the GRU runs over the window and the last hidden state is decoded; at test time the window slides
  by one sample (stateful streaming would be equivalent for a 1-layer GRU up to the warm-up).

    m = GRURegressor(57, hidden=256, layers=1, n_out=12)
    train_gru_regressor(m, seqs, etas, win, stride, epochs=100, batch=32, lr=1e-4)
    eta_hat = predict_eta(m, X)                                   # (n_windows, 12)
    faults = eta_lowpass_threshold(eta_hat, fc_hz=1.0, fs=50.0, thr=0.7)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LIU_TABLE_I = {"input": 57, "hidden": 256, "output": 12, "epochs": 100, "batch": 32, "lr": 1e-4, "layers": 1,
               "layers_verified": False, "inference_hz": 50, "threshold": 0.7, "loss": "mse"}


class GRUClassifier(nn.Module):
    def __init__(self, d: int, hidden: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                        # (B, win, d) -> logits (B,)
        h, _ = self.gru(x); return self.fc(h[:, -1])[:, 0]


class GRURegressor(nn.Module):
    """Liu-style GRUFD: (B, win, d) -> eta_hat (B, n_out); a linear head on the last hidden state (Table I: 57 -> 256
    -> 12); the output is not squashed (the targets are in [0.4, 1])."""

    def __init__(self, d: int = 57, hidden: int = 256, layers: int = 1, n_out: int = 12, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):
        h, _ = self.gru(x); return self.fc(h[:, -1])


class WindowSetReg:
    """Windows with vector targets (eta at the last step of the window)."""

    def __init__(self, seqs, targets, win, stride):
        self.seqs = seqs; self.targets = targets; self.win = win; self.index = []
        for si, s in enumerate(seqs):
            for a in range(0, len(s) - win + 1, stride):
                self.index.append((si, a))
        self.index = np.array(self.index, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def batch(self, idx):
        rows = self.index[idx]
        X = np.stack([self.seqs[si][a:a + self.win] for si, a in rows]).astype(np.float32)
        y = np.stack([self.targets[si][a + self.win - 1] for si, a in rows]).astype(np.float32)
        return X, y


def train_gru_regressor(model: GRURegressor, ws: WindowSetReg, epochs: int = 100, batch: int = 32, lr: float = 1e-4, device="cpu",
                        seed: int = 0, log=None, val: WindowSetReg | None = None, max_batches_per_epoch: int | None = None):
    """MSE regression of eta (Table I protocol). `max_batches_per_epoch` caps the work per epoch for large window sets
    (a uniform random subsample of windows per epoch); None = full pass."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss(); n = len(ws); hist = []
    for ep in range(epochs):
        model.train(); perm = rng.permutation(n); tot = 0.0; cnt = 0
        nb = (n + batch - 1) // batch if max_batches_per_epoch is None else min(max_batches_per_epoch, (n + batch - 1) // batch)
        for b in range(nb):
            X, y = ws.batch(perm[b * batch:(b + 1) * batch]); X = torch.as_tensor(X, device=device); y = torch.as_tensor(y, device=device)
            loss = lossf(model(X), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(y); cnt += len(y)
        rec = {"epoch": ep, "loss": tot / max(cnt, 1)}
        if val is not None:
            rec["val_mse"] = mse(model, val, device)
        hist.append(rec)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"    GRU-reg epoch {ep} loss {rec['loss']:.5f}" + (f" val MSE {rec['val_mse']:.5f}" if val is not None else ""))
    return hist


def predict_eta(model: GRURegressor, X: np.ndarray, device="cpu", batch: int = 4096) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(out)


def mse(model, ws: WindowSetReg, device="cpu", max_windows: int = 20000, seed: int = 0):
    idx = np.arange(len(ws))
    if len(idx) > max_windows:
        idx = np.random.default_rng(seed).choice(idx, size=max_windows, replace=False)
    X, y = ws.batch(idx); p = predict_eta(model, X, device)
    return float(np.mean((p - y) ** 2))


def eta_lowpass_threshold(eta_hat: np.ndarray, fc_hz: float = 1.0, fs: float = 50.0, thr: float = 0.7) -> np.ndarray:
    """Algorithm 1 deployment rule: first-order low-pass of eta_hat (cut-off fc_hz at rate fs) then joint faulty iff
    filtered eta_hat < thr. Returns boolean (T, n_out). The paper does not give the filter constant; fc = 1 Hz is our
    choice and is reported as such."""
    a = float(np.exp(-2 * np.pi * fc_hz / fs)); y = np.empty_like(eta_hat, dtype=float); s = np.ones(eta_hat.shape[1])
    for t in range(len(eta_hat)):
        s = a * s + (1 - a) * eta_hat[t]; y[t] = s
    return y < thr


class WindowSet:
    """Windows drawn on the fly from stored sequences (seq (T, d) float32) with per-window labels."""

    def __init__(self, seqs, labels, win, stride):
        self.seqs = seqs; self.win = win; self.index = []
        for si, (s, lab) in enumerate(zip(seqs, labels)):
            T = len(s)
            for a in range(0, T - win + 1, stride):
                self.index.append((si, a, float(lab[a + win - 1]) if np.ndim(lab) else float(lab)))
        self.index = np.array(self.index, dtype=object)

    def __len__(self):
        return len(self.index)

    def batch(self, idx):
        X = np.stack([self.seqs[si][a:a + self.win] for si, a, _ in self.index[idx]]); y = np.array([lab for _, _, lab in self.index[idx]], dtype=np.float32)
        return X, y


def train_gru(model: GRUClassifier, ws: WindowSet, epochs: int = 8, batch: int = 512, lr: float = 1e-3, device="cpu", seed: int = 0, log=None, val: WindowSet | None = None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(); n = len(ws); hist = []
    for ep in range(epochs):
        model.train(); perm = rng.permutation(n); tot = 0.0
        for i in range(0, n, batch):
            X, y = ws.batch(perm[i:i + batch]); X = torch.as_tensor(X, device=device); y = torch.as_tensor(y, device=device)
            loss = lossf(model(X), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(y)
        rec = {"epoch": ep, "loss": tot / n}
        if val is not None:
            rec["val_auc"] = auc(model, val, device)
        hist.append(rec)
        if log:
            log(f"    GRU epoch {ep} loss {tot / n:.4f}" + (f" val AUC {rec['val_auc']:.3f}" if val is not None else ""))
    return hist


def predict_windows(model: GRUClassifier, X: np.ndarray, device="cpu", batch: int = 4096) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device))).cpu().numpy())
    return np.concatenate(out)


def auc(model, ws: WindowSet, device="cpu"):
    from sklearn.metrics import roc_auc_score
    idx = np.arange(len(ws)); X, y = ws.batch(idx); p = predict_windows(model, X, device)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
