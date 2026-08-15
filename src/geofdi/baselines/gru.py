"""GRU fault classifier — the supervised baseline. TRAINED ON FAULT DATA (this is the method's protocol): windows of
`win` steps of the standardized data element Z (d channels) -> P(fault). Architecture (default 2 x 64 GRU + FC,
window 50 = 0.25 s at 200 Hz) is a config placeholder to be back-filled from Liu et al. (RA-L 2025) once the PDF is
available; the training/test split by fault MAGNITUDE (seen vs unseen) and by fault TYPE is the point of e07.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, d: int, hidden: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(d, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                        # (B, win, d) -> logits (B,)
        h, _ = self.gru(x); return self.fc(h[:, -1])[:, 0]


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
