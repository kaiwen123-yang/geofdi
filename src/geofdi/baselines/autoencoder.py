"""Window autoencoder trained on NOMINAL windows only; score = reconstruction error of a window."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class WindowAE(nn.Module):
    def __init__(self, win: int, d: int, latent: int = 32, hidden: int = 256):
        super().__init__()
        self.win, self.d = win, d
        self.enc = nn.Sequential(nn.Linear(win * d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, win * d))

    def forward(self, x):                        # x (B, win, d)
        z = self.enc(x.reshape(x.shape[0], -1)); return self.dec(z).reshape(x.shape)


def train_ae(model: WindowAE, X: np.ndarray, epochs: int = 20, batch: int = 512, lr: float = 1e-3, device="cpu", seed: int = 0, log=None):
    torch.manual_seed(seed); model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device); n = len(Xt); hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; xb = Xt[idx]; loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        hist.append(tot / n)
        if log and (ep % 5 == 0 or ep == epochs - 1):
            log(f"    AE epoch {ep} loss {tot / n:.4f}")
    return hist


def ae_scores(model: WindowAE, X: np.ndarray, device="cpu", batch: int = 4096) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)
