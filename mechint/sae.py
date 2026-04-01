from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def SAEloss(xhat: torch.Tensor, x: torch.Tensor, f: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    return F.mse_loss(xhat, x) + lam * f.abs().mean()


class SparseAutoEncoder(nn.Module):
    def __init__(self, d: int, m: int):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_enc = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(d, m)))
        self.W_dec = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(m, d)))
        self.renormalize_decoder()
        self.act = nn.ReLU()

    def renormalize_decoder(self) -> None:
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.b_pre
        x = x @ self.W_enc
        x = x + self.b_enc
        x = self.act(x)
        return x

    def decoder(self, f: torch.Tensor) -> torch.Tensor:
        xhat = f @ self.W_dec
        xhat = xhat + self.b_pre
        return xhat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x)
        xhat = self.decoder(f)
        return xhat, f
