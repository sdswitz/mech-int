"""
SAE training script for autoresearch.

This is the ONLY file the autonomous agent modifies. Everything is fair game:
model architecture, optimizer, hyperparameters, activation function, initialization,
batch size, expansion factor, etc.

Interface contract — the SAE model must have:
  encoder(x) -> f     where x is (batch, d), f is (batch, m)
  decoder(f) -> xhat  where xhat is (batch, d)
  forward(x) -> (xhat, f)
"""

import time
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import (
    ACTIVATION_DIM, TIME_BUDGET, DEVICE,
    load_activations, make_dataloader, evaluate_sae,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────

EXPANSION = 8
LEARNING_RATE = 3e-4
BATCH_SIZE = 4096
LAM_TARGET = 3.0
LAM_WARMUP_STEPS = 5000
WARMUP_STEPS = 2000
MAX_GRAD_NORM = 1.0

# ── SAE model ────────────────────────────────────────────────────────────────

d = ACTIVATION_DIM
m = d * EXPANSION


class SparseAutoEncoder(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_enc = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(d, m)))
        self.W_dec = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(m, d)))
        self.W_dec = nn.Parameter(self.W_dec / self.W_dec.norm(dim=1, keepdim=True))
        self.act = nn.ReLU()

    def encoder(self, x):
        x = x - self.b_pre
        x = x @ self.W_enc
        x = x + self.b_enc
        x = self.act(x)
        return x

    def decoder(self, f):
        xhat = f @ self.W_dec
        xhat = xhat + self.b_pre
        return xhat

    def forward(self, x):
        f = self.encoder(x)
        xhat = self.decoder(f)
        return xhat, f


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    total_start = time.time()

    # Load data
    train_acts, val_acts, device = load_activations()

    # Create model
    print(f"Creating SAE: d={d}, m={m}, expansion={EXPANSION}x", flush=True)
    model = SparseAutoEncoder(d=d, m=m).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M", flush=True)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # We don't know total steps ahead of time (time-budgeted), so we use
    # a lambda scheduler that takes progress as fraction of time budget
    feature_ever_active = torch.zeros(m, dtype=torch.bool, device=device)

    # Training loop — time-budgeted
    dataloader = make_dataloader(train_acts, BATCH_SIZE, device)
    step = 0
    train_start = time.time()

    print(f"Training for {TIME_BUDGET}s...", flush=True)
    while True:
        elapsed = time.time() - train_start
        if elapsed >= TIME_BUDGET:
            break

        x = next(dataloader)
        xhat, f = model(x)

        # Losses
        mse = F.mse_loss(xhat, x)
        l1 = f.abs().mean()
        l0 = (f > 0).float().mean()

        # Lambda warmup
        lam = LAM_TARGET * min(1.0, step / LAM_WARMUP_STEPS) if LAM_WARMUP_STEPS > 0 else LAM_TARGET
        loss = mse + lam * l1

        # Track active features
        feature_ever_active |= (f > 0).any(dim=0)
        dead_features = (~feature_ever_active).sum().item()

        # LR schedule: linear warmup then cosine decay
        progress = elapsed / TIME_BUDGET
        if step < WARMUP_STEPS:
            lr_mult = step / WARMUP_STEPS
        else:
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * lr_mult

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # Decoder norm constraint
        with torch.no_grad():
            model.W_dec.data = model.W_dec.data / model.W_dec.data.norm(dim=1, keepdim=True)

        # Logging every ~10% of time budget
        if step % max(1, int(TIME_BUDGET / 12)) == 0 or step < 3:
            print(f"step {step:5d} | {elapsed:5.1f}s | loss={loss:.4f} mse={mse:.4f} "
                  f"l0={l0:.4f} dead={dead_features} lr={LEARNING_RATE * lr_mult:.2e} "
                  f"lam={lam:.2f}", flush=True)

        step += 1

    training_seconds = time.time() - train_start
    print(f"\nTraining done: {step} steps in {training_seconds:.1f}s", flush=True)

    # Evaluate
    results = evaluate_sae(
        model=model,
        val_acts=val_acts,
        feature_ever_active=feature_ever_active,
        num_features=m,
        device=device,
        training_seconds=training_seconds,
        total_seconds=time.time() - total_start,
        expansion=EXPANSION,
    )

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/latest.pt")
    print(f"Model saved to saved_models/latest.pt")


if __name__ == "__main__":
    main()
