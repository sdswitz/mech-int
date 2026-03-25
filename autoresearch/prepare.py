"""
Immutable evaluation harness for SAE autoresearch.

DO NOT MODIFY THIS FILE. It defines the fixed evaluation contract,
data loading, and metric computation. The autonomous agent modifies
only train.py.

SAE interface contract — the model passed to evaluate_sae() must have:
  encoder(x) -> f     where x is (batch, ACTIVATION_DIM), f is (batch, num_features)
  decoder(f) -> xhat  where xhat is (batch, ACTIVATION_DIM)
  forward(x) -> (xhat, f)
"""

import os
import math
import torch
import torch.nn.functional as F

# ── Fixed constants ──────────────────────────────────────────────────────────

ACTIVATION_DIM = 512  # Pythia-70M layer 3 residual stream dimension
ACTIVATIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "activations", "activations_50000.pt")
VAL_FRACTION = 0.05   # 5% held out for validation (~294K vectors)
VAL_SEED = 42          # deterministic split
TIME_BUDGET = 120      # 2 minutes wall clock for training
EVAL_BATCH_SIZE = 8192

# ── Device detection ─────────────────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Data loading ─────────────────────────────────────────────────────────────

def load_activations():
    """
    Load cached activations and split into train/val sets.

    Returns:
        train_acts: CPU tensor (mmap-backed), shape (N_train, 512)
        val_acts:   CPU tensor (mmap-backed), shape (N_val, 512)
        device:     string — "cuda", "mps", or "cpu"

    Memory strategy: tensors stay on CPU (mmap-backed, shared with OS page
    cache). Batches are moved to device on demand via make_dataloader().
    This avoids loading 12GB into MPS/CUDA memory.
    """
    path = os.path.abspath(ACTIVATIONS_PATH)
    print(f"Loading activations from {path}...", flush=True)
    all_acts = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    n = all_acts.shape[0]
    print(f"Loaded {n:,} activation vectors, dim={all_acts.shape[1]}, "
          f"dtype={all_acts.dtype}", flush=True)

    # Deterministic train/val split
    gen = torch.Generator().manual_seed(VAL_SEED)
    perm = torch.randperm(n, generator=gen)
    n_val = int(n * VAL_FRACTION)
    n_train = n - n_val

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Index into mmap tensor — creates views, not copies
    train_acts = all_acts[train_idx]
    val_acts = all_acts[val_idx]

    print(f"Split: {n_train:,} train, {n_val:,} val", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    return train_acts, val_acts, DEVICE


def make_dataloader(activations, batch_size, device):
    """
    Infinite iterator yielding random batches on device.

    Args:
        activations: CPU tensor, shape (N, ACTIVATION_DIM)
        batch_size:  int
        device:      string — target device for batches
    """
    n = activations.shape[0]
    while True:
        idx = torch.randint(n, (batch_size,))
        yield activations[idx].to(device)

# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sae(model, val_acts, feature_ever_active, num_features, device,
                 training_seconds=0.0, total_seconds=0.0, expansion=0):
    """
    Evaluate an SAE on the held-out validation set.

    Args:
        model:               SAE model (must satisfy interface contract)
        val_acts:            CPU tensor, shape (N_val, ACTIVATION_DIM)
        feature_ever_active: bool tensor, shape (num_features,) — tracked during training
        num_features:        int — total number of SAE features (m)
        device:              string
        training_seconds:    float — wall clock training time
        total_seconds:       float — wall clock total time (training + eval)
        expansion:           int — expansion factor used

    Returns:
        dict with keys: sae_score, val_mse, val_l0, dead_frac
    """
    model.eval()
    n = val_acts.shape[0]

    total_mse = 0.0
    total_l0 = 0.0
    n_batches = 0

    for start in range(0, n, EVAL_BATCH_SIZE):
        batch = val_acts[start:start + EVAL_BATCH_SIZE].to(device)
        xhat, f = model(batch)

        total_mse += F.mse_loss(xhat, batch).item()
        total_l0 += (f > 0).float().mean().item()
        n_batches += 1

    val_mse = total_mse / n_batches
    val_l0 = total_l0 / n_batches
    dead_frac = 1.0 - feature_ever_active.float().mean().item()

    # Composite metric: higher is better
    # -log10(MSE) rewards lower reconstruction error
    # -0.5*log10(L0) rewards sparser activations
    # -0.1*dead_frac penalizes dead features
    sae_score = -math.log10(max(val_mse, 1e-10)) - 0.5 * math.log10(max(val_l0, 1e-10)) - 0.1 * dead_frac

    num_params = sum(p.numel() for p in model.parameters())

    # Print grep-friendly summary
    print(f"\n---")
    print(f"sae_score:        {sae_score:.6f}")
    print(f"val_mse:          {val_mse:.6f}")
    print(f"val_l0:           {val_l0:.6f}")
    print(f"dead_frac:        {dead_frac:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_features:     {num_features}")
    print(f"expansion:        {expansion}")
    print(f"num_params:       {num_params / 1e6:.1f}M")

    model.train()
    return {
        "sae_score": sae_score,
        "val_mse": val_mse,
        "val_l0": val_l0,
        "dead_frac": dead_frac,
    }
