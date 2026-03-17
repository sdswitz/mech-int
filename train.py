import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import argparse
import sys
import transformer_lens
import pandas as pd
from sae import SparseAutoEncoder, SAEloss
from datasets import load_dataset
import tqdm

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--expansion", type=int, default=8, help="Expansion factor (e.g. 4, 8, 16, 32)")
parser.add_argument("--lam", type=float, default=3.0, help="L1 penalty target")
parser.add_argument("--epochs", type=int, default=50000)
args = parser.parse_args()

EXPANSION = args.expansion
os.makedirs("saved_models", exist_ok=True)
MODEL_PATH = f"saved_models/sae_model_{EXPANSION}x.pt"

if os.path.exists(MODEL_PATH):
    print(f"Model already exists at {MODEL_PATH}, skipping {EXPANSION}x training.")
    exit(0)

NUM_TEXTS = 50000
ACTIVATIONS_PATH = f"activations/activations_{NUM_TEXTS}.pt"
os.makedirs("activations", exist_ok=True)

if os.path.exists(ACTIVATIONS_PATH):
    print(f"Loading cached activations from {ACTIVATIONS_PATH}...", flush=True)
    all_activations = torch.load(ACTIVATIONS_PATH)
    print(f"Activations loaded: shape={all_activations.shape}, dtype={all_activations.dtype}, "
          f"size={all_activations.nelement() * all_activations.element_size() / 1e9:.2f} GB", flush=True)
else:
    # Find the latest checkpoint to resume from
    start_text = 0
    all_activations = []
    for ckpt in sorted(
        [f for f in os.listdir(".") if f.startswith("activations_checkpoint_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    ):
        n = int(ckpt.split("_")[-1].split(".")[0])
        if n <= NUM_TEXTS:
            start_text = n
            latest_checkpoint = ckpt

    if start_text > 0:
        print(f"Resuming from checkpoint {latest_checkpoint} ({start_text}/{NUM_TEXTS} texts)")
        all_activations = [torch.load(latest_checkpoint)]

    # Load a model (eg GPT-2 Small)
    model = transformer_lens.HookedTransformer.from_pretrained("pythia-70m")

    # Load OpenWebText from HuggingFace
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    # Collect activations
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= NUM_TEXTS:
                break
            if i < start_text:
                continue
            text = example["text"][:512]  # truncate long texts to avoid OOM
            _, cache = model.run_with_cache(text, names_filter='blocks.3.hook_resid_post')
            acts = cache['blocks.3.hook_resid_post'][0].cpu()  # (seq_len, 512)
            all_activations.append(acts)
            del cache
            if (i + 1) % 5000 == 0:
                checkpoint = torch.cat(all_activations, dim=0)
                checkpoint_path = f"activations/activations_checkpoint_{i + 1}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path} ({i + 1}/{NUM_TEXTS} texts)")
                # Delete previous checkpoint
                prev_path = f"activations/activations_checkpoint_{i + 1 - 5000}.pt"
                if os.path.exists(prev_path):
                    os.remove(prev_path)

    all_activations = torch.cat(all_activations, dim=0)
    torch.save(all_activations, ACTIVATIONS_PATH)
    print(f"Saved activations to {ACTIVATIONS_PATH}")

print(f"Total activation vectors: {all_activations.shape[0]}")

## Hyperparameters
d = all_activations.shape[-1]
m = d * EXPANSION
learning_rate = 3e-4
batch_size = 4096
EPOCHS = args.epochs
WARMUP_STEPS = 2000
LAM_TARGET = args.lam
LAM_WARMUP_STEPS = 5000
MAX_GRAD_NORM = 1.0
interval = EPOCHS // 10

print(f"Training {EXPANSION}x SAE: d={d}, m={m}, lam={LAM_TARGET}, epochs={EPOCHS}", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

print(f"Moving activations to {device}...", flush=True)
all_activations = all_activations.to(device)
print(f"Activations on device.", flush=True)
if torch.cuda.is_available():
    print(f"GPU memory after activations: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)

print(f"Creating SAE (d={d}, m={m})...", flush=True)
SAE = SparseAutoEncoder(d=d, m=m)
SAE = SAE.to(device)
print(f"SAE on device. Parameters: {sum(p.numel() for p in SAE.parameters()) / 1e6:.1f}M", flush=True)
if torch.cuda.is_available():
    print(f"GPU memory after SAE: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

optimizer = torch.optim.Adam(SAE.parameters(), lr=learning_rate)
print(f"Optimizer created.", flush=True)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, EPOCHS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

feature_ever_active = torch.zeros(m, dtype=torch.bool, device=device)

os.makedirs("training_metrics", exist_ok=True)
METRICS_PATH = f"training_metrics/training_metrics_{EXPANSION}x.csv"
metrics_rows = []

print(f"Starting training loop...", flush=True)
for i in range(EPOCHS):
    if i == 0:
        print(f"Step 0: sampling batch...", flush=True)
    x = all_activations[torch.randperm(all_activations.shape[0])[:batch_size]].to(device)
    if i == 0:
        print(f"Step 0: forward pass...", flush=True)
    xhat, f = SAE(x)

    mse = F.mse_loss(xhat, x)
    l1 = f.abs().mean()
    l0 = (f > 0).float().mean()
    lam = LAM_TARGET * min(1.0, i / LAM_WARMUP_STEPS) if LAM_WARMUP_STEPS > 0 else LAM_TARGET
    loss = SAEloss(xhat, x, f, lam=lam)

    feature_ever_active |= (f > 0).any(dim=0)
    dead_features = (~feature_ever_active).sum().item()

    metrics_rows.append({
        "step": i,
        "loss": loss.item(),
        "mse": mse.item(),
        "l1": l1.item(),
        "l0": l0.item(),
        "dead_features": dead_features,
    })

    if i % interval == 0 or i == EPOCHS - 1:
        current_lr = scheduler.get_last_lr()[0]
        mem_str = f" gpu={torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else ""
        print(f"step {i}: loss={loss:.4f} mse={mse:.4f} l0={l0:.4f} dead={dead_features} lr={current_lr:.2e} lam={lam:.2f}{mem_str}", flush=True)
        pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(SAE.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        SAE.W_dec.data = SAE.W_dec.data / SAE.W_dec.data.norm(dim=1, keepdim=True)

# save the model after training
torch.save(SAE.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
