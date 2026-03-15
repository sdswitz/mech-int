import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import transformer_lens
import pandas as pd
from sae import SparseAutoEncoder, SAEloss
from datasets import load_dataset
import tqdm
import wandb

MODEL_PATH = "sae_model.pt"
# if os.path.exists(MODEL_PATH):
#     print(f"Model already exists at {MODEL_PATH}, skipping training.")
#     exit(0)

NUM_TEXTS = 50000
ACTIVATIONS_PATH = f"activations/activations_{NUM_TEXTS}.pt"

if os.path.exists(ACTIVATIONS_PATH):
    print(f"Loading cached activations from {ACTIVATIONS_PATH}")
    all_activations = torch.load(ACTIVATIONS_PATH)
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
                checkpoint_path = f"activations_checkpoint_{i + 1}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path} ({i + 1}/{NUM_TEXTS} texts)")
                # Delete previous checkpoint
                prev_path = f"activations_checkpoint_{i + 1 - 5000}.pt"
                if os.path.exists(prev_path):
                    os.remove(prev_path)

    all_activations = torch.cat(all_activations, dim=0)
    torch.save(all_activations, ACTIVATIONS_PATH)
    print(f"Saved activations to {ACTIVATIONS_PATH}")

print(f"Total activation vectors: {all_activations.shape[0]}")

## Hyperparameters
d = all_activations.shape[-1]
m = d * 8
learning_rate = 3e-4
batch_size = 4096
EPOCHS = 20000
interval = EPOCHS // 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_activations = all_activations.to(device)

SAE = SparseAutoEncoder(d=d, m=m)
SAE = SAE.to(device)
optimizer = torch.optim.Adam(SAE.parameters(), lr=learning_rate)

wandb.init(project="sae-pythia-70m",
           mode = "online",
           name = f"sae_d{d}_m{m}_lr{learning_rate}_bs{batch_size}_epochs{EPOCHS}",
           config={
            "d": d,
            "m": m,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": EPOCHS,
            "num_texts": NUM_TEXTS,
        })

feature_ever_active = torch.zeros(m, dtype=torch.bool, device=device)

for i in range(EPOCHS):
    x = all_activations[torch.randperm(all_activations.shape[0])[:batch_size]].to(device)
    xhat, f = SAE(x)

    mse = F.mse_loss(xhat, x)
    l1 = f.abs().mean()
    l0 = (f > 0).float().mean()
    loss = SAEloss(xhat, x, f)

    feature_ever_active |= (f > 0).any(dim=0)
    dead_features = (~feature_ever_active).sum().item()

    wandb.log({
        "mse": mse.item(),
        "l1": l1.item(),
        "l0": l0.item(),
        "loss": loss.item(),
        "dead_features": dead_features,
    }, step=i)

    if i % interval == 0 or i == EPOCHS - 1:
        print(f"step {i}: loss={loss:.4f} mse={mse:.4f} l0={l0:.4f} dead={dead_features}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        SAE.W_dec.data = SAE.W_dec.data / SAE.W_dec.data.norm(dim=1, keepdim=True)
        

# save the model after training
torch.save(SAE.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

wandb.finish()