"""Verify that all directories and expected files for the training pipeline exist."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)

errors = []
warnings = []

# --- Required source files ---
required_files = [
    "train.py",
    "sae.py",
]

for f in required_files:
    if not os.path.exists(f):
        errors.append(f"Missing required file: {f}")

# --- Directories that train.py creates (verify or note) ---
required_dirs = [
    "activations",
    "saved_models",
    "training_metrics",
]

for d in required_dirs:
    path = os.path.join(REPO_ROOT, d)
    if not os.path.isdir(path):
        warnings.append(f"Directory '{d}/' does not exist yet (will be created by train.py)")

# --- Check for cached activations ---
NUM_TEXTS = 50000
activations_path = f"activations/activations_{NUM_TEXTS}.pt"
if os.path.exists(activations_path):
    print(f"  Found cached activations: {activations_path}")
else:
    warnings.append(f"No cached activations at {activations_path} (will be generated on first run)")

# --- Check for trained models ---
expansions_found = []
for exp in [4, 8, 16, 32]:
    model_path = f"saved_models/sae_model_{exp}x.pt"
    metrics_path = f"training_metrics/training_metrics_{exp}x.csv"
    if os.path.exists(model_path):
        expansions_found.append(exp)
        print(f"  Found trained model: {model_path}")
        if not os.path.exists(metrics_path):
            warnings.append(f"Model {model_path} exists but metrics file {metrics_path} is missing")
    if os.path.exists(metrics_path):
        print(f"  Found metrics: {metrics_path}")

# --- Check for stale flat-structure files from old train_wandb.py / colab ---
stale_files = [
    "sae_model.pt",
    "training_metrics.csv",
]
for f in stale_files:
    if os.path.exists(f):
        warnings.append(f"Stale file from old layout: {f} (consider moving into saved_models/ or training_metrics/)")

# --- Check inspect-features.ipynb references ---
if os.path.exists("inspect-features.ipynb"):
    import json
    with open("inspect-features.ipynb") as nb:
        src = json.dumps(json.load(nb))
    # Check it's not pointing at the old un-suffixed model path
    if "sae_model.pt" in src and "sae_model_{EXPANSION}x" not in src:
        warnings.append("inspect-features.ipynb still references 'sae_model.pt' instead of 'sae_model_{EXPANSION}x.pt'")

# --- Report ---
print()
if warnings:
    print(f"Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠ {w}")
    print()

if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors:
        print(f"  ✗ {e}")
    print()
    sys.exit(1)
else:
    print("All paths look good.")
