"""Validate the default research pipeline layout and flag legacy artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parent

required_files = [
    "train.py",
    "sae.py",
    "scripts/train_sae.py",
    "scripts/eval_sae.py",
    "scripts/batch_ablate.py",
    "scripts/compare_expansions.py",
    "mechint/sae.py",
    "mechint/data.py",
    "mechint/eval.py",
    "plans/default-pipeline-refactor.md",
]

required_dirs = [
    "activations",
    "runs",
    "scripts",
    "mechint",
    "tests",
]

legacy_files = [
    "train_wandb.py",
    "colab-training.ipynb",
    "training_metrics/training_metrics_3016.csv",
]

warnings: list[str] = []
errors: list[str] = []

for rel_path in required_files:
    if not (REPO_ROOT / rel_path).exists():
        errors.append(f"Missing required file: {rel_path}")

for rel_path in required_dirs:
    if not (REPO_ROOT / rel_path).exists():
        warnings.append(f"Expected directory '{rel_path}/' does not exist yet")

activation_path = REPO_ROOT / "activations" / "activations_50000.pt"
if activation_path.exists():
    print(f"  Found shared activations: {activation_path.relative_to(REPO_ROOT)}")
else:
    warnings.append("No default activation cache found at activations/activations_50000.pt")

legacy_models = sorted((REPO_ROOT / "saved_models").glob("sae_model_*x.pt")) if (REPO_ROOT / "saved_models").exists() else []
for model_path in legacy_models:
    print(f"  Found legacy checkpoint: {model_path.relative_to(REPO_ROOT)}")

legacy_metrics = sorted((REPO_ROOT / "training_metrics").glob("training_metrics_*x.csv")) if (REPO_ROOT / "training_metrics").exists() else []
for metrics_path in legacy_metrics:
    print(f"  Found legacy metrics: {metrics_path.relative_to(REPO_ROOT)}")

for rel_path in legacy_files:
    if (REPO_ROOT / rel_path).exists():
        warnings.append(f"Legacy or stale artifact present: {rel_path}")

archive_dir = REPO_ROOT / "notebooks" / "archive"
if archive_dir.exists():
    archived = sorted(p.name for p in archive_dir.glob("*.ipynb"))
    print(f"  Archived notebooks: {', '.join(archived)}" if archived else "  Archived notebooks directory is empty")
else:
    warnings.append("Expected notebooks/archive/ for duplicate notebooks is missing")

for notebook_name in ["inspect-features.ipynb", "ablation.ipynb"]:
    if not (REPO_ROOT / notebook_name).exists():
        warnings.append(f"Canonical notebook missing: {notebook_name}")

sample_run_configs = sorted((REPO_ROOT / "runs").glob("*/config.json")) if (REPO_ROOT / "runs").exists() else []
for config_path in sample_run_configs[:3]:
    try:
        json.loads(config_path.read_text())
    except json.JSONDecodeError:
        errors.append(f"Invalid JSON config: {config_path.relative_to(REPO_ROOT)}")

print()
if warnings:
    print(f"Warnings ({len(warnings)}):")
    for warning in warnings:
        print(f"  - {warning}")
    print()

if errors:
    print(f"Errors ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")
    print()
    sys.exit(1)

print("Default research pipeline layout looks good.")
