from __future__ import annotations

from pathlib import Path
import csv
import json

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_feature_activity(
    model,
    activations: torch.Tensor,
    device: str,
    eval_batch_size: int = 8192,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    feature_active = torch.zeros(model.W_dec.shape[0], dtype=torch.bool, device=device)
    total = activations.shape[0] if indices is None else indices.shape[0]

    for start in range(0, total, eval_batch_size):
        if indices is None:
            batch = activations[start : start + eval_batch_size].to(device)
        else:
            batch = activations[indices[start : start + eval_batch_size]].to(device)
        features = model.encoder(batch)
        feature_active |= (features > 0).any(dim=0)

    model.train()
    return feature_active


def evaluate_sae(
    model,
    val_acts: torch.Tensor,
    feature_ever_active: torch.Tensor,
    num_features: int,
    device: str,
    eval_batch_size: int = 8192,
    training_seconds: float = 0.0,
    total_seconds: float = 0.0,
    run_metadata: dict | None = None,
    val_indices: torch.Tensor | None = None,
) -> dict:
    model.eval()
    n = val_acts.shape[0] if val_indices is None else val_indices.shape[0]
    total_mse = 0.0
    total_l0 = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, n, eval_batch_size):
            if val_indices is None:
                batch = val_acts[start : start + eval_batch_size].to(device)
            else:
                batch = val_acts[val_indices[start : start + eval_batch_size]].to(device)
            xhat, f = model(batch)
            total_mse += F.mse_loss(xhat, batch).item()
            total_l0 += (f > 0).float().mean().item()
            n_batches += 1

    dead_count = int((~feature_ever_active).sum().item())
    result = {
        "val_mse": total_mse / max(n_batches, 1),
        "val_l0": total_l0 / max(n_batches, 1),
        "dead_features": dead_count,
        "dead_fraction": dead_count / max(num_features, 1),
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "num_features": num_features,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    if run_metadata:
        result.update(run_metadata)

    model.train()
    return result


def print_eval_summary(summary: dict) -> None:
    ordered_keys = [
        "val_mse",
        "val_l0",
        "dead_features",
        "dead_fraction",
        "training_seconds",
        "total_seconds",
        "num_features",
        "num_params",
        "git_commit",
        "run_dir",
        "checkpoint_path",
    ]
    print("\n---")
    for key in ordered_keys:
        if key in summary:
            print(f"{key}: {summary[key]}")


def save_eval_summary(summary: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def append_metrics_row(path: str | Path, row: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return target
