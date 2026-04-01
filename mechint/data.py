from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable
import json
import os
import random
import subprocess

import torch
from datasets import load_dataset
import transformer_lens

from .config import SAETrainConfig


def detect_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_activation_tensor(path: str | Path, mmap: bool = True) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu", mmap=mmap, weights_only=True)


def deterministic_split_indices(n: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]


# NOTE: Removed — fancy indexing on mmap'd tensors forces a full copy (~24GB peak
# for a 12GB activation file), defeating the purpose of mmap. Use
# load_activations_with_splits() + split_batch_iterator() instead, which keep the
# mmap'd tensor intact and index only at batch time.
#
# def load_train_val_activations(
#     path: str | Path,
#     val_fraction: float,
#     seed: int,
#     mmap: bool = True,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     all_acts = load_activation_tensor(path, mmap=mmap)
#     train_idx, val_idx = deterministic_split_indices(all_acts.shape[0], val_fraction, seed)
#     return all_acts[train_idx], all_acts[val_idx]


def load_activations_with_splits(
    path: str | Path,
    val_fraction: float,
    seed: int,
    mmap: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_acts = load_activation_tensor(path, mmap=mmap)
    train_idx, val_idx = deterministic_split_indices(all_acts.shape[0], val_fraction, seed)
    return all_acts, train_idx, val_idx


def random_batch_iterator(
    activations: torch.Tensor,
    batch_size: int,
    device: str,
    seed: int,
) -> Generator[torch.Tensor, None, None]:
    gen = torch.Generator().manual_seed(seed)
    n = activations.shape[0]
    while True:
        idx = torch.randint(n, (batch_size,), generator=gen)
        yield activations[idx].to(device)


def split_batch_iterator(
    activations: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: str,
    seed: int,
) -> Generator[torch.Tensor, None, None]:
    gen = torch.Generator().manual_seed(seed)
    n = indices.shape[0]
    while True:
        positions = torch.randint(n, (batch_size,), generator=gen)
        batch_idx = indices[positions]
        yield activations[batch_idx].to(device)


def iter_activation_batches(
    activations: torch.Tensor,
    indices: torch.Tensor | None,
    batch_size: int,
    device: str,
) -> Generator[torch.Tensor, None, None]:
    if indices is None:
        total = activations.shape[0]
        for start in range(0, total, batch_size):
            yield activations[start : start + batch_size].to(device)
        return

    total = indices.shape[0]
    for start in range(0, total, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield activations[batch_idx].to(device)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_git_commit(default: str = "unknown") -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return default


def make_run_dir(runs_root: str | Path, run_name: str | None = None) -> Path:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or timestamp_slug()
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_activations(
    config: SAETrainConfig,
    output_path: str | Path | None = None,
    checkpoint_every: int = 5000,
) -> tuple[Path, Path]:
    set_global_seed(config.seed)
    output = Path(output_path or config.activations_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = transformer_lens.HookedTransformer.from_pretrained(config.model_name)
    dataset = load_dataset(config.dataset_name, split=config.dataset_split, streaming=True)

    all_activations: list[torch.Tensor] = []
    for i, example in enumerate(dataset):
        if i >= config.num_texts:
            break
        text = example["text"][: config.max_chars]
        _, cache = model.run_with_cache(text, names_filter=config.hook_name)
        acts = cache[config.hook_name][0].cpu()
        all_activations.append(acts)
        del cache
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            checkpoint_path = output.parent / f"{output.stem}_checkpoint_{i + 1}.pt"
            torch.save(torch.cat(all_activations, dim=0), checkpoint_path)

    tensor = torch.cat(all_activations, dim=0)
    torch.save(tensor, output)

    metadata = {
        "config": asdict(config),
        "activation_path": str(output),
        "num_vectors": int(tensor.shape[0]),
        "activation_dim": int(tensor.shape[1]),
        "git_commit": get_git_commit(),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output, metadata_path


def write_run_manifest(run_dir: str | Path, manifest: dict) -> Path:
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def iter_texts_from_path(path: str | Path) -> Iterable[str]:
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            yield line
