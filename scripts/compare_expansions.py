from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from mechint.sae import SparseAutoEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare decoder features across checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--activation-dim", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser


def load_decoder(path: str, activation_dim: int) -> torch.Tensor:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    expansion = state_dict["W_dec"].shape[0] // activation_dim
    model = SparseAutoEncoder(d=activation_dim, m=activation_dim * expansion)
    model.load_state_dict(state_dict)
    return model.W_dec.detach().cpu()


def compare_pair(left_path: str, right_path: str, activation_dim: int) -> list[dict]:
    left = load_decoder(left_path, activation_dim)
    right = load_decoder(right_path, activation_dim)
    sim = F.normalize(left, dim=1) @ F.normalize(right, dim=1).T
    values, indices = sim.max(dim=1)
    rows = []
    for left_idx, (value, right_idx) in enumerate(zip(values.tolist(), indices.tolist())):
        rows.append(
            {
                "left_checkpoint": left_path,
                "right_checkpoint": right_path,
                "left_feature": left_idx,
                "right_feature": right_idx,
                "cosine_similarity": value,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = []
    for i in range(len(args.checkpoints) - 1):
        rows.extend(compare_pair(args.checkpoints[i], args.checkpoints[i + 1], args.activation_dim))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["left_checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"comparison_output: {output_path}")


if __name__ == "__main__":
    main()
