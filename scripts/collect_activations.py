from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mechint.config import SAETrainConfig
from mechint.data import collect_activations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect residual stream activations for SAE training.")
    parser.add_argument("--model", default="pythia-70m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--hook", default="blocks.3.hook_resid_post")
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-texts", type=int, default=50000)
    parser.add_argument("--output", default="activations/activations_50000.pt")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SAETrainConfig(
        model_name=args.model,
        model_source=f"EleutherAI/{args.model}",
        layer=args.layer,
        hook_name=args.hook,
        dataset_name=args.dataset,
        dataset_split=args.split,
        num_texts=args.num_texts,
        max_chars=args.max_chars,
        seed=args.seed,
        activations_path=args.output,
    )
    activation_path, metadata_path = collect_activations(
        config=config,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"activation_path: {activation_path}")
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
