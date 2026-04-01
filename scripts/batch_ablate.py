from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mechint.ablation import build_transformer_model, load_sae_checkpoint, rank_features_by_ablation, save_ablation_rows
from mechint.config import SAETrainConfig
from mechint.data import detect_device, iter_texts_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank SAE features by mean KL under ablation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--eval-texts-path", required=True)
    parser.add_argument("--feature-subset", help="Comma-separated feature ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = SAETrainConfig.from_json(args.config) if args.config else SAETrainConfig()
    device = detect_device(args.device or config.device)
    sae = load_sae_checkpoint(
        args.checkpoint,
        activation_dim=config.activation_dim,
        expansion=None,
        device=device,
    )
    model = build_transformer_model(config.model_name)
    texts = list(iter_texts_from_path(args.eval_texts_path))
    subset = [int(item) for item in args.feature_subset.split(",")] if args.feature_subset else None
    rows = rank_features_by_ablation(sae, model, config.hook_name, texts, subset)
    output_path = save_ablation_rows(rows, args.output)
    print(f"ablation_output: {output_path}")
    if rows:
        print(f"top_feature: {rows[0]['feature_idx']} mean_kl={rows[0]['mean_kl']:.6f}")


if __name__ == "__main__":
    main()
