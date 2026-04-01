from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mechint.config import SAETrainConfig
from mechint.data import detect_device, get_git_commit, load_activations_with_splits
from mechint.eval import collect_feature_activity, evaluate_sae, print_eval_summary, save_eval_summary
from mechint.sae import SparseAutoEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAE checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--output")
    parser.add_argument("--device")
    return parser


def load_checkpoint_and_config(args: argparse.Namespace) -> tuple[Path, SAETrainConfig]:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        return run_dir / "checkpoint.pt", SAETrainConfig.from_json(run_dir / "config.json")
    if not args.checkpoint:
        raise SystemExit("Provide --checkpoint or --run-dir")
    checkpoint = Path(args.checkpoint)
    config = SAETrainConfig.from_json(args.config) if args.config else SAETrainConfig()
    return checkpoint, config


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint_path, config = load_checkpoint_and_config(args)
    device = detect_device(args.device or config.device)

    all_acts, train_idx, val_idx = load_activations_with_splits(
        config.activations_path,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "W_dec" in state_dict:
        expansion = state_dict["W_dec"].shape[0] // config.activation_dim
        config = config.with_overrides(expansion=expansion)

    model = SparseAutoEncoder(d=config.activation_dim, m=config.num_features).to(device)
    model.load_state_dict(state_dict)

    feature_ever_active = collect_feature_activity(
        model,
        activations=all_acts,
        indices=train_idx,
        device=device,
        eval_batch_size=config.eval_batch_size,
    )
    summary = evaluate_sae(
        model=model,
        val_acts=all_acts,
        feature_ever_active=feature_ever_active,
        num_features=config.num_features,
        device=device,
        eval_batch_size=config.eval_batch_size,
        val_indices=val_idx,
        run_metadata={
            "git_commit": get_git_commit(),
            "checkpoint_path": str(checkpoint_path),
        },
    )
    print_eval_summary(summary)
    if args.output:
        save_eval_summary(summary, args.output)


if __name__ == "__main__":
    main()
