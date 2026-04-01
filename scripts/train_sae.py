from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from mechint.config import SAETrainConfig
from mechint.data import (
    detect_device,
    get_git_commit,
    load_activations_with_splits,
    make_run_dir,
    set_global_seed,
    split_batch_iterator,
    write_run_manifest,
)
from mechint.eval import append_metrics_row, evaluate_sae, print_eval_summary, save_eval_summary
from mechint.sae import SAEloss, SparseAutoEncoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on cached activations.")
    parser.add_argument("--config", help="JSON config path")
    parser.add_argument("--model-name")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--hook-name")
    parser.add_argument("--dataset-name")
    parser.add_argument("--num-texts", type=int)
    parser.add_argument("--activation-dim", type=int)
    parser.add_argument("--activations-path")
    parser.add_argument("--expansion", type=int)
    parser.add_argument("--lam", type=float, dest="l1_lambda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lambda-warmup-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--runs-root")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--legacy-layout", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> SAETrainConfig:
    config = SAETrainConfig.from_json(args.config) if args.config else SAETrainConfig()
    override_keys = [
        "model_name",
        "layer",
        "hook_name",
        "dataset_name",
        "num_texts",
        "activation_dim",
        "activations_path",
        "expansion",
        "l1_lambda",
        "epochs",
        "learning_rate",
        "batch_size",
        "warmup_steps",
        "lambda_warmup_steps",
        "max_grad_norm",
        "val_fraction",
        "seed",
        "eval_batch_size",
        "runs_root",
        "run_name",
        "device",
    ]
    overrides = {key: getattr(args, key) for key in override_keys}
    config = config.with_overrides(**overrides)
    if args.legacy_layout:
        config = config.with_overrides(legacy_layout=True)
    return config


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    device = detect_device(config.device)
    config = config.with_overrides(device=device)
    set_global_seed(config.seed)

    run_dir = make_run_dir(config.runs_root, config.run_name) if not config.legacy_layout else None
    run_root = run_dir or Path(".")

    config_path = config.save_json(run_root / "config.json")
    print(f"Loading activations from {config.activations_path}", flush=True)
    all_acts, train_idx, val_idx = load_activations_with_splits(
        config.activations_path,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    d = config.activation_dim
    m = config.num_features
    model = SparseAutoEncoder(d=d, m=m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    feature_ever_active = torch.zeros(m, dtype=torch.bool, device=device)
    batch_iter = split_batch_iterator(all_acts, train_idx, config.batch_size, device, seed=config.seed)
    metrics_path = (
        Path("training_metrics") / f"training_metrics_{config.expansion}x.csv"
        if config.legacy_layout
        else run_root / "metrics.csv"
    )

    total_start = time.time()
    print(
        f"Training SAE: d={d}, m={m}, expansion={config.expansion}x, "
        f"epochs={config.epochs}, device={device}",
        flush=True,
    )
    for step in range(config.epochs):
        x = next(batch_iter)
        xhat, f = model(x)
        mse = F.mse_loss(xhat, x)
        l1 = f.abs().mean()
        l0 = (f > 0).float().mean()
        lam = (
            config.l1_lambda * min(1.0, step / config.lambda_warmup_steps)
            if config.lambda_warmup_steps > 0
            else config.l1_lambda
        )
        loss = SAEloss(xhat, x, f, lam=lam)

        feature_ever_active |= (f > 0).any(dim=0)

        if step < config.warmup_steps:
            lr_mult = step / max(config.warmup_steps, 1)
        else:
            progress = (step - config.warmup_steps) / max(config.epochs - config.warmup_steps, 1)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_mult

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        model.renormalize_decoder()

        if step % max(config.epochs // 10, 1) == 0 or step == config.epochs - 1:
            append_metrics_row(
                metrics_path,
                {
                    "step": step,
                    "loss": loss.item(),
                    "mse": mse.item(),
                    "l1": l1.item(),
                    "l0": l0.item(),
                    "dead_features": int((~feature_ever_active).sum().item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "lam": lam,
                },
            )
            print(
                f"step {step}: loss={loss.item():.4f} mse={mse.item():.4f} "
                f"l0={l0.item():.4f} dead={(~feature_ever_active).sum().item()} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} lam={lam:.2f}",
                flush=True,
            )

    training_seconds = time.time() - total_start
    summary = evaluate_sae(
        model=model,
        val_acts=all_acts,
        feature_ever_active=feature_ever_active,
        num_features=m,
        device=device,
        eval_batch_size=config.eval_batch_size,
        training_seconds=training_seconds,
        total_seconds=time.time() - total_start,
        val_indices=val_idx,
        run_metadata={
            "git_commit": get_git_commit(),
            "run_dir": str(run_dir) if run_dir else "legacy",
            "config_path": str(config_path),
        },
    )

    if config.legacy_layout:
        checkpoint_path = Path("saved_models") / f"sae_model_{config.expansion}x.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path = Path("training_metrics") / f"summary_{config.expansion}x.json"
    else:
        checkpoint_path = run_root / "checkpoint.pt"
        summary_path = run_root / "summary.json"

    torch.save(model.state_dict(), checkpoint_path)
    summary["checkpoint_path"] = str(checkpoint_path)
    save_eval_summary(summary, summary_path)
    print_eval_summary(summary)

    if run_dir:
        write_run_manifest(
            run_dir,
            {
                "git_commit": get_git_commit(),
                "config": config.to_dict(),
                "summary": summary,
                "checkpoint_path": str(checkpoint_path),
            },
        )


if __name__ == "__main__":
    main()
