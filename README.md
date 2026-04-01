# Mechanistic Interpretability: Sparse Autoencoder Pipeline

## Current Workflow

This repository trains and analyzes sparse autoencoders (SAEs) on **Pythia-70M, layer 3 residual stream** activations. The default research path is now script-first:

- Shared code lives in `mechint/`
- Reproducible entrypoints live in `scripts/`
- New experiment artifacts live under `runs/`
- Shared activation caches stay in `activations/`
- Legacy checkpoints in `saved_models/` and CSVs in `training_metrics/` remain readable but are no longer the primary output layout

The canonical SAE implementation uses **separate encoder and decoder weights** with decoder-row renormalization after each optimizer step.

## Default Pipeline

### 1. Collect activations

```bash
python scripts/collect_activations.py \
  --model pythia-70m \
  --layer 3 \
  --hook blocks.3.hook_resid_post \
  --dataset openwebtext \
  --num-texts 50000 \
  --output activations/activations_50000.pt
```

### 2. Train an SAE

```bash
python scripts/train_sae.py \
  --activations-path activations/activations_50000.pt \
  --activation-dim 512 \
  --expansion 8 \
  --lam 3 \
  --epochs 50000
```

This creates a run directory under `runs/` with:

- `config.json`
- `metrics.csv`
- `checkpoint.pt`
- `summary.json`
- `manifest.json`

### 3. Evaluate a run

```bash
python scripts/eval_sae.py --run-dir runs/<run-name>
```

### 4. Rank features by causal effect

```bash
python scripts/batch_ablate.py \
  --checkpoint runs/<run-name>/checkpoint.pt \
  --config runs/<run-name>/config.json \
  --eval-texts-path eval_texts.txt \
  --output runs/<run-name>/ablation.csv
```

### 5. Compare features across checkpoints

```bash
python scripts/compare_expansions.py \
  --checkpoints saved_models/sae_model_4x.pt saved_models/sae_model_8x.pt \
  --activation-dim 512 \
  --output runs/comparisons/4x_vs_8x.csv
```

## Manual Research Utilities

The notebooks remain for exploration, but core logic should move through the package and scripts:

- `inspect-features.ipynb`: canonical inspection notebook
- `ablation.ipynb`: canonical ablation notebook
- `notebooks/archive/`: archived scratch and duplicate notebooks

Key reusable functions now live in:

- `mechint.analysis`
- `mechint.ablation`
- `mechint.eval`
- `mechint.data`

## Backward Compatibility

- `python train.py ...` still works and now forwards to `scripts/train_sae.py`
- `from sae import SparseAutoEncoder` still works and now re-exports the canonical implementation from `mechint.sae`
- Existing legacy checkpoints in `saved_models/` can be evaluated by the new scripts

## Validation

Run:

```bash
python check_paths.py
python -m unittest tests.test_pipeline
```

## Notes

- The default training path uses a deterministic held-out validation split from the cached activation tensor.
- Training streams random batches from CPU-backed activations instead of moving the full cache onto the accelerator.
- The repo also contains an `autoresearch/` subtree, but the default manual research pipeline is intentionally separate and should be stabilized first.
