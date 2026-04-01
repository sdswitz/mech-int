# Default Research Pipeline Refactor Plan

## Summary
Refactor the manual SAE research workflow into a reproducible, script-first pipeline before building automation. The refactor keeps the current research scope and model target unchanged, but separates activation collection, training, evaluation, analysis, and ablation into stable modules and CLIs. Notebooks become consumers of the new modules instead of the primary implementation surface.

## Key Changes
- Add a `mechint/` package for shared data loading, configuration, SAE definitions, evaluation, analysis, and ablation utilities.
- Add script entrypoints under `scripts/` for activation collection, training, evaluation, batch ablation, and cross-expansion comparison.
- Make the default training path config-driven with a small JSON-backed dataclass and CLI overrides.
- Standardize experiment artifacts under `runs/`, while keeping `activations/` shared and treating `saved_models/` plus `training_metrics/` as legacy outputs.
- Use deterministic train/validation splits and streamed device batches instead of moving the full activation tensor onto the accelerator.
- Extract notebook-only research logic into importable functions so inspection and ablation become reproducible.
- Archive duplicate notebooks and update repo validation plus documentation to match the refactored workflow.

## Public Interfaces
- `scripts/collect_activations.py`
  - Inputs: model, layer, hook, dataset, split, text count, output path, checkpoint interval, max chars, seed
  - Outputs: activation tensor and JSON metadata manifest
- `scripts/train_sae.py`
  - Inputs: JSON config plus CLI overrides
  - Outputs: run directory with checkpoint, metrics CSV, config snapshot, manifest, and evaluation summary
- `scripts/eval_sae.py`
  - Inputs: checkpoint plus config or run directory
  - Outputs: evaluation summary in stdout and optional JSON file
- `scripts/batch_ablate.py`
  - Inputs: checkpoint, evaluation text file, optional feature subset, output path
  - Outputs: ranked CSV of feature KL effects
- `scripts/compare_expansions.py`
  - Inputs: checkpoint list and activation dimension
  - Outputs: CSV of best decoder matches across checkpoints

## Test Plan
- Unit tests cover SAE tensor shapes, decoder renormalization, deterministic splits, and streamed batch loading.
- Integration tests cover training on a synthetic activation file, evaluation from a run directory, and checkpoint comparison output.
- Regression checks keep legacy checkpoints readable by the new evaluation path and preserve top-level `train.py` compatibility for the old CLI.

## Assumptions
- No new third-party dependencies are added.
- The default research target remains Pythia-70M layer 3 residual stream on the existing activation cache.
- JSON config files are sufficient; no heavier config framework is introduced.
- Legacy files remain available during the transition, but the default path for future work is the new script-and-runs pipeline.
