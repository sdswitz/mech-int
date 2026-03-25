# Plan: Autoresearch Framework for SAE Experiments

## Context

The mech-int project has a working SAE training pipeline (train.py, sae.py) and analysis notebooks (inspect-features.ipynb, ablation.ipynb) for mechanistic interpretability of Pythia-70M. The autoresearch project provides a proven autonomous experimentation framework: an agent edits one file, runs time-budgeted training, measures a single scalar metric, and keeps/discards via git — looping indefinitely without human intervention.

The goal is to adapt autoresearch's scaffolding so an autonomous agent can iterate on SAE architecture and hyperparameters overnight, producing a results.tsv of experiments ranked by quality.

---

## Files (all in `autoresearch/`)

### 1. `prepare.py` — Immutable evaluation & data loading

**Role:** The fixed harness. Agent must never modify this file.

**Constants:**
- `ACTIVATION_DIM = 512`
- `ACTIVATIONS_PATH = "../activations/activations_50000.pt"`
- `VAL_FRACTION = 0.05` (~294K held-out vectors)
- `VAL_SEED = 42` (deterministic split)
- `TIME_BUDGET = 120` (2 min wall clock)
- `EVAL_BATCH_SIZE = 8192`

**Key functions:**
- `load_activations()` — mmap-backed loading, deterministic train/val split, CPU tensors
- `make_dataloader()` — infinite random batch iterator, moves to device on demand
- `evaluate_sae()` — computes val_mse, val_l0, dead_frac, and composite sae_score

**The composite metric — `sae_score`:**
```
sae_score = -log10(val_mse) - 0.5 * log10(val_l0) - 0.1 * dead_frac
```
Higher is better. MSE weighted most, sparsity secondary, dead features penalized.

### 2. `train.py` — The modifiable training file

The ONLY file the agent edits. Contains embedded SAE class, hyperparameters, and time-budgeted training loop. Imports evaluation from prepare.py.

### 3. `program.md` — Agent instructions

Adapted from the autoresearch project. Defines setup, constraints, experiment loop, keep/discard logic, results.tsv format, and NEVER STOP directive.

### 4. `analysis.ipynb` — Results visualization

SAE score over time, MSE vs L0 Pareto plot, dead features tracking, summary stats, top hits.

## How to Use

```bash
cd mech-int/autoresearch
# Start a new Claude Code session and point it at program.md
```

The agent will create a branch, establish a baseline, then loop autonomously.
