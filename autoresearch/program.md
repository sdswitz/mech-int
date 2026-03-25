# SAE Autoresearch

Autonomous experimentation loop for Sparse Autoencoder research on Pythia-70M activations.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `sae-mar23`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The autoresearch folder is small. Read these files for full context:
   - `prepare.py` — fixed constants, data loading, evaluation, metric computation. Do not modify.
   - `train.py` — the file you modify. SAE architecture, optimizer, training loop.
   - `../context.md` — background on SAE theory and this project's approach.
   - `../sae.py` — reference SAE implementation (not used by the auto loop, just for context).
4. **Verify activations exist**: Check that `../activations/activations_50000.pt` exists (~12GB file). If not, tell the human to run `python ../train.py` first to collect activations.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row (see format below). Leave it untracked by git.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device (MPS on Mac, or CUDA on GPU). The training script runs for a **fixed time budget of 2 minutes** (wall clock training time, excluding startup and eval). You launch it from the `autoresearch/` directory as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: SAE architecture, activation function (ReLU, TopK, JumpReLU, BatchTopK), encoder/decoder structure (tied weights, gated SAE), optimizer, learning rate schedule, initialization strategy, batch size, expansion factor, lambda, warmup, ghost gradients, feature resampling, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, validation split, etc).
- Install new packages or add dependencies. You can only use what's in `../requirements.txt`.
- Break the SAE interface contract. The model must have `encoder(x)->f`, `decoder(f)->xhat`, and `forward(x)->(xhat, f)` methods.

**The goal is simple: get the highest sae_score.** Since the time budget is fixed, you don't need to worry about training time — it's always 2 minutes. The `sae_score` is a composite metric:

```
sae_score = -log10(val_mse) - 0.5 * log10(val_l0) - 0.1 * dead_frac
```

- Higher is better
- `val_mse`: reconstruction error on held-out validation set (lower is better)
- `val_l0`: fraction of features active per input (lower is better — sparser is more interpretable)
- `dead_frac`: fraction of features that never activated during training (lower is better)
- MSE dominates the score; sparsity is a secondary objective; dead features are penalized

The individual metrics are always printed alongside `sae_score`, so you can reason about the tradeoffs. But `sae_score` is the single number that determines keep/discard.

**Memory** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. A small improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline — run the training script as is, without modifications.

## Output format

Once the script finishes it prints a summary like this:

```
---
sae_score:        2.349000
val_mse:          0.020000
val_l0:           0.050000
dead_frac:        0.001000
training_seconds: 118.3
total_seconds:    125.7
num_features:     4096
expansion:        8
num_params:       4.2M
```

You can extract the key metric from the log file:

```
grep "^sae_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	sae_score	val_mse	val_l0	dead_frac	status	description
```

1. git commit hash (short, 7 chars)
2. sae_score achieved (e.g. 2.349000) — use 0.000000 for crashes
3. val_mse (e.g. 0.020000) — use 0.000000 for crashes
4. val_l0 (e.g. 0.050000) — use 0.000000 for crashes
5. dead_frac (e.g. 0.001000) — use 0.000000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	sae_score	val_mse	val_l0	dead_frac	status	description
a1b2c3d	2.349000	0.020000	0.050000	0.001000	keep	baseline
b2c3d4e	2.410000	0.018000	0.048000	0.000500	keep	lower lambda to 2.0
c3d4e5f	2.100000	0.035000	0.030000	0.010000	discard	TopK k=32 (MSE regressed)
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	32x expansion (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/sae-mar23`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^sae_score:\|^val_mse:\|^val_l0:\|^dead_frac:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If sae_score improved (higher), you "advance" the branch, keeping the git commit
9. If sae_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck, you can rewind but you should probably do this very sparingly (if ever).

**Timeout**: Each experiment should take ~2.5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Research directions

These are suggestions to get you started, not an exhaustive list:

- **Lambda sweep**: The L1 penalty coefficient is the most impactful single hyperparameter. Try values from 0.5 to 10.
- **TopK activation**: Replace ReLU with a TopK function that keeps only the k highest activations and zeros the rest. From Anthropic's "Scaling Monosemanticity" work.
- **JumpReLU**: A threshold-based activation where `f = x * (x > threshold)` with a learned threshold. From DeepMind's "Jumping Ahead" paper.
- **Ghost gradients / feature resampling**: Periodically reinitialize dead features by sampling from high-loss inputs.
- **Gated SAE**: Separate gating and magnitude pathways in the encoder.
- **Tied encoder-decoder weights**: Set `W_enc = W_dec.T` to reduce parameters and enforce consistency.
- **Larger expansion factors**: Try 16x or 32x for more features (but watch for dead features and memory).
- **Pre-bias initialization**: Initialize `b_pre` to the dataset mean instead of zero.
- **Different optimizers**: Try AdamW with weight decay, or different learning rate schedules.
- **Initialization strategies**: Kaiming vs Xavier, or initializing encoder from decoder transpose.
- **Batch size**: Larger batches may give more stable gradients; smaller may allow more steps.
