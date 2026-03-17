# Mechanistic Interpretability: Sparse Autoencoder Pipeline

## Current State

Training and inspecting sparse autoencoders (SAEs) on **Pythia-70M, layer 3 residual stream** (512-dimensional activations).

- **SAE architecture:** 8x expansion factor (4096 features), tied decoder weights
- **Training hyperparameters:** L1 lambda=3, lr=3e-4, cosine LR schedule with linear warmup, lambda warmup, gradient clipping
- **Results:** L0 ~ 0.05, MSE ~ 0.02, 4 dead features out of 4096
- **Notable features discovered:**
  - "Reuters" detector (fires on Reuters-sourced news text)
  - "Feb" detector (month-of-February token)
  - Prefix detectors (" un", " re")
- **Pipeline:** `train.py` for training (supports `--expansion` and `--lam` CLI args), `inspect-features.ipynb` for feature analysis, `ablation.ipynb` for causal ablation, `colab-training.ipynb` for GPU training via Colab

---

## Roadmap

Suggested next steps: 3 (finish ablation) → 2 → cross-expansion analysis

### 1. Training Improvements ✅

- Cosine LR schedule with linear warmup (2K steps)
- Lambda warmup (ramps from 0 to target over 5K steps)
- 50K epochs (up from 20K)
- Gradient clipping (max norm 1.0)
- CLI args for expansion factor and lambda (`python train.py --expansion 16 --lam 3`)
- Auto-skip if model already exists

### 2. Multi-Expansion-Factor Training

Train SAEs at multiple widths to study how features split and refine at larger scales.

- `train.py` already supports `--expansion 4/8/16/32` with per-expansion output files
- Train remaining expansion factors, tuning lambda empirically per scale
- Compare learned features across scales using decoder cosine similarity
- Detect feature splitting (one feature at small scale → multiple features at larger scale)

### 3. Causal Ablation Pipeline (in progress)

Measure the causal importance of individual features. **File:** `ablation.ipynb`

**Done:**
- Single-feature ablation via TransformerLens hooks (subtract `f_i * W_dec[i]` from residual stream)
- KL divergence between clean and ablated output distributions
- Per-token KL breakdown and top-5 prediction comparison
- Greedy generation comparison (clean vs ablated)

**Key finding:** Ablating the "Reuters" detector (feature 3440) produces near-zero KL and identical greedy output — suggesting it's a recognizer/label rather than a causal driver of generation at layer 3.

**Next steps for ablation:**
- Batch-ablate all 4096 features across a diverse eval set (100–500 texts) to rank features by mean KL divergence
- Identify which features are causally important vs. merely correlated
- Cross-reference causal importance ranking with interpretability scores from `inspect-features.ipynb`
- Investigate whether causally important features tend to be interpretable or polysemantic
- Try ablating multiple related features simultaneously to detect feature circuits

### 4. Decoder Weight Cosine Similarity Analysis ✅

- Pairwise cosine similarity of all decoder vectors
- Similarity distribution histogram
- High-similarity pair inspection (filtered to multi-character tokens)
- Hierarchical clustering dendrogram
- Reverse token search (`search_token()`) to find which features fire on a given token
- **Added to:** `inspect-features.ipynb`
