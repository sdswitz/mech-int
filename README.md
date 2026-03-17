# Mechanistic Interpretability: Sparse Autoencoder Pipeline

## Current State

Training and inspecting sparse autoencoders (SAEs) on **Pythia-70M, layer 3 residual stream** (512-dimensional activations).

- **SAE architecture:** 8x expansion factor (4096 features), tied decoder weights
- **Training hyperparameters:** L1 lambda=3, lr=3e-4, 20K epochs on ~6M tokens from OpenWebText
- **Results:** L0 ~ 0.05, MSE ~ 0.02, 4 dead features out of 4096
- **Notable features discovered:**
  - "Reuters" detector (fires on Reuters-sourced news text)
  - "Feb" detector (month-of-February token)
  - Prefix detectors (" un", " re")
- **Pipeline:** `train.py` for training, `inspect-features.ipynb` for feature analysis, `colab-training.ipynb` for GPU training via Colab

---

## Roadmap

Suggested implementation order: 1 → 4 → 2 → 3

### 1. Training Improvements

Improve SAE training quality and stability before scaling up.

- Add a learning rate scheduler (cosine decay with linear warmup)
- Implement lambda warmup (start low, ramp to target value over early training)
- Scale to longer training runs (50K–100K epochs)
- Add gradient clipping to stabilize training
- **Modify:** `train.py`

### 2. Multi-Expansion-Factor Training

Train SAEs at multiple widths to study how features split and refine at larger scales.

- Parameterize the expansion factor as a CLI argument
- Train at 4x, 8x, 16x, and 32x expansion
- Compare learned features across scales using decoder cosine similarity
- Detect feature splitting (one feature at small scale → multiple features at larger scale)
- Document recommended lambda values per expansion factor
- **Modify:** `train.py`; add analysis section to a notebook

### 3. Causal Ablation Pipeline

Move beyond correlation to measure the causal importance of individual features.

- Use TransformerLens hooks to subtract a feature's contribution from the residual stream
- Measure effect via KL divergence on output logits and top-token changes
- Batch-ablate all features to produce a causal importance ranking
- **Create:** `ablation.ipynb`

### 4. Decoder Weight Cosine Similarity Analysis

Understand the geometry of the learned feature dictionary.

- Compute pairwise cosine similarity of all decoder weight vectors
- Plot the similarity distribution as a histogram
- Inspect high-similarity pairs to find redundant or related features
- Run hierarchical clustering and visualize as a dendrogram
- Compare similarity structure across expansion factors (after completing section 2)
- **Add to:** `inspect-features.ipynb`
