# Sparse Autoencoders for Mechanistic Interpretability

## Motivation

Neural networks likely represent more concepts than they have dimensions. A model with a residual stream of dimension $d$ may internally encode $m \gg d$ distinct features, packed into the space via approximately (but not exactly) orthogonal directions. This is the **superposition hypothesis**. It makes individual neurons hard to interpret, since each neuron responds to a mixture of unrelated concepts.

A **sparse autoencoder (SAE)** attempts to reverse this: it decomposes a dense activation vector into a sparse linear combination over a learned overcomplete dictionary of feature directions, effectively "unpacking" superposition into interpretable components.

---

## Architecture

### Setup

At a chosen layer $l$ of a transformer, each token position produces a residual stream vector:

$$\mathbf{x} \in \mathbb{R}^d$$

This is the object we decompose. Common hook points in TransformerLens:

- `blocks.{layer}.hook_resid_post` — residual stream after the full block (most standard)
- `blocks.{layer}.hook_resid_mid` — between attention and MLP
- `blocks.{layer}.hook_resid_pre` — before the block

### Encoder

Subtract a learned pre-bias (captures the mean activation direction), project up to a high-dimensional space, and apply ReLU:

$$\mathbf{f} = \text{ReLU}\!\Big(W_{\text{enc}}(\mathbf{x} - \mathbf{b}_{\text{pre}}) + \mathbf{b}_{\text{enc}}\Big)$$

- $W_{\text{enc}} \in \mathbb{R}^{m \times d}$ — encoder weights
- $\mathbf{b}_{\text{pre}} \in \mathbb{R}^d$ — pre-bias (initialized to zero)
- $\mathbf{b}_{\text{enc}} \in \mathbb{R}^m$ — encoder bias (initialized to zero)
- $\mathbf{f} \in \mathbb{R}^m$ — sparse feature activation vector ($m \gg d$)

The expansion factor $m / d$ is a key hyperparameter. Common values: 8x, 16x, 32x, 64x.

### Decoder

Reconstruct the original activation as a sparse weighted sum of learned feature directions:

$$\hat{\mathbf{x}} = W_{\text{dec}}\,\mathbf{f} + \mathbf{b}_{\text{pre}}$$

- $W_{\text{dec}} \in \mathbb{R}^{d \times m}$

Equivalently, writing $\mathbf{d}_i$ for the $i$-th column of $W_{\text{dec}}$:

$$\hat{\mathbf{x}} = \sum_{i=1}^{m} f_i\,\mathbf{d}_i + \mathbf{b}_{\text{pre}}$$

Each $\mathbf{d}_i$ is a **feature direction** in the residual stream. Each scalar $f_i$ tells you how strongly feature $i$ is active for a given input.

---

## Loss Function

$$\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{reconstruction (MSE)}} + \underbrace{\lambda\,\|\mathbf{f}\|_1}_{\text{sparsity}}$$

- The **MSE** term forces faithful reconstruction.
- The **L1** term penalizes having many nonzero $f_i$, pushing the model to explain each activation with as few features as possible.
- $\lambda$ controls the tradeoff: too low → dense, uninterpretable features; too high → poor reconstruction.

**Note on L1 convention:** $\|\mathbf{f}\|_1$ can be computed as a sum or mean over the entries. Either works — just be aware that the effective scale of $\lambda$ depends on which you choose. Using `f.abs().mean()` is common in practice.

---

## Training Details

### Data Collection

1. Choose a target layer and hook point (e.g., `blocks.3.hook_resid_post`).
2. Run a large corpus (OpenWebText, The Pile) through the frozen LLM.
3. Collect activation vectors $\mathbf{x} \in \mathbb{R}^d$ at that hook point across all token positions.
4. Flatten across batch and sequence dimensions → dataset of shape $(N, d)$.

The SAE never sees text. It only sees activation vectors. The LLM weights are frozen throughout.

### Initialization

- **Encoder/Decoder weights:** Xavier (Glorot) initialization.
- **Biases:** Zeros.
- **Decoder columns:** Normalize to unit norm at initialization.

```python
W_enc = torch.nn.init.xavier_normal_(torch.empty(d, m))
W_dec = torch.nn.init.xavier_normal_(torch.empty(m, d))
W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)  # unit-norm rows
b_pre = torch.zeros(d)
b_enc = torch.zeros(m)
```

### Decoder Norm Constraint

After each optimizer step, renormalize decoder rows to unit norm:

```python
optimizer.step()
with torch.no_grad():
    W_dec.data = W_dec.data / W_dec.data.norm(dim=1, keepdim=True)
```

Without this, the model cheats the L1 penalty by shrinking decoder norms and inflating feature activations. With the constraint, $f_i$ directly and meaningfully measures the magnitude of feature $i$'s contribution.

### Optimizer and Hyperparameters

- **Optimizer:** Adam (lr ~ 3e-4)
- **Expansion factor:** Start with 8x
- **$\lambda$:** Start around 1e-3, sweep to find the sparsity/reconstruction tradeoff
- **Batch size:** Standard (e.g., 4096 activation vectors per batch)

### Evaluation Metrics

- **Reconstruction fidelity:** Substitute SAE reconstructions for true activations in the LLM and measure "loss recovered" — how much of the model's original performance is preserved.
- **Average L0:** Mean number of nonzero features per input. Reasonable range: ~100–500 out of tens of thousands. If L0 is too high, increase $\lambda$.
- **Dead features:** Features that never activate across the dataset. A major failure mode. Track the fraction of dead features during training.

---

## Interpreting Features

After training, each of the $m$ features is just a direction in $\mathbb{R}^d$ with no label. Interpretation is empirical.

### Max-Activating Examples

Run a large corpus through the LLM → SAE pipeline. For each feature $i$, record which token contexts produce the highest $f_i$. Inspect the top 20–50 examples and look for patterns.

A feature is **monosemantic** if the pattern is clean (e.g., "fires on capital cities in geographic contexts"). A feature is **polysemantic** if it responds to apparently unrelated things — usually a sign the SAE needs more capacity or training.

### Activation Distributions

For each feature, plot the histogram of $f_i$ across the corpus. A good feature has a large spike at zero and a sparse tail of positive activations. Features active on >50% of inputs are likely capturing something generic.

### Logit Attribution

Since the LLM's output logits are approximately $W_U \mathbf{x}_{\text{final}}$, each feature's contribution to the logits is:

$$f_i\,(W_U\,\mathbf{d}_i)$$

This tells you which output tokens feature $i$ promotes or suppresses, independent of the max-activating examples.

### Causal Intervention (Ablation)

- **Zero ablation:** Set $f_i = 0$, re-inject the modified reconstruction into the model, observe how output changes.
- **Activation boosting:** Artificially increase $f_i$ and see if the model behaves as though the concept is present.

These give causal (not just correlational) evidence for what a feature does.

### Automated Interpretability

Feed max-activating examples to an LLM and ask it to generate a natural language description of the feature. Validate by using the description to predict activation on held-out examples.

---

## Key Concepts

- **Superposition:** The model encodes $m > d$ features as approximately orthogonal directions in $\mathbb{R}^d$. Individual neurons are mixtures of features.
- **Overcomplete dictionary:** The SAE's decoder columns $\{\mathbf{d}_i\}_{i=1}^m$ form an overcomplete basis. Sparsity selects which basis elements are active for a given input.
- **Monosemantic vs. polysemantic features:** The goal is monosemantic features (one concept per feature direction). Polysemantic features indicate incomplete disentanglement.
- **Dead features:** Dictionary elements that never activate. A common failure mode requiring resampling or ghost gradients.

---

## Practical Stack

- **Model:** Pythia-70M or Pythia-160M via TransformerLens (for easy hook access)
- **Activations:** Collected at a single layer to start (e.g., layer 3 or 4 for Pythia-70M's 6 layers)
- **SAE:** Implemented from scratch in PyTorch
- **Corpus:** OpenWebText or The Pile via HuggingFace Datasets

---

## Next Steps

1. Get TransformerLens running with Pythia-70M. Inspect residual stream shapes and activations at each layer.
2. Build an activation collection pipeline for a single layer.
3. Implement the SAE as an `nn.Module` with proper initialization, loss, and decoder norm constraint.
4. Train on collected activations. Monitor MSE, L1, L0, and dead feature count independently.
5. Inspect max-activating examples for a handful of features. See if they're interpretable.
6. Experiment with logit attribution and ablation to build causal understanding of features.
7. Once the single-layer pipeline is solid, consider training SAEs at multiple layers and comparing how features evolve through depth.