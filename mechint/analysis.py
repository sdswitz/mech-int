from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def collect_top_activations(
    sae,
    activations: torch.Tensor,
    k: int = 20,
    batch_size: int = 4096,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    num_features = sae.W_dec.shape[0]
    top_vals = torch.full((num_features, k), -float("inf"))
    top_idxs = torch.full((num_features, k), -1, dtype=torch.long)

    sae.eval()
    with torch.no_grad():
        for start in range(0, activations.shape[0], batch_size):
            batch = activations[start : start + batch_size].to(device)
            features = sae.encoder(batch).cpu()
            combined_vals = torch.cat([top_vals, features.T], dim=1)
            batch_idxs = (
                torch.arange(start, start + features.shape[0], dtype=torch.long)
                .unsqueeze(0)
                .expand(num_features, -1)
            )
            combined_idxs = torch.cat([top_idxs, batch_idxs], dim=1)
            vals, order = combined_vals.topk(k, dim=1)
            idxs = combined_idxs.gather(1, order)
            top_vals, top_idxs = vals, idxs
    return top_vals, top_idxs


def search_token(feature_texts: dict[int, list[str]], query: str) -> list[int]:
    return [feature_idx for feature_idx, tokens in feature_texts.items() if any(query in token for token in tokens)]


def feature_consistency_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return max(counts.values()) / len(tokens)


def decoder_cosine_similarity(sae) -> torch.Tensor:
    return F.cosine_similarity(sae.W_dec[:, None, :], sae.W_dec[None, :, :], dim=-1)


def top_decoder_pairs(sae, top_k: int = 20) -> list[tuple[int, int, float]]:
    sim = decoder_cosine_similarity(sae).clone()
    sim.fill_diagonal_(-1.0)
    values, flat_indices = sim.flatten().topk(top_k)
    pairs = []
    num_features = sim.shape[0]
    for value, flat_idx in zip(values.tolist(), flat_indices.tolist()):
        left = flat_idx // num_features
        right = flat_idx % num_features
        if left < right:
            pairs.append((left, right, value))
    return pairs
