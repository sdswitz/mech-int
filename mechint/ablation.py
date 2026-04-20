from __future__ import annotations

from functools import partial
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
import transformer_lens
from tqdm.auto import tqdm

from .config import SAETrainConfig
from .sae import SparseAutoEncoder


def build_transformer_model(model_name: str):
    return transformer_lens.HookedTransformer.from_pretrained(model_name)


def load_sae_checkpoint(
    checkpoint_path: str | Path,
    activation_dim: int,
    expansion: int | None = None,
    device: str = "cpu",
):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if expansion is None:
        expansion = state_dict["W_dec"].shape[0] // activation_dim
    sae = SparseAutoEncoder(d=activation_dim, m=activation_dim * expansion)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    return sae


def single_feature_ablation_hook(sae, feature_idx: int):
    def hook(resid, hook=None):
        f = sae.encoder(resid)
        contribution = f[..., feature_idx].unsqueeze(-1) * sae.W_dec[feature_idx]
        return resid - contribution

    return hook


@torch.no_grad()
def compute_text_kl(model, text: str, hook_name: str, hook_fn) -> float:
    clean_logits = model(text)
    ablated_logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, hook_fn)])
    clean_logprobs = F.log_softmax(clean_logits.float(), dim=-1)
    ablated_logprobs = F.log_softmax(ablated_logits.float(), dim=-1)
    clean_probs = clean_logprobs.exp()
    kl_div = (clean_probs * (clean_logprobs - ablated_logprobs)).sum(dim=-1)
    return float(kl_div.mean().item())


def rank_features_by_ablation(
    sae,
    model,
    hook_name: str,
    texts: list[str],
    feature_subset: list[int] | None = None,
    show_progress: bool = True,
) -> list[dict]:
    indices = feature_subset or list(range(sae.W_dec.shape[0]))
    rows = []
    iterator = tqdm(indices, desc="Ablating features", unit="feature") if show_progress else indices
    for feature_idx in iterator:
        hook_fn = single_feature_ablation_hook(sae, feature_idx)
        kls = [compute_text_kl(model, text, hook_name, hook_fn) for text in texts]
        rows.append(
            {
                "feature_idx": feature_idx,
                "mean_kl": sum(kls) / max(len(kls), 1),
                "max_kl": max(kls) if kls else 0.0,
                "num_texts": len(kls),
            }
        )
    rows.sort(key=lambda row: row["mean_kl"], reverse=True)
    return rows


def save_ablation_rows(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
