from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class SAETrainConfig:
    model_name: str = "pythia-70m"
    model_source: str = "EleutherAI/pythia-70m"
    layer: int = 3
    hook_name: str = "blocks.3.hook_resid_post"
    dataset_name: str = "openwebtext"
    dataset_split: str = "train"
    num_texts: int = 50000
    max_chars: int = 512
    activation_dim: int = 512
    expansion: int = 8
    l1_lambda: float = 3.0
    epochs: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    warmup_steps: int = 2000
    lambda_warmup_steps: int = 5000
    max_grad_norm: float = 1.0
    val_fraction: float = 0.05
    seed: int = 42
    activations_path: str = "activations/activations_50000.pt"
    runs_root: str = "runs"
    run_name: str | None = None
    eval_batch_size: int = 8192
    device: str | None = None
    legacy_layout: bool = False

    @property
    def num_features(self) -> int:
        return self.activation_dim * self.expansion

    def with_overrides(self, **kwargs: Any) -> "SAETrainConfig":
        data = asdict(self)
        for key, value in kwargs.items():
            if value is not None and key in data:
                data[key] = value
        return SAETrainConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "SAETrainConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)
