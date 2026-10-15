"""Main config class."""

from typing import Literal

from pydantic import BaseModel

Pooling = Literal["pooler", "cls", "mean"]


class Config(BaseModel):
    """Main config class."""

    # Reproducibility
    seed: int = 42

    # Data
    batch_size: int = 32
    max_length: int = 64

    # Training
    max_epochs: int = 10
    early_stopping_patience: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    slot_loss_weight: float = 1.0

    # Model
    base_model: str = "answerdotai/ModernBERT-base"
    # Intent representation: encoder pooler head, first token, or masked mean.
    pooling: Pooling = "cls"
