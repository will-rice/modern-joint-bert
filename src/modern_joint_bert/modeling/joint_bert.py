"""JointBERT with a swappable Hugging Face encoder."""

import torch
from torch import nn
from transformers import AutoModel

from modern_joint_bert.config import Pooling


class JointBert(nn.Module):
    """Encoder with a pooled intent head and a per-token slot head."""

    def __init__(
        self,
        base_model: str,
        num_intents: int,
        num_slots: int,
        dropout: float,
        pooling: Pooling,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.pooling = pooling
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_slots)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict intent and slot logits.

        Args:
            input_ids: Token ids with shape (batch, tokens).
            attention_mask: Attention mask with shape (batch, tokens).

        Returns:
            Intent logits (batch, intents) and slot logits (batch, tokens, slots).
        """
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        match self.pooling:
            case "pooler":
                pooled = output.pooler_output
            case "cls":
                pooled = hidden[:, 0]
            case "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
        return (
            self.intent_head(self.dropout(pooled)),
            self.slot_head(self.dropout(hidden)),
        )
