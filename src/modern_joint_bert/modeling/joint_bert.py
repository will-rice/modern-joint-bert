"""JointBERT with a swappable Hugging Face encoder."""

import torch
from torch import nn
from transformers import AutoModel


class JointBert(nn.Module):
    """Encoder with an intent head on the first token and a slot head per token."""

    def __init__(
        self, base_model: str, num_intents: int, num_slots: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
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
        hidden = self.dropout(
            self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        )
        return self.intent_head(hidden[:, 0]), self.slot_head(hidden)
