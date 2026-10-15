"""JointBERT evaluation metrics: intent accuracy, slot F1, sentence accuracy."""

import torch
from seqeval.metrics import f1_score

from modern_joint_bert.datasets import IGNORE_INDEX


def word_level_tags(
    slot_logits: torch.Tensor, slot_labels: torch.Tensor, slots: list[str]
) -> tuple[list[list[str]], list[list[str]]]:
    """Gather predicted and gold tag strings at labeled (first sub-token) positions.

    Args:
        slot_logits: Slot logits with shape (batch, tokens, slots).
        slot_labels: Gold slot ids with shape (batch, tokens).
        slots: Slot id to tag string.

    Returns:
        Predicted and gold tags for each utterance.
    """
    predictions = slot_logits.argmax(-1)
    mask = slot_labels != IGNORE_INDEX
    predicted = [
        [slots[i] for i in row[m]] for row, m in zip(predictions, mask, strict=True)
    ]
    gold = [
        [slots[i] for i in row[m]] for row, m in zip(slot_labels, mask, strict=True)
    ]
    return predicted, gold


def compute_metrics(
    intent_correct: list[bool],
    predicted_tags: list[list[str]],
    gold_tags: list[list[str]],
) -> dict[str, float]:
    """Compute intent accuracy, slot F1, and semantic frame (sentence) accuracy."""
    slot_correct = [p == g for p, g in zip(predicted_tags, gold_tags, strict=True)]
    return {
        "intent_acc": sum(intent_correct) / len(intent_correct),
        "slot_f1": float(f1_score(gold_tags, predicted_tags)),
        "sentence_acc": sum(
            i and s for i, s in zip(intent_correct, slot_correct, strict=True)
        )
        / len(intent_correct),
    }
