"""Tests for SNIPS data and metrics."""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modern_joint_bert.config import Config
from modern_joint_bert.datasets import (
    IGNORE_INDEX,
    SnipsDataset,
    collate,
    download_snips,
)
from modern_joint_bert.metrics import compute_metrics, word_level_tags


@pytest.fixture(
    scope="module",
    params=[Config().base_model, "bert-base-uncased", "dmis-lab/UltraBERT-150M"],
)
def dataset(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> SnipsDataset:
    """Real SNIPS train split with the default and paper models' tokenizers."""
    data_root: Path = tmp_path_factory.mktemp("snips")
    download_snips(data_root)
    tokenizer = AutoTokenizer.from_pretrained(request.param)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return SnipsDataset(data_root, "train", tokenizer, Config().max_length)


def test_snips_sizes_and_labels(dataset: SnipsDataset) -> None:
    """Train split matches the published SNIPS statistics."""
    assert len(dataset) == 13084
    assert len(dataset.labels.intents) == 7
    assert len(dataset.labels.slots) == 72


def test_slot_labels_align_to_first_subword(dataset: SnipsDataset) -> None:
    """Each word gets exactly one slot label."""
    for index in range(50):
        item = dataset[index]
        labeled = item["slot_labels"] != IGNORE_INDEX
        assert labeled.sum() == len(dataset.words[index])
        tags = [dataset.labels.slots[i] for i in item["slot_labels"][labeled]]
        assert tags == dataset.tags[index]


def test_collate_pads(dataset: SnipsDataset) -> None:
    """Collate pads ids and masks padding positions out."""
    pad_id = dataset.tokenizer.pad_token_id
    assert isinstance(pad_id, int)
    batch = collate([dataset[0], dataset[1]], pad_token_id=pad_id)
    assert batch.input_ids.shape == batch.slot_labels.shape
    lengths = torch.tensor([len(dataset[0]["input_ids"]), len(dataset[1]["input_ids"])])
    assert (batch.attention_mask.sum(1) == lengths).all()
    assert (batch.attention_mask[:, 0] == 1).all()
    assert (batch.slot_labels[batch.attention_mask == 0] == IGNORE_INDEX).all()


def test_metrics_perfect_and_partial() -> None:
    """Sentence accuracy needs both intent and every slot correct."""
    slots = ["O", "B-a"]
    labels = torch.tensor([[IGNORE_INDEX, 0, 1], [IGNORE_INDEX, 1, 0]])
    logits = torch.nn.functional.one_hot(
        torch.tensor([[0, 0, 1], [0, 0, 0]]), 2
    ).float()
    predicted, gold = word_level_tags(logits, labels, slots)
    assert gold == [["O", "B-a"], ["B-a", "O"]]
    assert predicted == [["O", "B-a"], ["O", "O"]]
    metrics = compute_metrics([True, True], predicted, gold)
    assert metrics["intent_acc"] == 1.0
    assert metrics["sentence_acc"] == 0.5
    assert compute_metrics([True], gold[:1], gold[:1])["slot_f1"] == 1.0
