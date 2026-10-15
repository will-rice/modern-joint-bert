"""SNIPS intent and slot dataset in the JointBERT file format."""

from pathlib import Path
from typing import NamedTuple
from urllib.request import urlretrieve

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

SNIPS_URL = "https://raw.githubusercontent.com/monologg/JointBERT/master/data/snips"
SPLITS = ("train", "dev", "test")
FILES = ("seq.in", "seq.out", "label")
IGNORE_INDEX = -100


class Batch(NamedTuple):
    """Batch of data.

    Attributes:
        input_ids: Token ids with shape (batch, tokens).
        attention_mask: Attention mask with shape (batch, tokens).
        intent_label: Intent ids with shape (batch,).
        slot_labels: Slot ids with shape (batch, tokens). Only the first
            sub-token of each word is labeled; the rest are IGNORE_INDEX.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    intent_label: torch.Tensor
    slot_labels: torch.Tensor


class SnipsLabels(NamedTuple):
    """Intent and slot label vocabularies built from the train split."""

    intents: list[str]
    slots: list[str]


def download_snips(data_root: Path) -> None:
    """Download the SNIPS splits used by JointBERT into data_root."""
    for split in SPLITS:
        (data_root / split).mkdir(parents=True, exist_ok=True)
        for name in FILES:
            path = data_root / split / name
            if not path.exists():
                urlretrieve(f"{SNIPS_URL}/{split}/{name}", path)


def read_lines(path: Path) -> list[str]:
    """Read stripped lines from a file."""
    return [line.strip() for line in path.read_text().splitlines()]


class SnipsDataset(Dataset):
    """SNIPS split with word-level slot labels aligned to sub-tokens."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.words = [line.split() for line in read_lines(data_root / split / "seq.in")]
        self.tags = [line.split() for line in read_lines(data_root / split / "seq.out")]
        self.intents = read_lines(data_root / split / "label")
        train_tags = [
            line.split() for line in read_lines(data_root / "train" / "seq.out")
        ]
        self.labels = SnipsLabels(
            intents=sorted(set(read_lines(data_root / "train" / "label"))),
            slots=sorted({tag for tags in train_tags for tag in tags}),
        )

    def __len__(self) -> int:
        """Number of utterances."""
        return len(self.words)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize one utterance and align slot tags to first sub-tokens."""
        encoding = self.tokenizer(
            self.words[index],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        slot_labels = torch.full((len(encoding["input_ids"]),), IGNORE_INDEX)
        previous_word = None
        for position, word in enumerate(encoding.word_ids()):
            if word is not None and word != previous_word:
                slot_labels[position] = self.labels.slots.index(self.tags[index][word])
            previous_word = word
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "intent_label": torch.tensor(
                self.labels.intents.index(self.intents[index])
            ),
            "slot_labels": slot_labels,
        }


def collate(items: list[dict[str, torch.Tensor]], pad_token_id: int) -> Batch:
    """Pad a list of utterances into a Batch.

    The attention mask comes from sequence lengths, not from comparing ids to
    the pad id, because some tokenizers reuse the pad id for a real token.
    """
    lengths = [len(item["input_ids"]) for item in items]
    return Batch(
        input_ids=pad_sequence(
            [item["input_ids"] for item in items],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        attention_mask=(
            torch.arange(max(lengths)) < torch.tensor(lengths).unsqueeze(1)
        ).long(),
        intent_label=torch.stack([item["intent_label"] for item in items]),
        slot_labels=pad_sequence(
            [item["slot_labels"] for item in items],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ),
    )
