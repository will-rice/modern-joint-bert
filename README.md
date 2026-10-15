# modern-joint-bert

[JointBERT](https://arxiv.org/abs/1902.10909) (Chen et al., 2019) jointly predicts utterance intent and slot labels with a single BERT encoder. This project asks: **how good can we get on the SNIPS benchmark by swapping BERT for modern embedding models?**

## Premise

- Keep the JointBERT recipe: pooled `[CLS]` representation → intent head, per-token representations → slot head, summed cross-entropy loss.
- Replace the encoder with modern embedding models and compare them under identical training conditions (clean ablations: only the encoder changes).
- Report intent accuracy, slot F1, and sentence-level semantic frame accuracy on SNIPS.

## Baseline

Original JointBERT results on SNIPS (BERT-base):

| Intent acc. | Slot F1 | Sentence acc. |
| ----------- | ------- | ------------- |
| 98.6        | 97.0    | 92.8          |

## Status

Scaffolded from [ml-template](https://github.com/will-rice/ml-template). Data loading, model, and evaluation are not implemented yet.

## Quick Start

```bash
uv sync
cp .env.example .env  # add your wandb API key
uv run pre-commit install
uv run pytest
```

## Layout

```
src/modern_joint_bert/
├── config.py            # Pydantic configuration
├── lightning_module.py  # Lightning module
├── datasets/            # SNIPS dataset
├── modeling/            # Encoder + intent/slot heads
└── scripts/train.py     # Training entry point (`uv run train`)
```

## License

See [LICENSE](LICENSE).
