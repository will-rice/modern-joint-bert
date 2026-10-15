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

## Results

Single run, seed 42, default `config.py`, fp32, test split, best dev-sentence-accuracy checkpoint:

| Encoder                     | Intent acc. | Slot F1 | Sentence acc. |
| --------------------------- | ----------- | ------- | ------------- |
| BERT-base (JointBERT paper) | 98.6        | 97.0    | 92.8          |
| ModernBERT-base             | 97.6        | 92.6    | 84.7          |

The first ModernBERT-base run is below the paper baseline, so it is untuned: no learning-rate or epoch sweep, a single seed, and the intent head pools the first token. Next steps are tuning shared hyperparameters and comparing more encoders.

## Quick Start

```bash
uv sync
cp .env.example .env  # add your wandb API key
uv run pre-commit install
uv run pytest
```

## Usage

SNIPS (the JointBERT split, 13,084 / 700 / 700) is downloaded into `data_root` on first run. The encoder is selected with `--base_model` (any Hugging Face encoder with a fast tokenizer whose first token summarizes the sequence; default `answerdotai/ModernBERT-base`). Hyperparameters live in `config.py` and are shared across encoders so comparisons are clean ablations.

```bash
uv run train data/snips --base_model answerdotai/ModernBERT-base
uv run sweep data/snips  # one-factor-at-a-time ablations, ranked on dev
```

Checkpoints are selected on dev sentence accuracy; test metrics are logged to wandb at the end.

## Layout

```
src/modern_joint_bert/
├── config.py            # Pydantic configuration
├── metrics.py           # intent acc, slot F1 (seqeval), sentence acc
├── lightning_module.py  # JointBertModule: joint loss, eval, optimizer
├── datasets/snips.py    # SNIPS download, tokenization, label alignment
├── modeling/joint_bert.py  # AutoModel encoder + intent/slot heads
└── scripts/train.py     # `uv run train`
```

## License

See [LICENSE](LICENSE).
