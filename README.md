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

SNIPS test split, best-dev-sentence-accuracy checkpoint, fp32, one seed unless noted. Intent accuracy / slot F1 / sentence accuracy (%). Run-to-run noise is about a point, so differences smaller than that are not meaningful.

| Model                                                   | Intent | Slot F1 | Sentence |
| ------------------------------------------------------- | ------ | ------- | -------- |
| JointBERT paper, BERT-base                              | 98.6   | 97.0    | 92.8     |
| BERT-base, paper settings, no grad clipping (mean of 3) | 98.1   | 95.3    | 89.2     |
| BERT-base, paper settings + grad clip 1.0               | 98.4   | 95.8    | 90.6     |
| ModernBERT-base, default config (lr 5e-5)               | 98.3   | 95.3    | 89.9     |
| ModernBERT-base, lr 2e-4 (best dev in ablations)        | 98.1   | 96.0    | 91.1     |

Paper settings: batch 128, lr 5e-5, dropout 0.1, 30 epochs, max length 50, BERT pooler for the intent head.

UltraBERT-150M (`dmis-lab/UltraBERT-150M`, first-token pooling) uses the same config as ModernBERT, run with `uv run sweep data/snips --base_model dmis-lab/UltraBERT-150M --runs baseline lr2e-4`. Dev sentence accuracy was 90.4 at lr 5e-5 and 85.1 at 2e-4, so unlike ModernBERT it prefers the lower rate.

### ModernBERT-base ablations

`uv run sweep`: each run changes one field from the default config and is ranked on dev sentence accuracy (test shown for reference, never used to select).

| Run       | Dev sent. | Test intent | Test slot F1 | Test sent. |
| --------- | --------- | ----------- | ------------ | ---------- |
| lr2e-4    | 89.6      | 98.1        | 96.0         | 91.1       |
| bs16      | 89.1      | 98.0        | 95.5         | 89.4       |
| mean-pool | 89.0      | 97.7        | 95.5         | 90.0       |
| patience6 | 88.9      | 98.6        | 94.7         | 88.9       |
| slotw2    | 88.9      | 98.3        | 95.6         | 90.3       |
| lr1e-4    | 88.1      | 98.1        | 95.9         | 91.0       |
| baseline  | 87.6      | 98.3        | 95.3         | 89.9       |
| lr2e-5    | 84.7      | 97.6        | 92.7         | 84.1       |

Findings:

- Two pipeline bugs were inflating ModernBERT's gap. It was trained without gradient clipping (the reference implementation clips to norm 1.0), and its BPE tokenizer was fed word by word, which drops the leading-space marker the model was pretrained with. Fixing both took the default config from 93.1 / 84.4 (slot F1 / sentence) to 95.3 / 89.9.
- With the fixes, ModernBERT-base at lr 2e-4 is on par with the BERT-base control (96.0 / 91.1 vs 95.8 / 90.6), within noise. Higher learning rates still help (dev 84.7 at 2e-5, 87.6 at 5e-5, 89.6 at 2e-4), and the optimum may be above 2e-4.
- Mean pooling and first-token pooling are within noise of each other for ModernBERT.
- UltraBERT-150M at the default config (96.3 slot F1, 91.4 sentence) is the best result so far, about 1 point above the BERT control and ModernBERT, which is at the edge of the noise level (one seed). It still trails the paper's 97.0 / 92.8.
- Both BERT-base and ModernBERT-base are still 1+ slot F1 and 2+ sentence points below the paper's 97.0 / 92.8, so something in the pipeline or the paper's unstated details (LR schedule, checkpoint selection, the intent representation) still differs. The reference implementation's own SNIPS numbers (99.14 / 96.90 / 93.00) are the best of several runs scored on test, so they are optimistic.

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
