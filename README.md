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

SNIPS test split, best-dev-sentence-accuracy checkpoint, fp32. Intent accuracy / slot F1 / sentence accuracy (%).

| Model                                                    | Intent | Slot F1 | Sentence |
| -------------------------------------------------------- | ------ | ------- | -------- |
| JointBERT paper, BERT-base                               | 98.6   | 97.0    | 92.8     |
| BERT-base, paper settings, no grad clipping (mean of 3)  | 98.1   | 95.3    | 89.2     |
| BERT-base, paper settings + grad clip 1.0 (1 run)        | 98.4   | 95.8    | 90.6     |
| ModernBERT-base, default config (1 seed)                 | 98.6   | 93.1    | 84.4     |
| ModernBERT-base, lr 2e-4 (1 seed, best dev in ablations) | 98.4   | 94.6    | 88.6     |

Paper settings: batch 128, lr 5e-5, dropout 0.1, 30 epochs, max length 50, BERT pooler for the intent head.

**Caveat:** every row except the grad-clip one was trained without gradient clipping. The reference implementation ([monologg/JointBERT](https://github.com/monologg/JointBERT)) clips to norm 1.0 every step, and `Config.grad_clip_norm` now does the same, so those rows predate that fix and need re-running. The reference's own SNIPS numbers (99.14 / 96.90 / 93.00) are the best of several 5-10 epoch runs scored on test, so they are optimistic. Ours select on dev and average seeds.

Findings so far:

- Gradient clipping helps BERT-base (slot F1 95.3 to 95.8, sentence 89.2 to 90.6, though the first row is a 3-seed mean and the second a single run), but it still lands 1.2 slot F1 and 2.2 sentence accuracy below the paper, so something else in the pipeline or the paper's unstated details (LR schedule, checkpoint selection, the intent representation) still differs.
- ModernBERT-base needs a much higher learning rate than BERT (dev sentence accuracy 76.4 at 2e-5, 83.4 at 5e-5, 87.7 at 2e-4) and is roughly on par with the BERT control at 2e-4. The best rate may be above 2e-4.
- Run-to-run noise is about a point, so single-seed differences below that are not meaningful.

Ablations (`uv run sweep`) change one field at a time and are ranked on dev, never test.

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
