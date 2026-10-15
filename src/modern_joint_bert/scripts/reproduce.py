"""Reproduce the JointBERT paper's SNIPS result with BERT-base.

Settings follow the paper (Sec. 4.2): bert-base-uncased, max length 50,
batch size 128, Adam with lr 5e-5, dropout 0.1, and 30 epochs, which is the
setting behind its 98.6 intent / 97.0 slot F1 result. The paper does not state
the LR schedule; like the authors' reference implementation we use linear decay
with no warmup and no weight decay. The BERT pooler feeds the intent head. Runs
several seeds and reports the mean.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from modern_joint_bert.config import Config
from modern_joint_bert.scripts.train import train

PAPER_CONFIG = Config(
    base_model="bert-base-uncased",
    pooling="pooler",
    max_length=50,
    batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.0,
    warmup_ratio=0.0,
    dropout=0.1,
    max_epochs=30,
    early_stopping_patience=30,
)
SEEDS = (42, 43, 44)


def main() -> None:
    """Train PAPER_CONFIG for each seed and log per-seed and mean test metrics."""
    parser = ArgumentParser(description="Reproduce JointBERT on SNIPS.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--project", default="modern-joint-bert", type=str)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    results = [
        train(
            PAPER_CONFIG.model_copy(update={"seed": seed}),
            args.data_root,
            args.project,
            args.log_root,
            num_devices=1,
            num_workers=args.num_workers,
            tag=f"paper-seed{seed}",
        )
        for seed in SEEDS
    ]
    for name in ("test/intent_acc", "test/slot_f1", "test/sentence_acc"):
        logging.info(
            "%s: %s mean=%.4f",
            name,
            [round(result[name], 4) for result in results],
            mean(result[name] for result in results),
        )


if __name__ == "__main__":
    main()
