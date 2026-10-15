"""One-factor-at-a-time ablations around the default config.

Every run changes exactly one field from the default Config. Runs are ranked
by best dev sentence accuracy; test metrics are reported but never used to
select.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from modern_joint_bert.config import Config
from modern_joint_bert.scripts.train import train

ABLATIONS: dict[str, Config] = {
    "baseline": Config(),
    "lr2e-5": Config(learning_rate=2e-5),
    "lr1e-4": Config(learning_rate=1e-4),
    "lr2e-4": Config(learning_rate=2e-4),
    "bs16": Config(batch_size=16),
    "patience6": Config(early_stopping_patience=6, max_epochs=30),
    "slotw2": Config(slot_loss_weight=2.0),
    "mean-pool": Config(pooling="mean"),
}


def main() -> None:
    """Run every ablation and print a table ranked by dev sentence accuracy."""
    parser = ArgumentParser(description="Run JointBERT ablations on SNIPS.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--base_model", default=Config().base_model, type=str)
    parser.add_argument(
        "--runs", nargs="+", choices=list(ABLATIONS), default=list(ABLATIONS)
    )
    parser.add_argument("--project", default="modern-joint-bert", type=str)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    results = {}
    for tag in args.runs:
        ablation = ABLATIONS[tag]
        config = ablation.model_copy(update={"base_model": args.base_model})
        results[tag] = train(
            config,
            args.data_root,
            args.project,
            args.log_root,
            num_devices=1,
            num_workers=args.num_workers,
            tag=tag,
        )

    logging.info(
        "%-10s %8s %8s %8s %8s", "run", "dev_sent", "intent", "slot_f1", "sent"
    )
    for tag, metrics in sorted(
        results.items(), key=lambda item: -item[1]["dev/sentence_acc"]
    ):
        logging.info(
            "%-10s %8.4f %8.4f %8.4f %8.4f",
            tag,
            metrics["dev/sentence_acc"],
            metrics["test/intent_acc"],
            metrics["test/slot_f1"],
            metrics["test/sentence_acc"],
        )


if __name__ == "__main__":
    main()
