"""Train and evaluate JointBERT on SNIPS."""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from git import Repo
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from modern_joint_bert.config import Config
from modern_joint_bert.datasets import SnipsDataset, collate, download_snips
from modern_joint_bert.lightning_module import JointBertModule


def main() -> None:
    """Train script."""
    parser = ArgumentParser(description="Train JointBERT on SNIPS.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--project", default="modern-joint-bert", type=str)
    parser.add_argument("--base_model", default=None, type=str)
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    config = Config()
    if args.base_model:
        config = config.model_copy(update={"base_model": args.base_model})

    seed_everything(config.seed, workers=True)

    git_hash = Repo().head.object.hexsha[:7]
    model_name = config.base_model.split("/")[-1]
    experiment_path = args.log_root / f"{model_name}-{git_hash}"
    experiment_path.mkdir(exist_ok=True, parents=True)

    download_snips(args.data_root)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.base_model)
    datasets = {
        split: SnipsDataset(args.data_root, split, tokenizer, config.max_length)
        for split in ("train", "dev", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
        )
        for split, dataset in datasets.items()
    }

    module = JointBertModule(config, datasets["train"].labels)
    checkpoint = ModelCheckpoint(
        experiment_path, monitor="val/sentence_acc", mode="max"
    )
    trainer = Trainer(
        accelerator="auto",
        devices=args.num_devices,
        max_epochs=config.max_epochs,
        logger=WandbLogger(
            project=args.project,
            name=f"{model_name}-{git_hash}",
            save_dir=experiment_path,
        ),
        callbacks=[
            checkpoint,
            EarlyStopping(
                "val/sentence_acc",
                mode="max",
                patience=config.early_stopping_patience,
            ),
        ],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, loaders["train"], loaders["dev"])
    trainer.test(
        module,
        loaders["test"],
        ckpt_path=None if args.fast_dev_run else "best",
    )


if __name__ == "__main__":
    main()
