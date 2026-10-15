"""Train and evaluate JointBERT on SNIPS."""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import wandb
from dotenv import load_dotenv
from git import Repo
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    config = Config()
    if args.base_model:
        config = Config(**{**config.model_dump(), "base_model": args.base_model})

    metrics = train(
        config,
        args.data_root,
        args.project,
        args.log_root,
        args.num_devices,
        args.num_workers,
    )
    logging.info(metrics)


def train(
    config: Config,
    data_root: Path,
    project: str,
    log_root: Path,
    num_devices: int,
    num_workers: int,
    tag: str = "baseline",
) -> dict[str, float]:
    """Train one run, then evaluate the best dev checkpoint on test.

    Args:
        config: Run configuration.
        data_root: Directory holding (or receiving) the SNIPS splits.
        project: wandb project.
        log_root: Directory for checkpoints and logs.
        num_devices: Number of GPUs.
        num_workers: Data loading workers.
        tag: Short name for this run, used in paths and the wandb run name.

    Returns:
        Dev sentence accuracy and test metrics of the best dev checkpoint.
    """
    seed_everything(config.seed, workers=True)

    git_hash = Repo().head.object.hexsha[:7]
    run_name = f"{config.base_model.split('/')[-1]}-{tag}-{git_hash}"
    experiment_path = log_root / run_name
    experiment_path.mkdir(exist_ok=True, parents=True)

    download_snips(data_root)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    datasets = {
        split: SnipsDataset(data_root, split, tokenizer, config.max_length)
        for split in ("train", "dev", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
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
        devices=num_devices,
        max_epochs=config.max_epochs,
        logger=WandbLogger(
            project=project,
            name=run_name,
            save_dir=experiment_path,
            config=config.model_dump(),
        ),
        callbacks=[
            checkpoint,
            EarlyStopping(
                "val/sentence_acc",
                mode="max",
                patience=config.early_stopping_patience,
            ),
        ],
    )
    trainer.fit(module, loaders["train"], loaders["dev"])
    dev_metrics = trainer.validate(module, loaders["dev"], ckpt_path="best")[0]
    test_metrics = trainer.test(module, loaders["test"], ckpt_path="best")[0]
    wandb.finish()
    return {"dev/sentence_acc": dev_metrics["val/sentence_acc"], **test_metrics}


if __name__ == "__main__":
    main()
