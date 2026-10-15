"""Lightning Module."""

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn import functional
from transformers import get_linear_schedule_with_warmup

from modern_joint_bert.config import Config
from modern_joint_bert.datasets import IGNORE_INDEX, Batch, SnipsLabels
from modern_joint_bert.metrics import compute_metrics, word_level_tags
from modern_joint_bert.modeling import JointBert


class JointBertModule(LightningModule):
    """Trains JointBert with summed intent and slot cross-entropy."""

    def __init__(self, config: Config, labels: SnipsLabels) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.labels = labels
        self.model = JointBert(
            config.base_model,
            num_intents=len(labels.intents),
            num_slots=len(labels.slots),
            dropout=config.dropout,
        )
        self.intent_correct: list[bool] = []
        self.predicted_tags: list[list[str]] = []
        self.gold_tags: list[list[str]] = []

    def loss(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total loss, intent logits, and slot logits."""
        intent_logits, slot_logits = self.model(batch.input_ids, batch.attention_mask)
        intent_loss = functional.cross_entropy(intent_logits, batch.intent_label)
        slot_loss = functional.cross_entropy(
            slot_logits.flatten(0, 1),
            batch.slot_labels.flatten(),
            ignore_index=IGNORE_INDEX,
        )
        total = intent_loss + self.config.slot_loss_weight * slot_loss
        return total, intent_logits, slot_logits

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Train step."""
        loss, _, _ = self.loss(batch)
        self.log("train/loss", loss, batch_size=len(batch.input_ids))
        return loss

    def eval_step(self, batch: Batch, stage: str) -> None:
        """Accumulate predictions for epoch-level metrics."""
        loss, intent_logits, slot_logits = self.loss(batch)
        self.log(f"{stage}/loss", loss, batch_size=len(batch.input_ids))
        self.intent_correct += (intent_logits.argmax(-1) == batch.intent_label).tolist()
        predicted, gold = word_level_tags(
            slot_logits, batch.slot_labels, self.labels.slots
        )
        self.predicted_tags += predicted
        self.gold_tags += gold

    def eval_epoch_end(self, stage: str) -> None:
        """Log metrics over the whole split and reset accumulators."""
        metrics = compute_metrics(
            self.intent_correct, self.predicted_tags, self.gold_tags
        )
        self.log_dict({f"{stage}/{name}": value for name, value in metrics.items()})
        self.intent_correct, self.predicted_tags, self.gold_tags = [], [], []

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Validate step."""
        self.eval_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.eval_epoch_end("val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Test step."""
        self.eval_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        self.eval_epoch_end("test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW with linear warmup and decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.config.warmup_ratio * total_steps), total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
