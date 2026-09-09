"""Training and evaluation loop."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mres_fer.config import Config
from mres_fer.engine.losses import MicroMacroLoss
from mres_fer.engine.metrics import ClassificationTracker
from mres_fer.models.mres_fer import MresFer, build_model
from mres_fer.utils.logging import JsonlWriter, setup_logging
from mres_fer.utils.seed import set_seed

Batch = dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_optimizer(model: MresFer, config: Config) -> AdamW:
    """Give the pretrained appearance backbone a smaller learning rate."""
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("appearance.") else head_params).append(param)
    groups = [{"params": head_params, "lr": config.optim.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": config.optim.backbone_lr})
    return AdamW(groups, lr=config.optim.lr, weight_decay=config.optim.weight_decay)


def build_scheduler(optimizer: AdamW, config: Config, steps_per_epoch: int) -> LambdaLR:
    """Linear warmup followed by cosine decay, stepped per iteration."""
    warmup_steps = max(1, config.optim.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, config.optim.epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


class Trainer:
    def __init__(
        self,
        config: Config,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        model: MresFer | None = None,
    ) -> None:
        set_seed(config.run.seed)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device(config.run.device)
        self.output_dir = Path(config.run.output_dir)
        self.logger = setup_logging(self.output_dir)
        self.history = JsonlWriter(self.output_dir / "metrics.jsonl")

        self.model = (
            model
            if model is not None
            else build_model(config.model, config.data.image_size, config.magnification)
        )
        self.model.to(self.device)
        self.criterion = MicroMacroLoss(config.loss).to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config, max(1, len(train_loader)))
        self.amp = config.optim.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.best_score = -math.inf

        (self.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    def _to_device(self, batch: Batch) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        frames = batch["frames"].to(self.device, non_blocking=True)
        flow = batch["flow"].to(self.device, non_blocking=True) if "flow" in batch else None
        micro = batch["micro_label"].to(self.device, non_blocking=True)
        macro = batch["macro_label"].to(self.device, non_blocking=True)
        return frames, flow, micro, macro

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "micro_loss": 0.0, "macro_loss": 0.0}
        seen = 0
        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for step, batch in enumerate(progress):
            frames, flow, micro, macro = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(frames, flow)
                losses = self.criterion(output, micro, macro)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["loss"]).backward()
            if self.config.optim.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            batch_size = frames.shape[0]
            seen += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_size
            if step % self.config.run.log_interval == 0:
                progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")

        return {key: value / max(1, seen) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        micro_tracker = ClassificationTracker(
            self.config.model.num_micro_classes, self.config.loss.ignore_index
        )
        macro_tracker = ClassificationTracker(
            self.config.model.num_macro_classes, self.config.loss.ignore_index
        )
        total_loss, seen = 0.0, 0
        for batch in tqdm(self.val_loader, desc="val", leave=False):
            frames, flow, micro, macro = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                output = self.model(frames, flow)
                losses = self.criterion(output, micro, macro)
            micro_tracker.update(output.micro_logits.float(), micro)
            macro_tracker.update(output.macro_logits.float(), macro)
            total_loss += float(losses["loss"]) * frames.shape[0]
            seen += frames.shape[0]

        metrics = {"val_loss": total_loss / max(1, seen)}
        metrics.update({f"micro_{k}": v for k, v in micro_tracker.compute().items()})
        metrics.update({f"macro_{k}": v for k, v in macro_tracker.compute().items()})
        return metrics

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "metrics": metrics,
            },
            path,
        )
        return path

    def fit(self) -> dict[str, float]:
        monitor = self.config.run.monitor
        last_metrics: dict[str, float] = {}
        for epoch in range(1, self.config.optim.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            last_metrics = {**train_metrics, **val_metrics}

            if monitor not in val_metrics:
                raise KeyError(f"monitor '{monitor}' not in metrics {sorted(val_metrics)}")
            score = val_metrics[monitor]
            record = {"epoch": epoch, **last_metrics}
            self.history.write(record)
            self.logger.info(
                "epoch %d | loss %.4f | val_loss %.4f | macro_uf1 %.4f | micro_uf1 %.4f",
                epoch,
                train_metrics["loss"],
                val_metrics["val_loss"],
                val_metrics["macro_uf1"],
                val_metrics["micro_uf1"],
            )

            self.save_checkpoint("last.pt", epoch, last_metrics)
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint("best.pt", epoch, last_metrics)
                self.logger.info("new best %s = %.4f", monitor, score)
        return last_metrics
