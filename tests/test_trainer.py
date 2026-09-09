from __future__ import annotations

import json
from pathlib import Path

import torch

from mres_fer.config import Config
from mres_fer.data.dataset import build_dataloaders
from mres_fer.engine.losses import MicroMacroLoss
from mres_fer.engine.trainer import Trainer, build_scheduler
from mres_fer.models.mres_fer import ModelOutput


def test_missing_micro_labels_do_not_contribute(tiny_config: Config) -> None:
    criterion = MicroMacroLoss(tiny_config.loss)
    output = ModelOutput(
        micro_logits=torch.randn(2, 3),
        macro_logits=torch.randn(2, 2),
        gate=None,
        clip_embedding=torch.randn(2, 4),
    )
    losses = criterion(output, torch.tensor([-100, -100]), torch.tensor([0, 1]))
    assert float(losses["micro_loss"]) == 0.0
    assert torch.allclose(losses["loss"], losses["macro_loss"])


def test_scheduler_warms_up_then_decays(tiny_config: Config) -> None:
    tiny_config.optim.epochs = 4
    tiny_config.optim.warmup_epochs = 1
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_scheduler(optimizer, tiny_config, steps_per_epoch=4)
    seen = []
    for _ in range(16):
        seen.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert seen[0] < seen[3]
    assert seen[-1] < seen[3]


def test_one_epoch_writes_checkpoints_and_metrics(tiny_config: Config) -> None:
    train_loader, val_loader = build_dataloaders(tiny_config.data, tiny_config.optim.batch_size)
    trainer = Trainer(tiny_config, train_loader, val_loader)
    metrics = trainer.fit()

    assert "macro_uf1" in metrics
    output_dir = Path(tiny_config.run.output_dir)
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "last.pt").exists()
    history = (output_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert json.loads(history[0])["epoch"] == 1
