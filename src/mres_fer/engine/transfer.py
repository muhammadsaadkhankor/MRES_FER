"""Two-stage micro-to-macro transfer.

The hypothesis under test is that a representation learned for micro-expression
recognition carries fine-grained evidence that helps macro-expression recognition, which
matters most when micro annotations are scarce. Stage 1 therefore trains the encoder and
the micro head on the micro-annotated clips only; stage 2 reloads that representation and
fits the macro head on the macro clips, freezing as much of the encoder as
:class:`~mres_fer.config.TransferConfig` asks for.

With ``freeze: encoder`` the macro head is the only thing that moves in stage 2, so the
macro accuracy it reaches is attributable to the micro-derived features alone; the joint
and macro-only configs are the baselines it should be compared against.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from mres_fer.config import Config
from mres_fer.data.dataset import ClipRecord, build_dataloaders_from_records, read_manifest
from mres_fer.engine.trainer import Trainer
from mres_fer.models.mres_fer import MresFer, build_model

FREEZE_POLICIES = ("encoder", "appearance", "none")


def apply_freeze(model: MresFer, policy: str) -> int:
    """Freeze the micro-pretrained parts; returns the number of frozen parameters."""
    if policy not in FREEZE_POLICIES:
        raise ValueError(f"unknown freeze policy '{policy}', expected one of {FREEZE_POLICIES}")
    frozen = 0
    for name, param in model.named_parameters():
        if policy == "none":
            keep = True
        elif policy == "appearance":
            keep = not name.startswith("appearance.")
        else:  # encoder: only the macro head keeps learning
            keep = name.startswith("macro_head.")
        param.requires_grad_(keep)
        frozen += 0 if keep else param.numel()
    return frozen


def _subset(records: Sequence[ClipRecord], field: str, ignore_index: int) -> list[ClipRecord]:
    selected = [r for r in records if getattr(r, field) != ignore_index]
    if not selected:
        raise ValueError(f"no clips with a {field.split('_')[0]} label in the manifest")
    return selected


def _stage_config(
    config: Config,
    name: str,
    epochs: int,
    monitor: str,
    micro_weight: float,
    macro_weight: float,
) -> Config:
    return replace(
        config,
        loss=replace(config.loss, micro_weight=micro_weight, macro_weight=macro_weight),
        optim=replace(config.optim, epochs=epochs),
        run=replace(
            config.run, output_dir=str(Path(config.run.output_dir) / name), monitor=monitor
        ),
    )


def run_transfer(
    config: Config,
    train_records: Sequence[ClipRecord] | None = None,
    val_records: Sequence[ClipRecord] | None = None,
) -> dict[str, dict[str, float]]:
    """Run both stages and return their final metrics keyed by stage.

    Records default to the configured manifests; LOSO passes its fold instead.
    """
    ignore = config.loss.ignore_index
    if train_records is None:
        train_records = read_manifest(config.data.train_manifest)
    if val_records is None:
        val_records = read_manifest(config.data.val_manifest)

    model = build_model(config.model, config.data.image_size, config.magnification)

    stage1 = _stage_config(
        config,
        "stage1_micro",
        config.transfer.stage1_epochs,
        "micro_uf1",
        micro_weight=config.loss.micro_weight,
        macro_weight=0.0,
    )
    trainer = Trainer(
        stage1,
        *build_dataloaders_from_records(
            config.data,
            _subset(train_records, "micro_label", ignore),
            _subset(val_records, "micro_label", ignore),
            config.optim.batch_size,
            with_flow=config.model.use_motion,
        ),
        model=model,
    )
    micro_metrics = trainer.fit()

    # Start stage 2 from the best micro epoch rather than the last one.
    best = trainer.output_dir / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location="cpu")["model"])
    frozen = apply_freeze(model, config.transfer.freeze)

    stage2 = _stage_config(
        config,
        "stage2_macro",
        config.transfer.stage2_epochs,
        "macro_uf1",
        micro_weight=config.loss.micro_weight if config.transfer.keep_micro_loss else 0.0,
        macro_weight=config.loss.macro_weight,
    )
    macro_trainer = Trainer(
        stage2,
        *build_dataloaders_from_records(
            config.data,
            _subset(train_records, "macro_label", ignore),
            _subset(val_records, "macro_label", ignore),
            config.optim.batch_size,
            with_flow=config.model.use_motion,
        ),
        model=model,
    )
    macro_trainer.logger.info("stage 2: %.2fM parameters frozen", frozen / 1e6)
    macro_metrics = macro_trainer.fit()

    summary = {"stage1_micro": micro_metrics, "stage2_macro": macro_metrics}
    root = Path(config.run.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
