"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from mres_fer.config import load_config
from mres_fer.data.dataset import build_dataloaders
from mres_fer.engine.trainer import Trainer, resolve_device
from mres_fer.models.mres_fer import build_model


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"override must look like section.key=value, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config, _parse_overrides(args.override))
    train_loader, val_loader = build_dataloaders(
        config.data, config.optim.batch_size, with_flow=config.model.use_motion
    )
    metrics = Trainer(config, train_loader, val_loader).fit()
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config, _parse_overrides(args.override))
    _, val_loader = build_dataloaders(
        config.data, config.optim.batch_size, with_flow=config.model.use_motion
    )
    trainer = Trainer(config, val_loader, val_loader)
    checkpoint = torch.load(args.checkpoint, map_location=resolve_device(config.run.device))
    trainer.model.load_state_dict(checkpoint["model"])
    print(json.dumps(trainer.evaluate(), indent=2))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Instantiate the model and report shapes and parameter counts."""
    config = load_config(args.config, _parse_overrides(args.override))
    model = build_model(config.model, config.data.image_size, config.magnification)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    size = config.data.image_size
    frames = torch.rand(1, config.data.num_frames, 3, size, size)
    flow = (
        torch.zeros(1, config.data.num_frames, 2, size, size) if config.model.use_motion else None
    )
    model.eval()
    with torch.no_grad():
        output = model(frames, flow)
    print(f"micro_logits: {tuple(output.micro_logits.shape)}")
    print(f"macro_logits: {tuple(output.macro_logits.shape)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mres-fer", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, required=True, help="path to a YAML config")
        sub.add_argument(
            "--override",
            action="append",
            default=[],
            metavar="SECTION.KEY=VALUE",
            help="override a config entry, repeatable",
        )

    train = subparsers.add_parser("train", help="train the model")
    add_common(train)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint")
    add_common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    summary = subparsers.add_parser("summary", help="print model shapes and parameter counts")
    add_common(summary)
    summary.set_defaults(func=cmd_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
