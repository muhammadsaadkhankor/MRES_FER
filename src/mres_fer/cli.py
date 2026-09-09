"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from mres_fer.config import load_config
from mres_fer.data import mmew
from mres_fer.data.dataset import (
    build_dataloaders,
    build_dataloaders_from_records,
    read_manifest,
    write_manifest,
)
from mres_fer.data.splits import loso_folds, subject_holdout
from mres_fer.engine.trainer import Trainer, resolve_device
from mres_fer.engine.transfer import run_transfer
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


def cmd_transfer(args: argparse.Namespace) -> int:
    """Pretrain on the micro clips, then fit the macro head on the macro clips."""
    config = load_config(args.config, _parse_overrides(args.override))
    print(json.dumps(run_transfer(config), indent=2))
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


def _find_annotations(root: Path, keyword: str) -> Path | None:
    """MMEW ships its label table at the dataset root, e.g. ``MMEW_Micro_Exp.xlsx``."""
    candidates = [
        path
        for path in sorted(root.glob("*"))
        if path.suffix.lower() in {".xlsx", ".xlsm", ".csv"} and keyword in path.stem.lower()
    ]
    return candidates[0] if candidates else None


def cmd_prepare_mmew(args: argparse.Namespace) -> int:
    """Scan an MMEW release into manifests (micro clips also carry their macro label)."""
    root = Path(args.root)
    micro_emotions = mmew.discover_emotions(root, args.micro_dir, args.skip_unknown)
    macro_emotions = (
        () if args.micro_only else mmew.discover_emotions(root, args.macro_dir, args.skip_unknown)
    )
    maps = mmew.label_maps(micro_emotions, macro_emotions)

    micro_table = args.micro_annotations or _find_annotations(root, "micro")
    records = mmew.build_records(
        root, args.micro_dir, "micro", maps, micro_table, args.skip_unknown
    )
    micro_count = len(records)
    if not args.micro_only:
        macro_table = args.macro_annotations or _find_annotations(root, "macro")
        records += mmew.build_records(
            root, args.macro_dir, "macro", maps, macro_table, args.skip_unknown
        )

    out = Path(args.out)
    write_manifest(out / "manifest.json", records)
    # Single runs need a subject-disjoint split; training and validating on the same
    # manifest would only measure memorisation.
    train_records, val_records = subject_holdout(records, args.val_fraction)
    write_manifest(out / "train.json", train_records)
    write_manifest(out / "val.json", val_records)
    (out / "labels.json").write_text(json.dumps(maps, indent=2))
    subjects = sorted({r.subject for r in records if r.subject is not None})
    val_subjects = sorted({r.subject for r in val_records if r.subject is not None})
    print(
        json.dumps(
            {
                "manifest": str(out / "manifest.json"),
                "train_manifest": str(out / "train.json"),
                "val_manifest": str(out / "val.json"),
                "clips": len(records),
                "micro_clips": micro_count,
                "macro_clips": len(records) - micro_count,
                "subjects": len(subjects),
                "val_subjects": val_subjects,
                "micro_annotations": str(micro_table) if micro_table else None,
                "model.num_micro_classes": len(maps["micro"]),
                "model.num_macro_classes": len(maps["macro"]),
            },
            indent=2,
        )
    )
    return 0


def cmd_loso(args: argparse.Namespace) -> int:
    """Leave-one-subject-out cross-validation over a single manifest."""
    config = load_config(args.config, _parse_overrides(args.override))
    records = read_manifest(args.manifest)
    folds = loso_folds(records, args.subject or None)
    root = Path(config.run.output_dir)

    per_fold: dict[str, dict[str, float]] = {}
    for fold in folds:
        fold_config = load_config(args.config, _parse_overrides(args.override))
        fold_config.run.output_dir = str(root / f"fold_{fold.subject}")
        if args.transfer:
            per_fold[fold.subject] = run_transfer(fold_config, fold.train, fold.val)["stage2_macro"]
            continue
        train_loader, val_loader = build_dataloaders_from_records(
            fold_config.data,
            fold.train,
            fold.val,
            fold_config.optim.batch_size,
            with_flow=fold_config.model.use_motion,
        )
        per_fold[fold.subject] = Trainer(fold_config, train_loader, val_loader).fit()

    keys = sorted({key for metrics in per_fold.values() for key in metrics})
    mean = {key: mean_of(per_fold, key) for key in keys}
    summary = {"folds": per_fold, "mean": mean}
    root.mkdir(parents=True, exist_ok=True)
    (root / "loso_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def mean_of(per_fold: dict[str, dict[str, float]], key: str) -> float:
    values = [metrics[key] for metrics in per_fold.values() if key in metrics]
    return sum(values) / len(values) if values else float("nan")


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

    transfer = subparsers.add_parser(
        "transfer", help="two-stage micro pretraining then macro fine-tuning"
    )
    add_common(transfer)
    transfer.set_defaults(func=cmd_transfer)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint")
    add_common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    summary = subparsers.add_parser("summary", help="print model shapes and parameter counts")
    add_common(summary)
    summary.set_defaults(func=cmd_summary)

    prepare = subparsers.add_parser("prepare-mmew", help="build manifests from an MMEW release")
    prepare.add_argument("--root", type=Path, required=True, help="MMEW dataset root")
    prepare.add_argument("--out", type=Path, required=True, help="directory for the manifests")
    prepare.add_argument("--micro-dir", default="Micro_Expression")
    prepare.add_argument("--macro-dir", default="Macro_Expression")
    prepare.add_argument(
        "--micro-annotations",
        type=Path,
        help="micro .xlsx/.csv label table (default: auto-detected in --root)",
    )
    prepare.add_argument(
        "--macro-annotations",
        type=Path,
        help="macro .xlsx/.csv label table (default: auto-detected in --root)",
    )
    prepare.add_argument("--micro-only", action="store_true", help="skip the macro subset")
    prepare.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="share of subjects held out in val.json (default: 0.2)",
    )
    prepare.add_argument(
        "--skip-unknown", action="store_true", help="ignore unrecognised emotion directories"
    )
    prepare.set_defaults(func=cmd_prepare_mmew)

    loso = subparsers.add_parser("loso", help="leave-one-subject-out cross-validation")
    add_common(loso)
    loso.add_argument("--manifest", type=Path, required=True, help="manifest holding all clips")
    loso.add_argument(
        "--subject", action="append", default=[], help="restrict to these held-out subjects"
    )
    loso.add_argument(
        "--transfer",
        action="store_true",
        help="run the two-stage transfer schedule per fold and report its macro stage",
    )
    loso.set_defaults(func=cmd_loso)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
