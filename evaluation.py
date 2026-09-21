"""Step 3 - evaluate a checkpoint on the held-out subjects.

    python evaluation.py --config configs/default.yaml --checkpoint workdir/checkpoints/last.pt

Writes results/metrics_<split>.json, results/predictions_<split>.csv and
results/confusion_matrix_<split>.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.datasets import MacroClipDataset, read_splits  # noqa: E402
from mres_fer.models import MicroGuidedFER  # noqa: E402
from mres_fer.utils import get_logger, resolve_device, save_json  # noqa: E402


def load_model(checkpoint_path: Path, cfg, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", list(cfg.labels["macro_classes"]))
    model = MicroGuidedFER(checkpoint.get("config", cfg), num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes, checkpoint


@torch.no_grad()
def infer(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | list]:
    logits, labels, clip_ids, subjects, embeddings = [], [], [], [], []
    for batch in loader:
        output = model(batch["frames"].to(device), batch["windows"].to(device))
        logits.append(output["logits"].cpu())
        embeddings.append(output["embedding"].cpu())
        labels.append(batch["label"])
        clip_ids += list(batch["clip_id"])
        subjects += list(batch["subject"])
    logits_t = torch.cat(logits)
    return {
        "logits": logits_t.numpy(),
        "probs": torch.softmax(logits_t, dim=1).numpy(),
        "preds": logits_t.argmax(1).numpy(),
        "labels": torch.cat(labels).numpy(),
        "embeddings": torch.cat(embeddings).numpy(),
        "clip_ids": clip_ids,
        "subjects": subjects,
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="", help="defaults to <work_root>/checkpoints/last.pt")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("eval", workspace.logs / "evaluation.log")
    device = resolve_device(cfg.train["device"])

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else workspace.checkpoints / "last.pt"
    model, classes, _ = load_model(checkpoint_path, cfg, device)
    class_to_idx = {name: i for i, name in enumerate(classes)}

    splits = read_splits(workspace.splits)
    dataset = MacroClipDataset(
        workspace.macro_index, class_to_idx, cfg.sampling, splits[args.split], train=False
    )
    loader = DataLoader(dataset, batch_size=cfg.train["batch_size"], num_workers=cfg.train["num_workers"])
    logger.info("evaluating %s on %d clips (%d subjects)", checkpoint_path, len(dataset), len(splits[args.split]))

    out = infer(model, loader, device)
    labels, preds = out["labels"], out["preds"]
    present = sorted(set(labels.tolist()) | set(preds.tolist()))

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_clips": int(len(labels)),
        "subjects": splits[args.split],
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "per_class": classification_report(
            labels,
            preds,
            labels=present,
            target_names=[classes[i] for i in present],
            output_dict=True,
            zero_division=0,
        ),
    }
    save_json(workspace.results / f"metrics_{args.split}.json", metrics)

    pd.DataFrame(
        {
            "clip_id": out["clip_ids"],
            "subject": out["subjects"],
            "true": [classes[i] for i in labels],
            "pred": [classes[i] for i in preds],
            "confidence": out["probs"].max(axis=1),
        }
    ).to_csv(workspace.results / f"predictions_{args.split}.csv", index=False)

    matrix = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        workspace.results / f"confusion_matrix_{args.split}.csv"
    )

    if args.save_embeddings:
        np.savez(
            workspace.results / f"embeddings_{args.split}.npz",
            embeddings=out["embeddings"],
            labels=labels,
            preds=preds,
            clip_ids=np.array(out["clip_ids"]),
        )

    logger.info(
        "accuracy %.4f | macro-F1 %.4f | results in %s",
        metrics["accuracy"],
        metrics["macro_f1"],
        workspace.results,
    )


if __name__ == "__main__":
    main()
