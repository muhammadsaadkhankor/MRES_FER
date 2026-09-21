"""Optional step - subject-independent k-fold cross-validation.

    python cross_validate.py --config configs/default.yaml

Trains one model per fold (every subject is tested exactly once), evaluates each fold
on its held-out subjects and writes results/cross_validation.json with the mean +- std
accuracy. With ~30 subjects a single 6-subject test split is very high variance, so
these numbers are the ones to report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evaluation import infer, load_model  # noqa: E402
from mres_fer.config import Workspace, add_config_args, load_config  # noqa: E402
from mres_fer.datasets import MacroClipDataset, kfold_subject_splits, write_splits  # noqa: E402
from mres_fer.utils import get_logger, resolve_device, save_json  # noqa: E402
from train import run_training  # noqa: E402


def test_fold(cfg, workspace: Workspace, classes: list[str], subjects: list[str], tag: str) -> dict:
    device = resolve_device(cfg.train["device"])
    model, classes, _ = load_model(workspace.checkpoints / f"best_{tag}.pt", cfg, device)
    dataset = MacroClipDataset(
        workspace.macro_index,
        {name: i for i, name in enumerate(classes)},
        cfg.sampling,
        subjects,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=cfg.train["batch_size"], num_workers=0)
    out = infer(model, loader, device)
    return {
        "accuracy": float(accuracy_score(out["labels"], out["preds"])),
        "macro_f1": float(
            f1_score(out["labels"], out["preds"], average="macro", zero_division=0)
        ),
        "num_clips": int(len(out["labels"])),
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--folds", type=int, default=None, help="overrides split.num_folds")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    workspace = Workspace(cfg)
    workspace.create()
    logger = get_logger("cv", workspace.logs / "cross_validate.log")

    classes = list(cfg.labels["macro_classes"])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    index = MacroClipDataset(workspace.macro_index, class_to_idx, cfg.sampling, train=False)
    subjects = [record.subject for record in index.records]
    num_folds = args.folds or cfg.split["num_folds"]
    folds = kfold_subject_splits(subjects, num_folds, cfg.train["seed"])

    results = []
    for index_fold, splits in enumerate(folds):
        tag = f"fold{index_fold}"
        logger.info("=== fold %d/%d | test subjects %s", index_fold + 1, num_folds, splits["test"])
        write_splits(workspace.manifests / f"splits_{tag}.json", splits)
        summary = run_training(cfg, workspace, logger, splits, tag=tag)
        scores = test_fold(cfg, workspace, classes, splits["test"], tag)
        scores.update(fold=index_fold, val_acc=summary["best_val_acc"], test_subjects=splits["test"])
        logger.info(
            "fold %d -> test acc %.3f | macro-F1 %.3f",
            index_fold,
            scores["accuracy"],
            scores["macro_f1"],
        )
        results.append(scores)
        del summary
        torch.cuda.empty_cache()

    accuracies = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    report = {
        "num_folds": num_folds,
        "folds": results,
        "mean_accuracy": statistics.fmean(accuracies),
        "std_accuracy": statistics.pstdev(accuracies),
        "mean_macro_f1": statistics.fmean(f1s),
        "std_macro_f1": statistics.pstdev(f1s),
    }
    save_json(workspace.results / "cross_validation.json", report)
    logger.info(
        "cross-validation: accuracy %.3f +- %.3f | macro-F1 %.3f +- %.3f | %s",
        report["mean_accuracy"],
        report["std_accuracy"],
        report["mean_macro_f1"],
        report["std_macro_f1"],
        workspace.results / "cross_validation.json",
    )


if __name__ == "__main__":
    main()
