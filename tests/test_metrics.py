from __future__ import annotations

import numpy as np
import torch

from mres_fer.engine.metrics import (
    ClassificationTracker,
    unweighted_average_recall,
    unweighted_f1,
)


def test_perfect_predictions_score_one() -> None:
    targets = np.array([0, 1, 2, 2])
    assert unweighted_f1(targets, targets, 3) == 1.0
    assert unweighted_average_recall(targets, targets, 3) == 1.0


def test_absent_classes_are_excluded() -> None:
    # Class 2 never appears in the targets, so it must not drag the average down.
    preds = np.array([0, 1])
    targets = np.array([0, 1])
    assert unweighted_average_recall(preds, targets, 3) == 1.0


def test_uar_ignores_class_imbalance() -> None:
    # Always predicting the majority class gives 0.5 UAR on a 2-class problem.
    targets = np.array([0] * 9 + [1])
    preds = np.zeros_like(targets)
    assert unweighted_average_recall(preds, targets, 2) == 0.5


def test_tracker_skips_ignored_labels() -> None:
    tracker = ClassificationTracker(num_classes=2, ignore_index=-100)
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    tracker.update(logits, torch.tensor([0, -100]))
    metrics = tracker.compute()
    assert metrics["count"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_tracker_without_updates_is_zero() -> None:
    assert ClassificationTracker(num_classes=2).compute()["uf1"] == 0.0
