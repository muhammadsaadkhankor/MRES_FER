"""Evaluation metrics.

UF1 (unweighted F1) and UAR (unweighted average recall) are the standard
micro-expression benchmarks; they weight every class equally, which matters because the
class distribution of CASME II / SAMM / MEGC is heavily skewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


def _confusion(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, preds), 1)
    return matrix


def unweighted_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    matrix = _confusion(preds, targets, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    present = actual > 0
    if not present.any():
        return 0.0
    denominator = predicted + actual
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(f1[present].mean())


def unweighted_average_recall(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    matrix = _confusion(preds, targets, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    present = actual > 0
    if not present.any():
        return 0.0
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    return float(recall[present].mean())


@dataclass
class ClassificationTracker:
    """Accumulates predictions of one head across a validation epoch."""

    num_classes: int
    ignore_index: int = -100
    _preds: list[np.ndarray] = field(default_factory=list)
    _targets: list[np.ndarray] = field(default_factory=list)

    def update(self, logits: Tensor, targets: Tensor) -> None:
        keep = targets != self.ignore_index
        if not bool(torch.any(keep)):
            return
        self._preds.append(logits[keep].argmax(dim=-1).detach().cpu().numpy())
        self._targets.append(targets[keep].detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        if not self._preds:
            return {"accuracy": 0.0, "uf1": 0.0, "uar": 0.0, "count": 0.0}
        preds = np.concatenate(self._preds)
        targets = np.concatenate(self._targets)
        return {
            "accuracy": float((preds == targets).mean()),
            "uf1": unweighted_f1(preds, targets, self.num_classes),
            "uar": unweighted_average_recall(preds, targets, self.num_classes),
            "count": float(preds.size),
        }

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()
