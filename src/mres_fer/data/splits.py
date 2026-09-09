"""Subject-wise splits, the standard protocol for micro-expression benchmarks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from mres_fer.data.dataset import ClipRecord


@dataclass
class Fold:
    subject: str
    train: list[ClipRecord]
    val: list[ClipRecord]


def group_by_subject(records: Sequence[ClipRecord]) -> dict[str, list[ClipRecord]]:
    groups: dict[str, list[ClipRecord]] = OrderedDict()
    for record in records:
        if record.subject is None:
            raise ValueError(f"clip '{record.clip_id}' has no subject; LOSO needs one per clip")
        groups.setdefault(record.subject, []).append(record)
    return groups


def subject_holdout(
    records: Sequence[ClipRecord], fraction: float = 0.2
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Split into subject-disjoint train/val parts, holding out every ``1/fraction``-th subject.

    Sampling subject ids at a fixed stride rather than taking a contiguous tail keeps the
    held-out group spread across the recording order, and makes the split reproducible
    without a seed. LOSO remains the reporting protocol; this is for single runs.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    subjects = sorted(group_by_subject(records))
    if len(subjects) < 2:
        raise ValueError("a subject-wise split needs at least two subjects")
    stride = max(2, round(1 / fraction))
    held_out = set(subjects[::stride])
    train = [r for r in records if r.subject not in held_out]
    val = [r for r in records if r.subject in held_out]
    if not train:
        raise ValueError("holdout left no training clips")
    return train, val


def loso_folds(records: Sequence[ClipRecord], subjects: Sequence[str] | None = None) -> list[Fold]:
    """Leave-one-subject-out folds, ordered by subject id."""
    groups = group_by_subject(records)
    selected = sorted(groups) if subjects is None else list(subjects)
    unknown = [s for s in selected if s not in groups]
    if unknown:
        raise ValueError(f"unknown subjects: {unknown}")
    if len(groups) < 2:
        raise ValueError("LOSO needs at least two subjects")

    folds = []
    for subject in selected:
        train = [r for r in records if r.subject != subject]
        folds.append(Fold(subject=subject, train=train, val=list(groups[subject])))
    return folds
