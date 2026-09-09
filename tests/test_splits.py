from __future__ import annotations

import pytest

from mres_fer.data.dataset import ClipRecord
from mres_fer.data.splits import group_by_subject, loso_folds, subject_holdout


def _records() -> list[ClipRecord]:
    return [
        ClipRecord(clip_id="a", frames_dir="a", macro_label=0, subject="S01"),
        ClipRecord(clip_id="b", frames_dir="b", macro_label=1, subject="S01"),
        ClipRecord(clip_id="c", frames_dir="c", macro_label=1, subject="S02"),
        ClipRecord(clip_id="d", frames_dir="d", macro_label=0, subject="S03"),
    ]


def test_group_by_subject() -> None:
    groups = group_by_subject(_records())
    assert sorted(groups) == ["S01", "S02", "S03"]
    assert [r.clip_id for r in groups["S01"]] == ["a", "b"]


def test_group_by_subject_requires_subject() -> None:
    with pytest.raises(ValueError, match="no subject"):
        group_by_subject([ClipRecord(clip_id="a", frames_dir="a", macro_label=0)])


def test_loso_folds_hold_out_one_subject_each() -> None:
    folds = loso_folds(_records())
    assert [f.subject for f in folds] == ["S01", "S02", "S03"]
    for fold in folds:
        assert {r.subject for r in fold.val} == {fold.subject}
        assert fold.subject not in {r.subject for r in fold.train}
        assert len(fold.train) + len(fold.val) == 4


def test_loso_folds_can_be_restricted() -> None:
    folds = loso_folds(_records(), ["S02"])
    assert len(folds) == 1 and folds[0].subject == "S02"


def test_subject_holdout_keeps_the_split_subject_disjoint() -> None:
    train, val = subject_holdout(_records(), 0.34)
    assert len(train) + len(val) == 4
    assert val and not {r.subject for r in train} & {r.subject for r in val}


def test_subject_holdout_rejects_a_degenerate_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        subject_holdout(_records(), 1.0)


def test_loso_folds_reject_unknown_subject() -> None:
    with pytest.raises(ValueError, match="unknown subjects"):
        loso_folds(_records(), ["S09"])
