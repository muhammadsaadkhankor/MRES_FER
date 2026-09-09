from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_mmew_clip
from mres_fer.data.dataset import IGNORE_INDEX
from mres_fer.data.mmew import (
    build_records,
    label_maps,
    normalise_emotion,
    read_annotations,
    rebase_indices,
)


def test_normalise_emotion_maps_aliases() -> None:
    assert normalise_emotion("Happy") == "happiness"
    assert normalise_emotion(" Surprise ") == "surprise"


def test_label_maps_are_alphabetical() -> None:
    maps = label_maps()
    assert maps["micro"]["anger"] == 0
    assert maps["micro"]["repression"] == 4
    assert len(maps["micro"]) == 7
    assert len(maps["macro"]) == 6


def test_rebase_indices_shifts_absolute_frame_numbers() -> None:
    assert rebase_indices(100, 112, 130, 40) == (0, 12, 30)


def test_rebase_indices_keeps_clip_relative_values() -> None:
    assert rebase_indices(0, 5, 9, 10) == (0, 5, 9)


def test_rebase_indices_clamps_to_clip_length() -> None:
    assert rebase_indices(0, 99, 120, 10) == (0, 9, 9)


def test_build_records_micro_carries_both_labels(mmew_root: Path) -> None:
    records = {r.clip_id: r for r in build_records(mmew_root, "Micro_Expression", "micro")}
    happy = records["micro_S03-01-002"]
    assert happy.micro_label == label_maps()["micro"]["happiness"]
    assert happy.macro_label == label_maps()["macro"]["happiness"]
    assert happy.subject == "S03"
    assert happy.frames_dir == "Micro_Expression/happiness/S03-01-002"


def test_build_records_micro_only_class_ignores_macro_head(mmew_root: Path) -> None:
    records = {r.clip_id: r for r in build_records(mmew_root, "Micro_Expression", "micro")}
    assert records["micro_S05-07-001"].macro_label == IGNORE_INDEX


def test_build_records_macro_ignores_micro_head_and_nested_subject_dirs(mmew_root: Path) -> None:
    records = build_records(mmew_root, "Macro_Expression", "macro")
    assert [r.micro_label for r in records] == [IGNORE_INDEX]
    assert records[0].macro_label == label_maps()["macro"]["anger"]
    assert records[0].subject == "S03"


def test_build_records_rejects_unknown_emotion_directories(mmew_root: Path) -> None:
    write_mmew_clip(mmew_root / "Micro_Expression" / "confusion" / "S09-09-001")
    with pytest.raises(ValueError, match="unrecognised emotion"):
        build_records(mmew_root, "Micro_Expression", "micro")
    kept = build_records(mmew_root, "Micro_Expression", "micro", skip_unknown=True)
    assert all("S09" not in r.clip_id for r in kept)


def test_build_records_applies_annotations(mmew_root: Path, tmp_path: Path) -> None:
    table = tmp_path / "micro.csv"
    table.write_text(
        "Subject,Filename,OnsetFrame,ApexFrame,OffsetFrame\nS03,S03-01-002,100,103,105\n"
    )
    records = {
        r.clip_id: r
        for r in build_records(mmew_root, "Micro_Expression", "micro", annotations=table)
    }
    assert (records["micro_S03-01-002"].onset, records["micro_S03-01-002"].apex) == (0, 3)
    assert records["micro_S05-07-001"].apex is None


def test_read_annotations_normalises_headers(tmp_path: Path) -> None:
    table = tmp_path / "micro.csv"
    table.write_text(
        "subject ,File Name,Onset Frame,Apex Frame,Offset Frame\nS03,S03-01-002,1,4,9\n"
    )
    parsed = read_annotations(table)["S03-01-002"]
    assert (parsed.subject, parsed.onset, parsed.apex, parsed.offset) == ("S03", 1, 4, 9)
