from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_mmew_clip
from mres_fer.data.dataset import IGNORE_INDEX
from mres_fer.data.mmew import (
    build_records,
    discover_emotions,
    label_maps,
    normalise_emotion,
    read_annotations,
    rebase_indices,
)


def _maps(root: Path) -> dict[str, dict[str, int]]:
    return label_maps(
        discover_emotions(root, "Micro_Expression"), discover_emotions(root, "Macro_Expression")
    )


def test_normalise_emotion_maps_aliases() -> None:
    assert normalise_emotion("Happy") == "happiness"
    assert normalise_emotion(" Surprise ") == "surprise"


def test_discover_emotions_reads_taxonomy_off_disk(mmew_root: Path) -> None:
    assert discover_emotions(mmew_root, "Micro_Expression") == ("anger", "happiness", "repression")
    assert discover_emotions(mmew_root, "Macro_Expression") == ("anger", "happiness")


def test_discover_emotions_rejects_unknown_directories(mmew_root: Path) -> None:
    write_mmew_clip(mmew_root / "Micro_Expression" / "confusion" / "S09-09-001")
    with pytest.raises(ValueError, match="unrecognised emotion"):
        discover_emotions(mmew_root, "Micro_Expression")
    assert "confusion" not in discover_emotions(mmew_root, "Micro_Expression", skip_unknown=True)


def test_label_maps_are_alphabetical() -> None:
    maps = label_maps(("surprise", "anger", "repression"), ("surprise", "anger"))
    assert maps["micro"] == {"anger": 0, "repression": 1, "surprise": 2}
    assert maps["macro"] == {"anger": 0, "surprise": 1}


def test_label_maps_without_macro_subset_drop_micro_only_classes() -> None:
    maps = label_maps(("anger", "repression"))
    assert maps["macro"] == {"anger": 0}


def test_rebase_indices_shifts_absolute_frame_numbers() -> None:
    assert rebase_indices(100, 112, 130, 40) == (0, 12, 30)


def test_rebase_indices_keeps_clip_relative_values() -> None:
    assert rebase_indices(0, 5, 9, 10) == (0, 5, 9)


def test_rebase_indices_clamps_to_clip_length() -> None:
    assert rebase_indices(0, 99, 120, 10) == (0, 9, 9)


def test_build_records_micro_carries_both_labels(mmew_root: Path) -> None:
    maps = _maps(mmew_root)
    records = {r.clip_id: r for r in build_records(mmew_root, "Micro_Expression", "micro", maps)}
    happy = records["micro_S03-01-002"]
    assert happy.micro_label == maps["micro"]["happiness"]
    assert happy.macro_label == maps["macro"]["happiness"]
    assert happy.subject == "S03"
    assert happy.frames_dir == "Micro_Expression/happiness/S03-01-002"


def test_build_records_micro_only_class_ignores_macro_head(mmew_root: Path) -> None:
    maps = _maps(mmew_root)
    records = {r.clip_id: r for r in build_records(mmew_root, "Micro_Expression", "micro", maps)}
    assert records["micro_S05-07-001"].macro_label == IGNORE_INDEX


def test_build_records_macro_ignores_micro_head_and_reads_stills_as_clips(
    mmew_root: Path,
) -> None:
    maps = _maps(mmew_root)
    records = {r.clip_id: r for r in build_records(mmew_root, "Macro_Expression", "macro", maps)}
    assert {r.micro_label for r in records.values()} == {IGNORE_INDEX}
    still = records["macro_S03-07-001"]
    assert still.frames_dir == "Macro_Expression/S03/anger/S03-07-001.jpg"
    assert (still.macro_label, still.subject) == (maps["macro"]["anger"], "S03")
    sequence = records["macro_S04-05-001"]
    assert sequence.frames_dir == "Macro_Expression/S04/happiness/S04-05-001"


def test_build_records_rejects_emotions_missing_from_the_maps(mmew_root: Path) -> None:
    maps = _maps(mmew_root)
    write_mmew_clip(mmew_root / "Micro_Expression" / "confusion" / "S09-09-001")
    with pytest.raises(ValueError, match="unrecognised emotion"):
        build_records(mmew_root, "Micro_Expression", "micro", maps)
    kept = build_records(mmew_root, "Micro_Expression", "micro", maps, skip_unknown=True)
    assert all("S09" not in r.clip_id for r in kept)


def test_build_records_applies_annotations(mmew_root: Path, tmp_path: Path) -> None:
    table = tmp_path / "micro.csv"
    table.write_text(
        "Subject,Filename,OnsetFrame,ApexFrame,OffsetFrame\nS03,S03-01-002,100,103,105\n"
    )
    records = {
        r.clip_id: r for r in build_records(mmew_root, "Micro_Expression", "micro", None, table)
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
