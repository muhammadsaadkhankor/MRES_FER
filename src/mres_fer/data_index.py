"""Scans the MMEW_Final folder tree and turns it into clip records.

Expected layout (as shipped by the user's dataset):

    MMEW_Final/
      Macro_Expression/<subject>/<emotion>/<subject>-<session>-<frame>.jpg
      Micro_Expression/<emotion>/<clip_id>/<frame_number>.jpg

Macro frames of one subject/emotion folder may belong to several recordings, so
frames are grouped by their filename prefix (everything before the last "-NNN").
Micro clips already live in their own folder, frames are numbered 1.jpg, 2.jpg...
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_FRAME_SUFFIX = re.compile(r"^(?P<stem>.+)[-_](?P<index>\d+)$")
_LEADING_NUMBER = re.compile(r"(\d+)")


@dataclass
class Clip:
    clip_id: str
    subject: str
    label: str
    domain: str  # "macro" or "micro"
    frames: list[Path] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def _numeric_key(path: Path) -> tuple:
    """Sorts 1.jpg, 2.jpg, 10.jpg naturally instead of lexicographically."""
    parts = _LEADING_NUMBER.split(path.stem)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _images(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=_numeric_key,
    )


def _subject_from_clip_id(clip_id: str) -> str:
    return clip_id.split("-")[0].split("_")[0]


def scan_macro(macro_root: Path, classes: list[str]) -> list[Clip]:
    """Macro_Expression/<subject>/<emotion>/frames -> one clip per filename prefix."""
    clips: list[Clip] = []
    for subject_dir in sorted(p for p in macro_root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            label = label_dir.name.lower()
            if classes and label not in classes:
                continue
            groups: dict[str, list[Path]] = defaultdict(list)
            for frame in _images(label_dir):
                match = _FRAME_SUFFIX.match(frame.stem)
                key = match.group("stem") if match else frame.stem
                groups[key].append(frame)
            for key, frames in sorted(groups.items()):
                clip_id = f"{subject_dir.name}__{label}__{key}"
                clips.append(
                    Clip(
                        clip_id=clip_id,
                        subject=subject_dir.name,
                        label=label,
                        domain="macro",
                        frames=sorted(frames, key=_numeric_key),
                    )
                )
    return clips


def scan_micro(micro_root: Path, ignore: list[str] | None = None) -> list[Clip]:
    """Micro_Expression/<emotion>/<clip_id>/frames -> one clip per folder."""
    ignore = {name.lower() for name in (ignore or [])}
    clips: list[Clip] = []
    for label_dir in sorted(p for p in micro_root.iterdir() if p.is_dir()):
        label = label_dir.name.lower()
        if label in ignore:
            continue
        for clip_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            frames = _images(clip_dir)
            if not frames:
                continue
            clips.append(
                Clip(
                    clip_id=clip_dir.name,
                    subject=_subject_from_clip_id(clip_dir.name),
                    label=label,
                    domain="micro",
                    frames=frames,
                )
            )
    return clips


def load_annotations(path: str | Path) -> dict[str, dict[str, int]]:
    """Reads the MMEW annotation sheet -> {clip_id: {onset, apex, offset}}.

    Accepts .csv/.xlsx with columns Filename, OnsetFrame, ApexFrame, OffsetFrame
    (column names are matched case/space insensitively).
    """
    import pandas as pd

    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    normalised = {str(c).strip().lower().replace(" ", ""): c for c in frame.columns}

    def column(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in normalised:
                return normalised[candidate]
        return None

    name_col = column("filename", "clip", "clipid", "name")
    if name_col is None:
        raise ValueError(f"{path} has no Filename-like column (found {list(frame.columns)})")
    onset_col = column("onsetframe", "onset")
    apex_col = column("apexframe", "apex")
    offset_col = column("offsetframe", "offset")

    annotations: dict[str, dict[str, int]] = {}
    for _, row in frame.iterrows():
        clip_id = str(row[name_col]).strip()
        if not clip_id or clip_id.lower() == "nan":
            continue

        def as_int(col: str | None) -> int:
            if col is None:
                return -1
            try:
                return int(row[col])
            except (TypeError, ValueError):
                return -1

        annotations[clip_id] = {
            "onset": as_int(onset_col),
            "apex": as_int(apex_col),
            "offset": as_int(offset_col),
        }
    return annotations
