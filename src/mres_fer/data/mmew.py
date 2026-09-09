"""Manifest builder for the Micro-and-Macro Expression Warehouse (MMEW).

MMEW ships micro- and macro-expression clips recorded from the same subjects, which is
exactly the pairing this project relies on: a micro clip carries both a micro label and
the macro label of the same emotion, while a macro clip only supervises the macro head.

Frames are stored as ``<index>.jpg`` inside a per-clip directory, nested under an emotion
directory (and, in some distributions, a subject directory)::

    Micro_Expression/happiness/S03-01-002/1.jpg
    Micro_Expression/S03/happiness/S03-01-002/1.jpg   # equally accepted

The scan therefore treats any directory holding image files as a clip and resolves the
emotion from the nearest matching ancestor directory, so both layouts work unchanged.

Releases differ in which emotions they ship (some carry the micro-only ``repression``,
some only the six emotions shared with the macro side), so the class vocabulary is read
off disk by :func:`discover_emotions` instead of hard-coded.

Onset/apex/offset columns of the shipped spreadsheets are absolute frame numbers of the
recording, whereas the clip directories are already trimmed. Indices are rebased onto the
clip when they fall outside it; see :func:`rebase_indices`.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mres_fer.data.dataset import IGNORE_INDEX, IMAGE_SUFFIXES, ClipRecord

#: Every emotion any MMEW release is known to ship; anything else is a directory typo.
KNOWN_EMOTIONS: frozenset[str] = frozenset(
    {
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "repression",
        "others",
        "other",
    }
)
#: Emotions MMEW only labels on the micro side, so they cannot supervise the macro head.
MICRO_ONLY_EMOTIONS: frozenset[str] = frozenset({"repression", "others", "other"})
EMOTION_ALIASES: Mapping[str, str] = {
    "angry": "anger",
    "disgusted": "disgust",
    "happy": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "surprised": "surprise",
    "surprize": "surprise",
}

_CLIP_COLUMNS = ("filename", "file", "clip", "sample", "name", "filenamenew")
_SUBJECT_COLUMNS = ("subject", "sub", "subjectname")
_ONSET_COLUMNS = ("onsetframe", "onset")
_APEX_COLUMNS = ("apexframe", "apex")
_OFFSET_COLUMNS = ("offsetframe", "offset")


@dataclass(frozen=True)
class Annotation:
    onset: int | None = None
    apex: int | None = None
    offset: int | None = None
    subject: str | None = None


def normalise_emotion(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("_", "")
    return EMOTION_ALIASES.get(key, key)


def label_maps(
    micro_emotions: Sequence[str], macro_emotions: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    """Emotion -> class index, alphabetical so runs stay comparable.

    Releases differ in which emotions they ship (some carry ``repression``, some do not),
    so the taxonomy comes from :func:`discover_emotions` rather than a hard-coded list.
    Without a macro subset the macro head reuses the micro emotions that have a macro
    counterpart.
    """
    micro = sorted({normalise_emotion(name) for name in micro_emotions})
    macro = (
        sorted({normalise_emotion(name) for name in macro_emotions})
        if macro_emotions
        else [name for name in micro if name not in MICRO_ONLY_EMOTIONS]
    )
    return {
        "micro": {name: index for index, name in enumerate(micro)},
        "macro": {name: index for index, name in enumerate(macro)},
    }


def rebase_indices(
    onset: int | None, apex: int | None, offset: int | None, num_frames: int
) -> tuple[int | None, int | None, int | None]:
    """Map spreadsheet frame numbers onto positions inside a trimmed clip directory."""
    if apex is None:
        return None, None, None
    shift = onset if onset is not None and apex >= num_frames else 0
    relative = max(0, min(apex - shift, num_frames - 1))
    lo = 0 if onset is None else max(0, min(onset - shift, relative))
    hi = num_frames - 1 if offset is None else max(relative, min(offset - shift, num_frames - 1))
    return lo, relative, hi


def _normalise_header(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _pick(row: Mapping[str, str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        value = row.get(candidate)
        if value not in (None, ""):
            return value
    return None


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        yield from _read_xlsx_rows(path)
        return
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = [_normalise_header(cell) for cell in next(reader, [])]
        for row in reader:
            yield {key: str(cell).strip() for key, cell in zip(header, row, strict=False) if key}


def _read_xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(f"reading {path.name} needs openpyxl; pip install openpyxl") from exc

    sheet = load_workbook(path, read_only=True, data_only=True).active
    if sheet is None:  # pragma: no cover - empty workbook
        return
    rows = sheet.iter_rows(values_only=True)
    header = [_normalise_header(str(cell)) if cell is not None else "" for cell in next(rows, ())]
    for row in rows:
        yield {
            key: str(cell).strip()
            for key, cell in zip(header, row, strict=False)
            if key and cell is not None and str(cell).strip() != ""
        }


def read_annotations(path: str | Path) -> dict[str, Annotation]:
    """Index a MMEW spreadsheet (``.xlsx``) or CSV export by clip name."""
    table: dict[str, Annotation] = {}
    for row in _read_rows(Path(path)):
        clip = _pick(row, _CLIP_COLUMNS)
        if clip is None:
            continue
        table[clip] = Annotation(
            onset=_as_int(_pick(row, _ONSET_COLUMNS)),
            apex=_as_int(_pick(row, _APEX_COLUMNS)),
            offset=_as_int(_pick(row, _OFFSET_COLUMNS)),
            subject=_pick(row, _SUBJECT_COLUMNS),
        )
    return table


def _clip_directories(root: Path) -> list[Path]:
    clips = [
        path
        for path in root.rglob("*")
        if path.is_dir() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir())
    ]
    return sorted(clips)


def discover_emotions(
    root: str | Path, subset_dir: str, skip_unknown: bool = False
) -> tuple[str, ...]:
    """Emotions actually present in a subset; a clip directory always sits in one.

    Releases differ (the six shared emotions, sometimes plus ``repression``), so reading
    the taxonomy off disk keeps class indices matched to the copy being trained on.
    """
    subset_root = Path(root) / subset_dir
    if not subset_root.is_dir():
        raise FileNotFoundError(f"missing MMEW subset directory {subset_root}")
    found = {normalise_emotion(clip.parent.name) for clip in _clip_directories(subset_root)}
    unknown = found - KNOWN_EMOTIONS
    if unknown and not skip_unknown:
        raise ValueError(
            f"unrecognised emotion directories under {subset_root}: {sorted(unknown)}; "
            "pass skip_unknown to ignore them"
        )
    return tuple(sorted(found & KNOWN_EMOTIONS))


def _emotion_for(clip: Path, root: Path, known: Iterable[str]) -> str | None:
    vocabulary = set(known)
    for parent in clip.relative_to(root).parents:
        if parent == Path("."):
            break
        candidate = normalise_emotion(parent.name)
        if candidate in vocabulary:
            return candidate
    return None


def _subject_for(clip: Path, root: Path, fallback: str | None) -> str:
    """MMEW names clips ``PersonIndex-EmotionIndex-SampleIndex`` (e.g. ``S03-01-002``)."""
    head = clip.name.split("-")[0].split("_")[0]
    if head and head != clip.name:
        return head
    if fallback:
        return fallback
    parents = clip.relative_to(root).parts[:-1]
    return parents[0] if parents else clip.name


def build_records(
    root: str | Path,
    subset_dir: str,
    subset: str,
    maps: Mapping[str, Mapping[str, int]] | None = None,
    annotations: str | Path | None = None,
    skip_unknown: bool = False,
) -> list[ClipRecord]:
    """Scan one MMEW subset (``micro`` or ``macro``) into manifest records.

    Micro clips also receive the macro label of the same emotion; macro clips leave the
    micro label at ``IGNORE_INDEX`` so they only supervise the macro head. ``maps`` must
    cover both heads (see :func:`label_maps`); it defaults to the emotions of this subset
    alone, which is only correct when the manifest holds a single subset.
    """
    if subset not in {"micro", "macro"}:
        raise ValueError(f"subset must be 'micro' or 'macro', got '{subset}'")
    dataset_root = Path(root)
    subset_root = dataset_root / subset_dir
    if not subset_root.is_dir():
        raise FileNotFoundError(f"missing MMEW subset directory {subset_root}")

    if maps is None:
        emotions = discover_emotions(dataset_root, subset_dir, skip_unknown)
        maps = label_maps(emotions) if subset == "micro" else label_maps((), emotions)
    table = read_annotations(annotations) if annotations is not None else {}
    records: list[ClipRecord] = []
    unknown: set[str] = set()

    for clip in _clip_directories(subset_root):
        emotion = _emotion_for(clip, subset_root, maps[subset])
        if emotion is None:
            unknown.add(clip.parent.name)
            continue
        macro_label = maps["macro"].get(emotion)
        if macro_label is None:
            # e.g. micro-only "repression": keep the micro supervision, drop the macro one.
            macro_label = IGNORE_INDEX

        annotation = table.get(clip.name, Annotation())
        num_frames = sum(1 for p in clip.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        onset, apex, offset = rebase_indices(
            annotation.onset, annotation.apex, annotation.offset, num_frames
        )
        records.append(
            ClipRecord(
                clip_id=f"{subset}_{clip.name}",
                frames_dir=str(clip.relative_to(dataset_root)),
                macro_label=macro_label,
                micro_label=maps["micro"][emotion] if subset == "micro" else IGNORE_INDEX,
                onset=onset,
                apex=apex,
                offset=offset,
                subject=_subject_for(clip, subset_root, annotation.subject),
            )
        )

    if unknown and not skip_unknown:
        raise ValueError(
            f"unrecognised emotion directories under {subset_root}: {sorted(unknown)}; "
            "pass skip_unknown to ignore them"
        )
    if not records:
        raise ValueError(f"no clips found under {subset_root}")
    return records
