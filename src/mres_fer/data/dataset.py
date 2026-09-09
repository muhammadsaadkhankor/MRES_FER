"""Clip dataset driven by a JSON manifest.

A manifest is a JSON list of records, one per clip::

    [
      {
        "clip_id": "sub01_EP02_01f",
        "frames_dir": "casme2/sub01/EP02_01f",
        "macro_label": 2,
        "micro_label": 1,
        "onset": 0, "apex": 27, "offset": 55,
        "subject": "sub01"
      }
    ]

``frames_dir`` is relative to ``DataConfig.root`` and holds frames named in sortable
order (``img_00001.jpg`` ...). It may also point at a single image file, as MMEW does for
its macro samples: the still is then repeated across the clip and its flow is zero, so
such samples train the appearance branch only. ``micro_label`` may be omitted for
datasets that only carry macro annotations; such clips are ignored by the micro loss.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from mres_fer.config import DataConfig
from mres_fer.data.optical_flow import FlowCache, compute_flow_sequence
from mres_fer.data.sampling import apex_centered_indices, uniform_indices
from mres_fer.data.transforms import ClipTransform

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IGNORE_INDEX = -100


@dataclass
class ClipRecord:
    clip_id: str
    frames_dir: str
    macro_label: int
    micro_label: int = IGNORE_INDEX
    onset: int | None = None
    apex: int | None = None
    offset: int | None = None
    subject: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ClipRecord:
        return cls(
            clip_id=str(raw["clip_id"]),
            frames_dir=str(raw["frames_dir"]),
            macro_label=int(raw["macro_label"]),
            micro_label=int(raw.get("micro_label", IGNORE_INDEX)),
            onset=raw.get("onset"),
            apex=raw.get("apex"),
            offset=raw.get("offset"),
            subject=raw.get("subject"),
        )


def frame_order_key(path: Path) -> tuple[int, str]:
    """Natural order, so MMEW frames named ``2.jpg`` sort before ``10.jpg``."""
    stem = path.stem
    return (int(stem), "") if stem.isdigit() else (0, stem)


def is_still(root: str | Path, record: ClipRecord) -> bool:
    """A record pointing at an image file rather than a frame directory."""
    return (Path(root) / record.frames_dir).suffix.lower() in IMAGE_SUFFIXES


def frame_paths(root: str | Path, record: ClipRecord) -> list[Path]:
    """Frames of a clip in natural order, or the single image of a still record."""
    directory = Path(root) / record.frames_dir
    if directory.suffix.lower() in IMAGE_SUFFIXES:
        return [directory]
    paths = sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=frame_order_key,
    )
    if not paths:
        raise FileNotFoundError(f"no frames found in {directory}")
    return paths


def read_manifest(path: str | Path) -> list[ClipRecord]:
    raw = json.loads(Path(path).read_text())
    return [ClipRecord.from_dict(item) for item in raw]


def write_manifest(path: str | Path, records: Sequence[ClipRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return destination


class ClipDataset(Dataset[dict[str, Tensor | str]]):
    """Yields ``frames`` ``(T, 3, H, W)`` in ``[0, 1]`` and ``flow`` ``(T, 2, H, W)``."""

    def __init__(
        self,
        records: list[ClipRecord],
        config: DataConfig,
        train: bool,
        with_flow: bool = True,
    ) -> None:
        if not records:
            raise ValueError("manifest is empty")
        self.records = records
        self.config = config
        self.train = train
        self.with_flow = with_flow
        self.root = Path(config.root)
        self.transform = ClipTransform(
            image_size=config.image_size,
            train=train,
            horizontal_flip=config.horizontal_flip,
        )
        self.flow_cache = (
            FlowCache(config.flow_cache_dir, config.flow_algorithm)
            if config.flow_cache_dir is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _select_indices(self, record: ClipRecord, num_available: int) -> list[int]:
        num_frames = self.config.num_frames
        if self.config.sampling == "apex_centered" and record.apex is not None:
            return apex_centered_indices(
                num_available, num_frames, record.apex, record.onset, record.offset
            )
        return uniform_indices(num_available, num_frames)

    def _load_frames(self, paths: list[Path], indices: list[int]) -> np.ndarray:
        frames = []
        for index in indices:
            image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read frame {paths[index]}")
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)

    def _flow_for(self, record: ClipRecord, frames: np.ndarray, indices: list[int]) -> np.ndarray:
        size = (frames.shape[1], frames.shape[2])
        if len(set(indices)) == 1:  # still image: no motion to estimate
            return np.zeros((frames.shape[0], *size, 2), dtype=np.float32)
        if self.flow_cache is not None:
            cached = self.flow_cache.load(record.clip_id, indices, size)
            if cached is not None:
                return cached.astype(np.float32)
        flow = compute_flow_sequence(list(frames), self.config.flow_algorithm)
        if self.flow_cache is not None:
            self.flow_cache.save(record.clip_id, indices, size, flow)
        return flow

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        paths = frame_paths(self.root, record)
        indices = self._select_indices(record, len(paths))
        frames = self._load_frames(paths, indices)
        flow = self._flow_for(record, frames, indices) if self.with_flow else None

        frames_t, flow_t = self.transform(frames, flow)
        sample: dict[str, Tensor | str] = {
            "frames": frames_t,
            "macro_label": torch.tensor(record.macro_label, dtype=torch.long),
            "micro_label": torch.tensor(record.micro_label, dtype=torch.long),
            "clip_id": record.clip_id,
        }
        if flow_t is not None:
            sample["flow"] = flow_t
        return sample


def collate_clips(batch: list[dict[str, Tensor | str]]) -> dict[str, Tensor | list[str]]:
    """Stack tensor fields and keep ``clip_id`` as a plain list of strings."""
    out: dict[str, Tensor | list[str]] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key == "clip_id":
            out[key] = [str(v) for v in values]
        else:
            out[key] = torch.stack([v for v in values if isinstance(v, Tensor)])
    return out


def build_dataloaders_from_records(
    config: DataConfig,
    train_records: Sequence[ClipRecord],
    val_records: Sequence[ClipRecord],
    batch_size: int,
    with_flow: bool = True,
) -> tuple[DataLoader[dict[str, Tensor | str]], DataLoader[dict[str, Tensor | str]]]:
    train_set = ClipDataset(list(train_records), config, True, with_flow)
    val_set = ClipDataset(list(val_records), config, False, with_flow)
    common = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_clips,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)  # type: ignore[arg-type]
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)  # type: ignore[arg-type]
    return train_loader, val_loader


def build_dataloaders(
    config: DataConfig, batch_size: int, with_flow: bool = True
) -> tuple[DataLoader[dict[str, Tensor | str]], DataLoader[dict[str, Tensor | str]]]:
    return build_dataloaders_from_records(
        config,
        read_manifest(config.train_manifest),
        read_manifest(config.val_manifest),
        batch_size,
        with_flow,
    )
