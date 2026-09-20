"""Datasets over the pre-extracted ViT features (no images are read here)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def uniform_indices(num_frames: int, num_samples: int) -> np.ndarray:
    """Uniformly spaced indices, repeating frames when the clip is too short."""
    if num_frames <= 0:
        raise ValueError("clip has no frames")
    return np.clip(np.round(np.linspace(0, num_frames - 1, num_samples)).astype(int), 0, num_frames - 1)


def centred_indices(num_frames: int, num_samples: int, centre: int) -> np.ndarray:
    """Window of `num_samples` indices around `centre` (used for the apex frame)."""
    half = num_samples // 2
    start = int(np.clip(centre - half, 0, max(num_frames - num_samples, 0)))
    idx = np.arange(start, start + num_samples)
    return np.clip(idx, 0, num_frames - 1)


def make_windows(features: np.ndarray, size: int, stride: int, max_windows: int) -> np.ndarray:
    """Short overlapping windows over a [T, F] feature sequence -> [N, size, F]."""
    length = features.shape[0]
    if length < size:
        pad = np.repeat(features[-1:], size - length, axis=0)
        features = np.concatenate([features, pad], axis=0)
        length = size
    starts = list(range(0, length - size + 1, max(stride, 1)))
    if len(starts) > max_windows:  # keep windows spread over the whole clip
        keep = np.round(np.linspace(0, len(starts) - 1, max_windows)).astype(int)
        starts = [starts[i] for i in keep]
    while len(starts) < max_windows:  # pad by repeating the last window
        starts.append(starts[-1])
    return np.stack([features[s : s + size] for s in starts], axis=0)


@dataclass
class ClipRecord:
    clip_id: str
    subject: str
    label: str
    label_idx: int
    feature_path: Path
    num_frames: int
    apex: int = -1


def _records(index_csv: Path, class_to_idx: dict[str, int] | None) -> list[ClipRecord]:
    frame = pd.read_csv(index_csv)
    records: list[ClipRecord] = []
    for row in frame.itertuples(index=False):
        label = str(row.label)
        if class_to_idx is not None and label not in class_to_idx:
            continue
        records.append(
            ClipRecord(
                clip_id=str(row.clip_id),
                subject=str(row.subject),
                label=label,
                label_idx=class_to_idx[label] if class_to_idx else -1,
                feature_path=Path(str(row.feature_path)),
                num_frames=int(row.num_frames),
                apex=int(getattr(row, "apex", -1)),
            )
        )
    return records


class MacroClipDataset(Dataset):
    """Full macro clip + its short overlapping windows, both from ViT features."""

    def __init__(
        self,
        index_csv: Path,
        class_to_idx: dict[str, int],
        sampling: dict,
        subjects: list[str] | None = None,
        train: bool = False,
    ) -> None:
        self.records = _records(Path(index_csv), class_to_idx)
        if subjects is not None:
            keep = set(subjects)
            self.records = [r for r in self.records if r.subject in keep]
        self.sampling = sampling
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, record: ClipRecord) -> np.ndarray:
        return np.load(record.feature_path).astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        features = self._load(record)
        idx = uniform_indices(features.shape[0], self.sampling["macro_frames"])
        if self.train:  # jitter the uniform sampling grid a little
            jitter = np.random.randint(-1, 2, size=idx.shape)
            idx = np.clip(idx + jitter, 0, features.shape[0] - 1)
        frames = features[idx]
        windows = make_windows(
            features,
            self.sampling["window_size"],
            self.sampling["window_stride"],
            self.sampling["max_windows"],
        )
        return {
            "frames": torch.from_numpy(frames),
            "windows": torch.from_numpy(windows),
            "label": torch.tensor(record.label_idx, dtype=torch.long),
            "clip_id": record.clip_id,
            "subject": record.subject,
        }


class MicroClipDataset(Dataset):
    """Micro-expression clips, used as an unlabeled corpus of facial dynamics."""

    def __init__(self, index_csv: Path, sampling: dict) -> None:
        self.records = _records(Path(index_csv), None)
        self.sampling = sampling
        self.classes = sorted({r.label for r in self.records})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        features = np.load(record.feature_path).astype(np.float32)
        num = self.sampling["micro_frames"]
        if record.apex >= 0:
            idx = centred_indices(features.shape[0], num, record.apex)
        else:
            idx = uniform_indices(features.shape[0], num)
        clip = features[idx]
        windows = make_windows(
            clip,
            self.sampling["window_size"],
            self.sampling["window_stride"],
            max_windows=2,
        )
        return {
            "windows": torch.from_numpy(windows),
            "label": torch.tensor(self.class_to_idx[record.label], dtype=torch.long),
            "clip_id": record.clip_id,
        }


def augment_views(windows: torch.Tensor, noise: float = 0.05, drop: float = 0.1) -> torch.Tensor:
    """Second view of a window: feature noise + random frame dropout (repeat)."""
    view = windows + noise * torch.randn_like(windows)
    if drop > 0:
        keep = (torch.rand(view.shape[:-1], device=view.device) > drop).float().unsqueeze(-1)
        view = view * keep + windows.mean(dim=-2, keepdim=True) * (1 - keep)
    return view


def subject_split(
    subjects: list[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    """Subject-independent split (no subject appears in two sets)."""
    unique = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_ratio)))
    n_val = max(1, int(round(len(unique) * val_ratio)))
    test, val, train = unique[:n_test], unique[n_test : n_test + n_val], unique[n_test + n_val :]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_splits(path: Path, splits: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(splits, handle, indent=2)


def read_splits(path: Path) -> dict[str, list[str]]:
    with open(path) as handle:
        return json.load(handle)
