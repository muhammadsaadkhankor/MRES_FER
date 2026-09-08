from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mres_fer.config import Config, DataConfig, ModelConfig, OptimConfig, RunConfig


def _write_clip(directory: Path, num_frames: int, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    for i in range(num_frames):
        frame = np.roll(base, shift=i, axis=1)
        cv2.imwrite(str(directory / f"img_{i:05d}.jpg"), frame)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A two-clip dataset on disk with train/val manifests."""
    root = tmp_path / "data"
    records = []
    for i in range(2):
        clip_id = f"clip{i}"
        _write_clip(root / clip_id, num_frames=8, size=48)
        records.append(
            {
                "clip_id": clip_id,
                "frames_dir": clip_id,
                "macro_label": i % 2,
                "micro_label": i % 2,
                "onset": 0,
                "apex": 4,
                "offset": 7,
                "subject": f"s{i}",
            }
        )
    (root / "train.json").write_text(json.dumps(records))
    (root / "val.json").write_text(json.dumps(records))
    return root


@pytest.fixture
def tiny_config(tiny_dataset: Path, tmp_path: Path) -> Config:
    return Config(
        data=DataConfig(
            train_manifest=str(tiny_dataset / "train.json"),
            val_manifest=str(tiny_dataset / "val.json"),
            root=str(tiny_dataset),
            num_frames=4,
            image_size=32,
            flow_cache_dir=str(tmp_path / "flow_cache"),
            num_workers=0,
        ),
        model=ModelConfig(
            appearance_pretrained=False,
            appearance_backbone="vit_tiny_patch16_224",
            fusion_dim=64,
            temporal_layers=1,
            temporal_heads=2,
            num_micro_classes=3,
            num_macro_classes=2,
        ),
        optim=OptimConfig(epochs=1, batch_size=2, amp=False, warmup_epochs=1),
        run=RunConfig(device="cpu", output_dir=str(tmp_path / "run")),
    )
