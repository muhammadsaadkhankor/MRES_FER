from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import Tensor

from mres_fer.config import Config
from mres_fer.data.dataset import ClipDataset, build_dataloaders, read_manifest
from mres_fer.data.optical_flow import FlowCache, compute_flow_sequence
from mres_fer.data.transforms import ClipTransform


def test_flow_sequence_shape_and_zero_first_frame() -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    flow = compute_flow_sequence(frames)
    assert flow.shape == (3, 32, 32, 2)
    assert np.all(flow[0] == 0)


def test_flow_cache_roundtrip(tmp_path: Path) -> None:
    cache = FlowCache(tmp_path)
    flow = np.ones((2, 4, 4, 2), dtype=np.float32)
    assert cache.load("clip", [0, 1], (4, 4)) is None
    cache.save("clip", [0, 1], (4, 4), flow)
    loaded = cache.load("clip", [0, 1], (4, 4))
    assert loaded is not None
    assert loaded.shape == flow.shape


def test_horizontal_flip_negates_the_x_component() -> None:
    transform = ClipTransform(image_size=8, train=True, horizontal_flip=False)
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    flow = np.ones((2, 8, 8, 2), dtype=np.float32)
    clip, flow_t = transform(frames, flow)
    assert clip.shape == (2, 3, 8, 8)
    assert flow_t is not None
    assert float(flow_t[:, 0].mean()) > 0


def test_dataset_item_shapes(tiny_config: Config) -> None:
    records = read_manifest(tiny_config.data.train_manifest)
    dataset = ClipDataset(records, tiny_config.data, train=False)
    sample = dataset[0]
    frames, flow = sample["frames"], sample["flow"]
    assert isinstance(frames, Tensor) and isinstance(flow, Tensor)
    size = tiny_config.data.image_size
    assert frames.shape == (tiny_config.data.num_frames, 3, size, size)
    assert flow.shape == (tiny_config.data.num_frames, 2, size, size)
    assert float(frames.max()) <= 1.0


def test_dataloaders_collate_clip_ids(tiny_config: Config) -> None:
    train_loader, _ = build_dataloaders(tiny_config.data, batch_size=2)
    batch = next(iter(train_loader))
    assert batch["frames"].shape[0] == 2
    assert isinstance(batch["clip_id"], list)
    assert batch["macro_label"].shape == (2,)
