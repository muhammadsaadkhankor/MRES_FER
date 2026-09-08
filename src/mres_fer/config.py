"""Typed configuration objects loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the clips live and how they are turned into tensors."""

    train_manifest: str = "data/train.json"
    val_manifest: str = "data/val.json"
    root: str = "data"
    num_frames: int = 16
    image_size: int = 224
    sampling: str = "uniform"  # uniform | apex_centered
    flow_cache_dir: str | None = None
    flow_algorithm: str = "farneback"  # farneback | tvl1
    horizontal_flip: bool = True
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Architecture switches mirroring the optional blocks of the pipeline."""

    use_appearance: bool = True
    use_motion: bool = True
    use_motion_magnification: bool = False
    use_gate: bool = True
    appearance_backbone: str = "vit_small_patch16_224"
    appearance_pretrained: bool = True
    freeze_appearance: bool = False
    motion_width: int = 64
    fusion_dim: int = 384
    temporal_layers: int = 2
    temporal_heads: int = 6
    dropout: float = 0.1
    num_micro_classes: int = 5
    num_macro_classes: int = 7
    macro_from_micro: bool = True


@dataclass
class MagnificationConfig:
    """Eulerian motion magnification applied to the frame stream."""

    factor: float = 8.0
    low_cut: float = 0.05
    high_cut: float = 0.4
    pyramid_levels: int = 3
    attenuate_chrominance: bool = True


@dataclass
class LossConfig:
    micro_weight: float = 1.0
    macro_weight: float = 1.0
    label_smoothing: float = 0.1
    ignore_index: int = -100


@dataclass
class OptimConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "runs/default"
    log_interval: int = 20
    monitor: str = "macro_uf1"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    magnification: MagnificationConfig = field(default_factory=MagnificationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "magnification": MagnificationConfig,
    "loss": LossConfig,
    "optim": OptimConfig,
    "run": RunConfig,
}


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config, merging an optional flat ``a.b=c`` override mapping."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value

    kwargs: dict[str, Any] = {}
    for name, values in raw.items():
        if name not in SECTIONS:
            raise ValueError(f"Unknown config section: {name}")
        kwargs[name] = _build(SECTIONS[name], values)
    return Config(**kwargs)
