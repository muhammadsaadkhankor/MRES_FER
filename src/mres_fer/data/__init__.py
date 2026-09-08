"""Dataset, sampling and motion-representation utilities."""

from mres_fer.data.dataset import ClipDataset, ClipRecord, build_dataloaders, collate_clips
from mres_fer.data.motion_magnification import EulerianMagnification
from mres_fer.data.optical_flow import FlowCache, compute_flow_sequence
from mres_fer.data.transforms import ClipTransform

__all__ = [
    "ClipDataset",
    "ClipRecord",
    "ClipTransform",
    "EulerianMagnification",
    "FlowCache",
    "build_dataloaders",
    "collate_clips",
    "compute_flow_sequence",
]
