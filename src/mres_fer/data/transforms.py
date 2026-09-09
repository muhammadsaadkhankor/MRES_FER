"""Clip-consistent geometric transforms.

Every frame of a clip must receive the same geometric transform, otherwise the optical
flow computed from it no longer describes facial motion. Frames are left in ``[0, 1]``:
ImageNet normalisation happens inside the model, after the optional magnification block.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ClipTransform:
    """Resize and optionally augment a ``(T, H, W, 3)`` uint8 clip and its flow."""

    def __init__(
        self,
        image_size: int = 224,
        train: bool = True,
        horizontal_flip: bool = True,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.horizontal_flip = horizontal_flip

    def __call__(self, frames: np.ndarray, flow: np.ndarray | None) -> tuple[Tensor, Tensor | None]:
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        flow_t = None
        if flow is not None:
            flow_t = torch.from_numpy(np.ascontiguousarray(flow)).permute(0, 3, 1, 2).float()

        clip = torch.nn.functional.interpolate(
            clip, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        if flow_t is not None:
            scale_h = self.image_size / flow_t.shape[-2]
            scale_w = self.image_size / flow_t.shape[-1]
            flow_t = torch.nn.functional.interpolate(
                flow_t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            # Displacements are in pixels, so they scale with the resize.
            flow_t[:, 0] *= scale_w
            flow_t[:, 1] *= scale_h

        if self.train and self.horizontal_flip and random.random() < 0.5:
            clip = torch.flip(clip, dims=[-1])
            if flow_t is not None:
                flow_t = torch.flip(flow_t, dims=[-1])
                flow_t[:, 0] *= -1.0

        return clip, flow_t
