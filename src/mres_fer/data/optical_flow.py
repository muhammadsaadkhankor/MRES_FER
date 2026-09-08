"""Dense optical flow extraction with an on-disk cache.

Flow is the motion evidence of the pipeline: it is expensive to recompute every epoch,
so a cache keyed by clip id and sampled frame indices is provided.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _farneback(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore[call-overload]
    return np.asarray(flow)


def _tvl1(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    solver = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore[attr-defined]
    return np.asarray(solver.calc(prev, curr, None))


def compute_flow_sequence(
    frames: Sequence[np.ndarray],
    algorithm: str = "farneback",
    reference: str = "previous",
) -> np.ndarray:
    """Return ``(T, H, W, 2)`` flow, the first entry being zero motion.

    ``reference='previous'`` gives frame-to-frame motion; ``reference='onset'`` gives
    displacement relative to the first sampled frame, which is the usual choice for
    apex-based micro-expression descriptors.
    """
    if algorithm == "farneback":
        estimator = _farneback
    elif algorithm == "tvl1":
        estimator = _tvl1
    else:
        raise ValueError(f"unknown optical flow algorithm: {algorithm}")
    if reference not in {"previous", "onset"}:
        raise ValueError(f"unknown flow reference: {reference}")

    grays = [_to_gray(f) for f in frames]
    flows = [np.zeros((*grays[0].shape, 2), dtype=np.float32)]
    for i in range(1, len(grays)):
        prev = grays[0] if reference == "onset" else grays[i - 1]
        flows.append(estimator(prev, grays[i]).astype(np.float32))
    return np.stack(flows, axis=0)


class FlowCache:
    """Stores flow arrays as compressed ``.npz`` files under ``root``."""

    def __init__(self, root: str | Path, algorithm: str = "farneback") -> None:
        self.root = Path(root)
        self.algorithm = algorithm
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, clip_id: str, indices: Sequence[int], size: tuple[int, int]) -> str:
        payload = f"{clip_id}|{self.algorithm}|{size}|{','.join(map(str, indices))}"
        return hashlib.sha1(payload.encode()).hexdigest()

    def path(self, clip_id: str, indices: Sequence[int], size: tuple[int, int]) -> Path:
        return self.root / f"{self.key(clip_id, indices, size)}.npz"

    def load(
        self, clip_id: str, indices: Sequence[int], size: tuple[int, int]
    ) -> np.ndarray | None:
        path = self.path(clip_id, indices, size)
        if not path.exists():
            return None
        with np.load(path) as data:
            return data["flow"]

    def save(
        self, clip_id: str, indices: Sequence[int], size: tuple[int, int], flow: np.ndarray
    ) -> None:
        np.savez_compressed(self.path(clip_id, indices, size), flow=flow.astype(np.float16))
