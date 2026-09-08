"""Frame index sampling strategies for variable-length clips."""

from __future__ import annotations

import numpy as np


def uniform_indices(num_available: int, num_frames: int) -> list[int]:
    """Evenly spaced indices, repeating the last frame when the clip is too short."""
    if num_available <= 0:
        raise ValueError("clip has no frames")
    positions = np.linspace(0, num_available - 1, num=num_frames)
    return [int(i) for i in np.rint(positions).astype(int)]


def apex_centered_indices(
    num_available: int,
    num_frames: int,
    apex: int,
    onset: int | None = None,
    offset: int | None = None,
) -> list[int]:
    """Sample a window centred on the apex, clamped to the onset/offset bounds.

    Micro-expressions carry most of their evidence around the apex, so the window is
    kept tight instead of spanning the whole clip.
    """
    lo = 0 if onset is None else max(0, min(onset, num_available - 1))
    hi = num_available - 1 if offset is None else max(lo, min(offset, num_available - 1))
    apex = max(lo, min(apex, hi))

    half = (num_frames - 1) / 2
    start = apex - half
    end = apex + half
    if start < lo:
        end += lo - start
        start = lo
    if end > hi:
        start = max(lo, start - (end - hi))
        end = hi
    positions = np.linspace(start, end, num=num_frames)
    return [int(i) for i in np.rint(positions).astype(int)]
