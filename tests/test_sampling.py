from __future__ import annotations

import pytest

from mres_fer.data.sampling import apex_centered_indices, uniform_indices


def test_uniform_indices_span_the_clip() -> None:
    indices = uniform_indices(10, 4)
    assert indices == [0, 3, 6, 9]


def test_uniform_indices_repeat_when_clip_is_short() -> None:
    indices = uniform_indices(2, 5)
    assert len(indices) == 5
    assert min(indices) == 0
    assert max(indices) == 1


def test_apex_window_stays_inside_bounds() -> None:
    indices = apex_centered_indices(20, 6, apex=1, onset=0, offset=19)
    assert len(indices) == 6
    assert min(indices) >= 0
    assert max(indices) <= 19
    assert indices == sorted(indices)


def test_apex_window_is_centred_when_there_is_room() -> None:
    indices = apex_centered_indices(100, 5, apex=50)
    assert indices == [48, 49, 50, 51, 52]


def test_empty_clip_is_rejected() -> None:
    with pytest.raises(ValueError):
        uniform_indices(0, 4)
