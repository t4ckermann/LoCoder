from __future__ import annotations

import pytest

from locoder.config.manager import _CTX_SIZE, _parallel_slots


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("small", 8192),
        ("mid", 32768),
        ("large", 65536),
    ],
)
def test_ctx_size_by_hint(hint: str, expected: int) -> None:
    assert _CTX_SIZE[hint] == expected


@pytest.mark.parametrize(
    ("cores", "has_gpu", "expected"),
    [
        (4, False, 1),
        (8, False, 1),
        (16, False, 1),
        (4, True, 2),
        (8, True, 4),
        (16, True, 4),
    ],
)
def test_parallel_slots(cores: int, has_gpu: bool, expected: int) -> None:
    assert _parallel_slots(cores, has_gpu) == expected
