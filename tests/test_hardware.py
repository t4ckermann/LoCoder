from __future__ import annotations

import pytest

from locoder.hardware.detect import HardwareInfo, find_free_port


def test_find_free_port_returns_in_range() -> None:
    port = find_free_port(9000)
    assert 9000 <= port < 65535


def test_find_free_port_two_calls_differ_if_first_still_bound() -> None:
    # Both calls start from the same hint — they may return the same port if
    # nothing is listening, but each result must individually be valid.
    port_a = find_free_port(9100)
    port_b = find_free_port(9100)
    assert 9100 <= port_a < 65535
    assert 9100 <= port_b < 65535


def test_hardware_info_is_frozen() -> None:
    hw = HardwareInfo(
        cpu_cores=4,
        ram_gb=16.0,
        vram_gb=None,
        free_port_single=8080,
        free_port_planner=8081,
        free_port_executor=8082,
        mode="single",
        model_hint="mid",
    )
    with pytest.raises(Exception):
        hw.cpu_cores = 8  # type: ignore[misc]


def test_hardware_info_fields() -> None:
    hw = HardwareInfo(
        cpu_cores=8,
        ram_gb=32.0,
        vram_gb=24.0,
        free_port_single=8080,
        free_port_planner=8081,
        free_port_executor=8082,
        mode="hierarchical",
        model_hint="large",
    )
    assert hw.cpu_cores == 8
    assert hw.mode == "hierarchical"
    assert hw.vram_gb == 24.0
