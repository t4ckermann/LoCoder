from __future__ import annotations

from locoder.models.selector import QUANT_LADDER, quant_size_gb, select_quant


def test_quant_size_gb_q5() -> None:
    expected = 7.6 * 5.60 / 8.0 * 1.15
    assert abs(quant_size_gb(7.6, "q5_k_m") - expected) < 0.001


def test_quant_size_gb_unknown_falls_back_to_q4() -> None:
    result = quant_size_gb(7.0, "q99_unknown")
    expected = 7.0 * 4.85 / 8.0 * 1.15
    assert abs(result - expected) < 0.001


def test_select_quant_ample_ram() -> None:
    assert select_quant("qwen2.5-coder-7b", 100.0) == "q5_k_m"


def test_select_quant_custom_prefer() -> None:
    assert select_quant("qwen2.5-coder-7b", 100.0, prefer="q8_0") == "q8_0"


def test_select_quant_tight_ram_steps_down() -> None:
    # ~3 GB — q5_k_m (≈6.1 GB) won't fit for a 7.6B model
    result = select_quant("qwen2.5-coder-7b", 3.0)
    assert result in {"q2_k", "q3_k_m"}


def test_select_quant_tiny_ram_falls_to_smallest() -> None:
    result = select_quant("qwen2.5-coder-7b", 0.1)
    assert result == "q2_k"


def test_select_quant_unknown_model_returns_prefer() -> None:
    assert select_quant("nonexistent-model", 8.0) == "q5_k_m"


def test_select_quant_prefer_not_in_ladder_falls_back() -> None:
    # "bogus" is not in the ladder — should fall back to q5_k_m start index
    result = select_quant("qwen2.5-coder-7b", 100.0, prefer="bogus")
    assert result in {name for name, _ in QUANT_LADDER}
