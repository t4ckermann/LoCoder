from __future__ import annotations

from locoder.models.client import active_model_name, supports_thinking, thinking_prefix

_CONFIG = {
    "inference": {
        "host": "127.0.0.1",
        "single": {"model": "qwen2.5-coder-7b", "port": 8080},
    }
}


def test_supports_thinking_gemma4_models() -> None:
    for model in ("gemma4-e2b", "gemma4-e4b", "gemma4-26b", "gemma4-31b"):
        assert supports_thinking(model) is True, f"{model} should support thinking"


def test_supports_thinking_non_gemma() -> None:
    for model in ("qwen2.5-coder-7b", "mistral-nemo", "phi-4", "codellama-7b"):
        assert supports_thinking(model) is False, f"{model} should not support thinking"


def test_thinking_prefix_enabled_for_gemma() -> None:
    assert thinking_prefix("gemma4-e4b", True) == "<|think|>"


def test_thinking_prefix_disabled_returns_empty() -> None:
    assert thinking_prefix("gemma4-e4b", False) == ""


def test_thinking_prefix_non_gemma_always_empty() -> None:
    assert thinking_prefix("qwen2.5-coder-7b", True) == ""
    assert thinking_prefix("qwen2.5-coder-7b", False) == ""


def test_active_model_name() -> None:
    assert active_model_name(_CONFIG) == "qwen2.5-coder-7b"
