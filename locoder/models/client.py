from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

# Models that support Gemma 4 thinking mode via <|think|> prefix
_THINKING_MODELS = frozenset({"gemma4-e2b", "gemma4-e4b", "gemma4-26b", "gemma4-31b"})


def _connect_host(host: str) -> str:
    """Resolve the host to connect to. 0.0.0.0 binds all interfaces but can't be dialled."""
    return "127.0.0.1" if host == "0.0.0.0" else host


def get_client(config: dict[str, Any]) -> AsyncOpenAI:
    """Return an OpenAI-compatible async client pointed at the local llama-server."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    port: int = inf["single"]["port"]
    return AsyncOpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="not-needed",
    )


def get_sync_client(config: dict[str, Any]) -> OpenAI:
    """Return a synchronous OpenAI-compatible client pointed at the local llama-server."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    port: int = inf["single"]["port"]
    return OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="not-needed",
    )


def active_model_name(config: dict[str, Any]) -> str:
    """Return the configured model name."""
    return str(config["inference"]["single"]["model"])


def supports_thinking(model_name: str) -> bool:
    """Return True if the model supports Gemma 4 thinking mode."""
    return model_name in _THINKING_MODELS


def thinking_prefix(model_name: str, enabled: bool) -> str:
    """Return the system-message prefix to enable thinking mode, or empty string."""
    if enabled and supports_thinking(model_name):
        return "<|think|>"
    return ""
