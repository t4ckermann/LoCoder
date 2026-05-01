from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

# Maps model name → system-message prefix that activates thinking mode.
# Gemma 4 uses <|think|>; Qwen3 uses /think (hybrid thinking token).
_THINKING_PREFIXES: dict[str, str] = {
    "gemma4-e2b": "<|think|>",
    "gemma4-e4b": "<|think|>",
    "gemma4-26b": "<|think|>",
    "gemma4-31b": "<|think|>",
    "qwen3-4b": "/think",
    "qwen3-8b": "/think",
    "qwen3-coder-next": "/think",
}


def _connect_host(host: str) -> str:
    """Resolve the host to connect to. 0.0.0.0 binds all interfaces but can't be dialled."""
    return "127.0.0.1" if host == "0.0.0.0" else host


def _make_client(host: str, port: int) -> OpenAI:
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


def _make_async_client(host: str, port: int) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


def get_client(config: dict[str, Any]) -> AsyncOpenAI:
    """Return an OpenAI-compatible async client pointed at the local llama-server."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    port: int = inf["single"]["port"]
    return _make_async_client(host, port)


def get_sync_client(config: dict[str, Any]) -> OpenAI:
    """Return a synchronous OpenAI-compatible client pointed at the local llama-server."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    port: int = inf["single"]["port"]
    return _make_client(host, port)


def get_planner_client(config: dict[str, Any]) -> OpenAI:
    """Return sync client for the planner server (dual: planner port; single: single port)."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    if inf.get("mode", "single") == "dual":
        port: int = inf["dual"]["planner"]["port"]
    else:
        port = inf["single"]["port"]
    return _make_client(host, port)


def get_executor_client(config: dict[str, Any]) -> OpenAI:
    """Return sync client for the executor server (dual: executor port; single: single port)."""
    inf = config["inference"]
    host: str = _connect_host(inf.get("host", "127.0.0.1"))
    if inf.get("mode", "single") == "dual":
        port: int = inf["dual"]["executor"]["port"]
    else:
        port = inf["single"]["port"]
    return _make_client(host, port)


def active_model_name(config: dict[str, Any]) -> str:
    """Return the configured model name (single mode)."""
    return str(config["inference"]["single"]["model"])


def planner_model_name(config: dict[str, Any]) -> str:
    """Return the planner model (dual: planner model; single: single model)."""
    inf = config["inference"]
    if inf.get("mode", "single") == "dual":
        return str(inf["dual"]["planner"]["model"])
    return str(inf["single"]["model"])


def executor_model_name(config: dict[str, Any]) -> str:
    """Return the executor model (dual: executor model; single: single model)."""
    inf = config["inference"]
    if inf.get("mode", "single") == "dual":
        return str(inf["dual"]["executor"]["model"])
    return str(inf["single"]["model"])


def supports_thinking(model_name: str) -> bool:
    """Return True if the model supports a thinking mode prefix."""
    return model_name in _THINKING_PREFIXES


def thinking_prefix(model_name: str, enabled: bool) -> str:
    """Return the system-message prefix to enable thinking mode, or empty string."""
    if enabled:
        return _THINKING_PREFIXES.get(model_name, "")
    return ""
