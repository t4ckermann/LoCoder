from __future__ import annotations

from openai import AsyncOpenAI

# Models that support Gemma 4 thinking mode via <|think|> prefix
_THINKING_MODELS = frozenset({"gemma4-e2b", "gemma4-e4b", "gemma4-26b", "gemma4-31b"})


def get_client(config: dict, role: str = "single") -> AsyncOpenAI:  # type: ignore[type-arg]
    """Return an OpenAI-compatible async client pointed at the local llama-server.

    role: "single" | "planner" | "executor"
    In single mode both "planner" and "executor" resolve to the single-model port.
    """
    inf = config["inference"]
    mode: str = inf["mode"]
    host: str = inf.get("host", "127.0.0.1")

    if mode == "single" or role == "single":
        port: int = inf["single"]["port"]
    elif role == "planner":
        port = inf["hierarchical"]["planner_port"]
    elif role == "executor":
        port = inf["hierarchical"]["executor_port"]
    else:
        raise ValueError(f"Unknown role: {role!r}. Expected 'single', 'planner', or 'executor'.")

    return AsyncOpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="not-needed",
    )


def active_model_name(config: dict, role: str = "single") -> str:  # type: ignore[type-arg]
    """Return the configured model name for the given role."""
    inf = config["inference"]
    mode: str = inf["mode"]

    if mode == "single" or role == "single":
        return str(inf["single"]["model"])
    if role == "planner":
        return str(inf["hierarchical"]["planner_model"])
    if role == "executor":
        return str(inf["hierarchical"]["executor_model"])
    raise ValueError(f"Unknown role: {role!r}")


def supports_thinking(model_name: str) -> bool:
    """Return True if the model supports Gemma 4 thinking mode."""
    return model_name in _THINKING_MODELS


def thinking_prefix(model_name: str, enabled: bool) -> str:
    """Return the system-message prefix to enable thinking mode, or empty string."""
    if enabled and supports_thinking(model_name):
        return "<|think|>"
    return ""
