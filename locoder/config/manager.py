from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

from locoder.hardware.detect import HardwareInfo

# Project-local config (preferred) — gitignored, per-project settings
LOCAL_CONFIG_NAME = ".locoder.toml"
# Global fallback — used when no local config exists
GLOBAL_CONFIG_PATH = Path("~/.locoder/config.toml").expanduser()

# Maps model_hint → registry short name
_HINT_TO_MODEL: dict[str, str] = {
    "small": "qwen2.5-coder-1.5b",
    "mid": "qwen2.5-coder-7b",
    "large": "qwen2.5-coder-14b",
}
_PLANNER_MODEL = "mistral-nemo"


def config_path() -> Path:
    """
    Resolution order:
    1. LOCODER_CONFIG env var (explicit override)
    2. .locoder.toml in the current working directory (project-local)
    3. ~/.locoder/config.toml (global fallback)
    """
    env = os.environ.get("LOCODER_CONFIG")
    if env:
        return Path(env).expanduser()

    local = Path.cwd() / LOCAL_CONFIG_NAME
    if local.exists():
        return local

    return GLOBAL_CONFIG_PATH


def default_write_path() -> Path:
    """Path where `locoder setup` writes the config — always the local file."""
    env = os.environ.get("LOCODER_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / LOCAL_CONFIG_NAME


def read_config() -> dict:
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No config found. Run `locoder setup` in your project directory to create "
            f"{LOCAL_CONFIG_NAME}, or set LOCODER_CONFIG to an explicit path."
        )
    with path.open("rb") as f:
        return tomllib.load(f)


def write_config(hw: HardwareInfo, llama_server_bin: str) -> None:
    executor_model = _HINT_TO_MODEL.get(hw.model_hint, "qwen2.5-coder-7b")
    ngl = 9999 if hw.vram_gb is not None else 0

    config: dict = {
        "inference": {
            "llama_server_bin": llama_server_bin,
            "host": "127.0.0.1",
            "mode": hw.mode,
            "single": {
                "model": executor_model,
                "port": hw.free_port_single,
            },
            "hierarchical": {
                "planner_model": _PLANNER_MODEL,
                "planner_port": hw.free_port_planner,
                "executor_model": executor_model,
                "executor_port": hw.free_port_executor,
            },
            "server_args": {
                "threads": hw.cpu_cores,
                "ctx_size": 32768,
                "batch_size": 512,
                "ubatch_size": 512,
                "flash_attn": "on",
                "parallel": 4,
                "ngl": ngl,
                "planner": {
                    "ctx_size": 16384,
                    "parallel": 2,
                },
                "executor": {
                    "ctx_size": 32768,
                    "parallel": 4,
                },
            },
        },
        "models": {
            "dir": "~/.locoder/models",
        },
        "agent": {
            "clarification_timeout": 10,
            "context_compaction_threshold": 0.80,
        },
        "sandbox": {
            "execution_timeout": 60,
            "max_extensions": 10,
            "allow_network": False,
        },
        "rag": {
            "embeddings_model": "nomic-embed-text",
            "vector_store_dir": "~/.locoder/vectorstore",
            "exclude": [
                "**/.git",
                "**/node_modules",
                "**/__pycache__",
                "**/dist",
                "**/*.lock",
            ],
            "chunk_size": 512,
            "chunk_overlap": 64,
            "top_k": 5,
        },
    }

    path = default_write_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(config).encode())
