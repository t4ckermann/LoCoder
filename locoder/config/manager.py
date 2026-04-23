from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

from locoder.hardware.detect import HardwareInfo

DEFAULT_CONFIG_PATH = Path("~/.locoder/config.toml").expanduser()

# Maps model_hint → registry short name
_HINT_TO_MODEL: dict[str, str] = {
    "small": "qwen2.5-coder-1.5b",
    "mid": "qwen2.5-coder-7b",
    "large": "qwen2.5-coder-14b",
}
_PLANNER_MODEL = "mistral-nemo"


def config_path() -> Path:
    env = os.environ.get("LOCODER_CONFIG")
    return Path(env).expanduser() if env else DEFAULT_CONFIG_PATH


def read_config() -> dict:
    path = config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run `locoder setup` to create it."
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

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(config).encode())
