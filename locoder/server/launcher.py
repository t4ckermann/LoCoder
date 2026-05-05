from __future__ import annotations

import atexit
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locoder.models.downloader import model_dir


@dataclass
class ServerHandle:
    proc: subprocess.Popen[bytes]
    port: int
    host: str
    model_path: Path
    role: str


def build_argv(
    llama_server_bin: str,
    model_path: Path,
    port: int,
    args: dict[str, object],
    host: str = "127.0.0.1",
) -> list[str]:
    argv = [
        llama_server_bin,
        "--model",
        str(model_path),
        "--port",
        str(port),
        "--host",
        host,
    ]

    key_map = {
        "threads": "--threads",
        "ctx_size": "--ctx-size",
        "batch_size": "--batch-size",
        "ubatch_size": "--ubatch-size",
        "parallel": "--parallel",
        "ngl": "-ngl",
        "draft_max": "--draft-max",
    }

    for cfg_key, flag in key_map.items():
        if cfg_key in args:
            argv += [flag, str(args[cfg_key])]

    # flash_attn takes a value: "on", "off", or "auto"
    flash = args.get("flash_attn", "auto")
    argv += ["--flash-attn", str(flash)]

    if "model_draft" in args:
        argv += ["--model-draft", str(args["model_draft"])]

    return argv


def _poll_health(
    port: int, host: str = "127.0.0.1", timeout: float = 60.0, interval: float = 0.5
) -> bool:
    # 0.0.0.0 means all interfaces — poll the loopback instead
    poll_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{poll_host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    return False


def _launch_one(
    bin_path: str,
    model_path: Path,
    port: int,
    server_args: dict[str, object],
    role: str,
    host: str = "127.0.0.1",
) -> ServerHandle:
    argv = build_argv(bin_path, model_path, port, server_args, host)

    # Ensure shared libraries next to the binary are found (needed when locoder
    # installed a pre-built release bundle into ~/.locoder/bin/).
    env = os.environ.copy()
    bin_dir = str(Path(bin_path).parent)
    for lib_var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = env.get(lib_var, "")
        env[lib_var] = f"{bin_dir}:{existing}" if existing else bin_dir

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    if not _poll_health(port, host):
        proc.terminate()
        _, stderr = proc.communicate(timeout=5)
        tail = stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(
            f"llama-server ({role}) did not become healthy within 60 s.\nLast stderr:\n{tail}"
        )

    return ServerHandle(proc=proc, port=port, host=host, model_path=model_path, role=role)


def start_servers_dual(config: dict[str, Any]) -> tuple[ServerHandle, ServerHandle]:
    """Start two llama-server processes for dual-model (planner + executor) mode."""
    inf = config["inference"]
    bin_path: str = inf["llama_server_bin"]
    server_args: dict[str, Any] = dict(inf.get("server_args", {}))
    host: str = inf.get("host", "127.0.0.1")
    dual: dict[str, Any] = inf["dual"]

    planner_handle = _launch_one(
        bin_path,
        _resolve_gguf(str(dual["planner"]["model"])),
        int(dual["planner"]["port"]),
        server_args,
        "planner",
        host,
    )
    atexit.register(stop_server, planner_handle)

    executor_handle = _launch_one(
        bin_path,
        _resolve_gguf(str(dual["executor"]["model"])),
        int(dual["executor"]["port"]),
        server_args,
        "executor",
        host,
    )
    atexit.register(stop_server, executor_handle)

    return planner_handle, executor_handle


def start_server(config: dict[str, Any]) -> ServerHandle:
    inf = config["inference"]
    bin_path: str = inf["llama_server_bin"]
    server_args: dict[str, Any] = dict(inf.get("server_args", {}))
    model_name: str = inf["single"]["model"]
    port: int = inf["single"]["port"]
    host: str = inf.get("host", "127.0.0.1")
    gguf = _resolve_gguf(model_name)

    spec = inf.get("speculative", {})
    if spec.get("enabled", False):
        draft_name: str = spec["model_draft"]
        server_args["model_draft"] = str(_resolve_gguf(draft_name))
        server_args["draft_max"] = int(spec.get("draft_max", 8))

    handle = _launch_one(bin_path, gguf, port, server_args, "single", host)
    atexit.register(stop_server, handle)
    return handle


def _resolve_gguf(model_name: str) -> Path:
    d = model_dir(model_name)
    ggufs = list(d.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(
            f"No .gguf file found for model '{model_name}' in {d}. "
            "Run `locoder pull <model>` first."
        )
    return sorted(ggufs)[0]


def stop_server(handle: ServerHandle) -> None:
    try:
        handle.proc.terminate()
        handle.proc.wait(timeout=5)
    except Exception:
        pass
