from __future__ import annotations

import atexit
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int
    model_path: Path
    role: str


def build_argv(
    llama_server_bin: str,
    model_path: Path,
    port: int,
    args: dict,
) -> list[str]:
    argv = [
        llama_server_bin,
        "--model", str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    key_map = {
        "threads": "--threads",
        "ctx_size": "--ctx-size",
        "batch_size": "--batch-size",
        "ubatch_size": "--ubatch-size",
        "parallel": "--parallel",
        "ngl": "-ngl",
    }

    for cfg_key, flag in key_map.items():
        if cfg_key in args:
            argv += [flag, str(args[cfg_key])]

    # flash_attn takes a value: "on", "off", or "auto"
    flash = args.get("flash_attn", "auto")
    argv += ["--flash-attn", str(flash)]

    return argv


def _poll_health(port: int, timeout: float = 60.0, interval: float = 0.5) -> bool:
    url = f"http://127.0.0.1:{port}/health"
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
    server_args: dict,
    role: str,
) -> ServerHandle:
    argv = build_argv(bin_path, model_path, port, server_args)

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

    if not _poll_health(port):
        proc.terminate()
        _, stderr = proc.communicate(timeout=5)
        tail = stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(
            f"llama-server ({role}) did not become healthy within 60 s.\n"
            f"Last stderr:\n{tail}"
        )

    return ServerHandle(proc=proc, port=port, model_path=model_path, role=role)


def start_server(mode: str, config: dict) -> list[ServerHandle]:
    from locoder.models.downloader import model_dir

    inf = config["inference"]
    bin_path: str = inf["llama_server_bin"]
    shared_args: dict = dict(inf.get("server_args", {}))
    # Remove sub-tables before passing as flat args
    shared_args.pop("planner", None)
    shared_args.pop("executor", None)

    handles: list[ServerHandle] = []

    if mode == "single":
        model_name: str = inf["single"]["model"]
        port: int = inf["single"]["port"]
        gguf = _resolve_gguf(model_name)
        handles.append(_launch_one(bin_path, gguf, port, shared_args, "single"))

    elif mode == "hierarchical":
        planner_name: str = inf["hierarchical"]["planner_model"]
        planner_port: int = inf["hierarchical"]["planner_port"]
        executor_name: str = inf["hierarchical"]["executor_model"]
        executor_port: int = inf["hierarchical"]["executor_port"]

        planner_args = {**shared_args, **inf.get("server_args", {}).get("planner", {})}
        executor_args = {**shared_args, **inf.get("server_args", {}).get("executor", {})}

        planner_gguf = _resolve_gguf(planner_name)
        executor_gguf = _resolve_gguf(executor_name)

        handles.append(_launch_one(bin_path, planner_gguf, planner_port, planner_args, "planner"))
        handles.append(_launch_one(bin_path, executor_gguf, executor_port, executor_args, "executor"))

    else:
        raise ValueError(f"Unknown inference mode: {mode!r}")

    atexit.register(stop_servers, handles)
    return handles


def _resolve_gguf(model_name: str) -> Path:
    from locoder.models.downloader import model_dir

    d = model_dir(model_name)
    ggufs = list(d.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(
            f"No .gguf file found for model '{model_name}' in {d}. "
            "Run `locoder pull <model>` first."
        )
    return ggufs[0]


def stop_servers(handles: list[ServerHandle]) -> None:
    for handle in handles:
        try:
            handle.proc.terminate()
            handle.proc.wait(timeout=5)
        except Exception:
            pass
