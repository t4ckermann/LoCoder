from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil


@dataclass
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    vram_gb: float | None  # None = no discrete GPU detected
    free_port_single: int
    free_port_planner: int
    free_port_executor: int
    mode: Literal["single", "hierarchical"]
    model_hint: Literal["small", "mid", "large"]


def cpu_physical_cores() -> int:
    return psutil.cpu_count(logical=False) or 4


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / 1e9


def vram_gb() -> float | None:
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0]
            mib = float(first_line.strip())
            return mib / 1024.0
    except Exception:
        pass

    # Try Apple Silicon unified memory
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            bytes_total = int(result.stdout.strip())
            return bytes_total / 1e9
    except Exception:
        pass

    return None


def find_free_port(start: int) -> int:
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def detect() -> HardwareInfo:
    cores = cpu_physical_cores()
    ram = total_ram_gb()
    vram = vram_gb()

    effective_gb = max(ram, vram) if vram is not None else ram

    if effective_gb < 10:
        mode: Literal["single", "hierarchical"] = "single"
        model_hint: Literal["small", "mid", "large"] = "small"
    elif effective_gb <= 20:
        mode = "single"
        model_hint = "mid"
    else:
        mode = "hierarchical"
        model_hint = "large"

    port_single = find_free_port(8080)
    port_planner = find_free_port(8081)
    port_executor = find_free_port(8082)

    # Ensure no two ports collide when defaults overlap
    if port_planner == port_single:
        port_planner = find_free_port(port_single + 1)
    if port_executor in (port_single, port_planner):
        port_executor = find_free_port(max(port_single, port_planner) + 1)

    return HardwareInfo(
        cpu_cores=cores,
        ram_gb=round(ram, 1),
        vram_gb=round(vram, 1) if vram is not None else None,
        free_port_single=port_single,
        free_port_planner=port_planner,
        free_port_executor=port_executor,
        mode=mode,
        model_hint=model_hint,
    )
