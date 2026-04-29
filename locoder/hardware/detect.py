from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    vram_gb: float | None  # None = no discrete GPU detected
    free_port_single: int
    model_hint: Literal["small", "mid", "large"]


def cpu_physical_cores() -> int:
    return psutil.cpu_count(logical=False) or 4


def total_ram_gb() -> float:
    return float(psutil.virtual_memory().total) / 1e9


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


def available_gb() -> float:
    """Return the effective available memory in GB (VRAM if present, else RAM)."""
    ram = total_ram_gb()
    vram = vram_gb()
    return max(ram, vram) if vram is not None else ram


def detect() -> HardwareInfo:
    cores = cpu_physical_cores()
    ram = total_ram_gb()
    vram = vram_gb()

    effective_gb = max(ram, vram) if vram is not None else ram

    if effective_gb < 10:
        model_hint: Literal["small", "mid", "large"] = "small"
    elif effective_gb <= 20:
        model_hint = "mid"
    else:
        model_hint = "large"

    return HardwareInfo(
        cpu_cores=cores,
        ram_gb=round(ram, 1),
        vram_gb=round(vram, 1) if vram is not None else None,
        free_port_single=find_free_port(8080),
        model_hint=model_hint,
    )
