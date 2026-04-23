from __future__ import annotations

import io
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_url
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from locoder.models.registry import lookup

_MODELS_DIR = Path("~/.locoder/models").expanduser()


def model_dir(model_id: str) -> Path:
    return _MODELS_DIR / model_id


def is_installed(model_id: str) -> bool:
    d = model_dir(model_id)
    return d.is_dir() and any(d.glob("*.gguf"))


def download(name: str, quant: str | None = None) -> Path:
    entry = lookup(name)
    if entry is None:
        raise ValueError(
            f"Unknown model '{name}'. Run `locoder registry update` or add it to registry.json."
        )

    resolved_quant = quant or entry["default_quant"]
    # Support {quant} (lowercase, e.g. Qwen) and {QUANT} (uppercase, e.g. bartowski)
    filename = entry["filename"].format(
        quant=resolved_quant, QUANT=resolved_quant.upper()
    )
    repo_id = entry["repo"]
    dest_dir = model_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    url = hf_hub_url(repo_id=repo_id, filename=filename)

    req = urllib.request.Request(url, headers={"User-Agent": "locoder"})

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(f"Downloading {filename}", total=None)

            with urllib.request.urlopen(req, timeout=300) as resp:
                total = resp.headers.get("Content-Length")
                total_bytes = int(total) if total else None
                progress.update(task_id, total=total_bytes)

                with dest_path.open("wb") as f:
                    downloaded = 0
                    chunk_size = 65536
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task_id, completed=downloaded)
    except BaseException:
        # Remove partial file so is_installed() doesn't return a false positive
        if dest_path.exists():
            dest_path.unlink()
        raise

    return dest_path


def remove(name: str) -> None:
    d = model_dir(name)
    if not d.exists():
        raise FileNotFoundError(f"Model '{name}' is not installed.")
    shutil.rmtree(d)
