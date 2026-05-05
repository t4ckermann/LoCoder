from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

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
from locoder.models.selector import select_quant

MODELS_DIR = Path("~/.locoder/models").expanduser()


def model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def is_installed(model_id: str) -> bool:
    d = model_dir(model_id)
    return d.is_dir() and any(d.glob("*.gguf"))


def download(name: str, quant: str | None = None, available_gb: float | None = None) -> Path:
    """Download a model GGUF (single-file or sharded).

    available_gb: effective RAM/VRAM available for quant selection. When None and quant
    is also None, falls back to the registry's default_quant.
    """
    entry = lookup(name)
    if entry is None:
        raise ValueError(
            f"Unknown model '{name}'. Run `locoder registry update` or add it to registry.json."
        )

    if quant:
        resolved_quant = quant
    elif available_gb is not None:
        resolved_quant = select_quant(name, available_gb)
    else:
        resolved_quant = str(entry["default_quant"])

    repo_id: str = str(entry["repo"])
    shard_count = int(entry.get("shard_count", 1))
    dest_dir = model_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build (hf_path, local_dest) for every shard.
    # {SHARD}/{TOTAL} placeholders are used for multi-shard models; extra kwargs are
    # silently ignored by str.format() for single-file entries that lack them.
    shards: list[tuple[str, Path]] = []
    for shard_n in range(1, shard_count + 1):
        # Support {quant} (lowercase), {QUANT} (uppercase), {SHARD}, {TOTAL}
        hf_path = str(entry["filename"]).format(
            quant=resolved_quant,
            QUANT=resolved_quant.upper(),
            SHARD=shard_n,
            TOTAL=shard_count,
        )
        local_path = dest_dir / Path(hf_path).name  # strip any HF subdirectory
        shards.append((hf_path, local_path))

    first_hf_path, first_local = shards[0]

    # Pre-flight HEAD on first shard to catch wrong repo/filename early.
    first_url = hf_hub_url(repo_id=repo_id, filename=first_hf_path)
    head = urllib.request.Request(first_url, headers={"User-Agent": "locoder"}, method="HEAD")
    try:
        urllib.request.urlopen(head, timeout=15)
    except HTTPError as exc:
        raise ValueError(
            f"File '{first_hf_path}' not found in repo '{repo_id}' (HTTP {exc.code}). "
            "The registry entry may be outdated — run `locoder registry update` to refresh."
        ) from None

    for i, (hf_path, dest_path) in enumerate(shards, 1):
        if dest_path.exists():
            continue

        url = hf_hub_url(repo_id=repo_id, filename=hf_path)
        req = urllib.request.Request(url, headers={"User-Agent": "locoder"})
        label = (
            f"Downloading {dest_path.name} ({i}/{shard_count})"
            if shard_count > 1
            else f"Downloading {dest_path.name}"
        )

        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task_id = progress.add_task(label, total=None)

                with urllib.request.urlopen(req, timeout=300) as resp:
                    total = resp.headers.get("Content-Length")
                    total_bytes = int(total) if total else None
                    progress.update(task_id, total=total_bytes)

                    with dest_path.open("wb") as f:
                        done = 0
                        while chunk := resp.read(65536):
                            f.write(chunk)
                            done += len(chunk)
                            progress.update(task_id, completed=done)
        except BaseException:
            # Remove partial shard so is_installed() doesn't return a false positive.
            # Already-completed shards are kept so a retry can resume from this shard.
            if dest_path.exists():
                dest_path.unlink()
            raise

    return first_local


def remove(name: str) -> None:
    d = model_dir(name)
    if not d.exists():
        raise FileNotFoundError(f"Model '{name}' is not installed.")
    shutil.rmtree(d)
