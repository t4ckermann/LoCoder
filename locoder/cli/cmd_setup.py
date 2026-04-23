from __future__ import annotations

import subprocess

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from locoder.config.manager import config_path, write_config
from locoder.hardware.detect import detect
from locoder.server.install import download_and_install, find_on_path, installed_bin

console = Console()


def _resolve_llama_server() -> str:
    """
    Return an absolute path to llama-server, installing it automatically if needed.
    Priority: already-managed install → PATH → fresh download.
    """
    # 1. Already installed by locoder
    managed = installed_bin()
    if managed:
        console.print(f"[dim]Found managed llama-server at {managed}[/dim]")
        return str(managed)

    # 2. Already on PATH
    on_path = find_on_path()
    if on_path:
        console.print(f"[dim]Found llama-server on PATH: {on_path}[/dim]")
        return on_path

    # 3. Download pre-built binary
    console.print(
        "[yellow]llama-server not found — downloading pre-built binary from "
        "llama.cpp GitHub releases...[/yellow]"
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Downloading llama-server", total=None)

        def _cb(downloaded: int, total: int | None) -> None:
            progress.update(task_id, completed=downloaded, total=total)

        try:
            bin_path = download_and_install(progress_callback=_cb)
        except Exception as exc:
            console.print(f"[red]Download failed: {exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Installed llama-server to {bin_path}[/green]")
    return str(bin_path)


def _verify_binary(bin_path: str) -> None:
    try:
        result = subprocess.run(
            [bin_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            console.print(f"[red]Binary check failed: {result.stderr.strip()}[/red]")
            raise typer.Exit(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        console.print(f"[red]Failed to run binary: {exc}[/red]")
        raise typer.Exit(1)


def setup() -> None:
    """Detect hardware, install llama-server if needed, and write ~/.locoder/config.toml."""
    console.print("[bold]Detecting hardware...[/bold]")
    hw = detect()

    table = Table(title="Hardware Detection", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CPU cores (physical)", str(hw.cpu_cores))
    table.add_row("RAM", f"{hw.ram_gb:.1f} GB")
    table.add_row(
        "VRAM",
        f"{hw.vram_gb:.1f} GB" if hw.vram_gb is not None else "None detected",
    )
    table.add_row("Inference mode", hw.mode)
    table.add_row("Model hint", hw.model_hint)
    table.add_row("Port (single)", str(hw.free_port_single))
    table.add_row("Port (planner)", str(hw.free_port_planner))
    table.add_row("Port (executor)", str(hw.free_port_executor))
    console.print(table)

    bin_path = _resolve_llama_server()
    _verify_binary(bin_path)

    write_config(hw, bin_path)

    console.print("\n[bold green]Setup complete.[/bold green]")
    console.print(f"  Mode        : {hw.mode}")
    console.print(f"  llama-server: {bin_path}")
    console.print(f"  Models dir  : ~/.locoder/models/")
    console.print(f"  Config      : {config_path()}")
