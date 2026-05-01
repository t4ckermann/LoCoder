from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from locoder.agent.loop import interactive_loop
from locoder.config.manager import read_config
from locoder.hardware.detect import available_gb as _available_gb
from locoder.models.downloader import download, is_installed
from locoder.models.registry import lookup
from locoder.server.launcher import ServerHandle, start_server, start_servers_dual, stop_server

console = Console()


def _lan_ip() -> str | None:
    """Return the machine's LAN IP by probing a remote address (no packet sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except OSError:
        return None


def _print_ready(handle: ServerHandle) -> None:
    if handle.host == "0.0.0.0":
        console.print(f"[bold green]Server ready — all interfaces, port {handle.port}[/bold green]")
        console.print(f"  Local:  http://127.0.0.1:{handle.port}")
        lan = _lan_ip()
        if lan:
            console.print(f"  LAN:    http://{lan}:{handle.port}")
    else:
        console.print(
            f"[bold green]Server ready at http://{handle.host}:{handle.port}[/bold green]"
        )


def _ensure_installed(model_name: str) -> None:
    """Check model is installed; prompt to download if not. Raises typer.Exit on failure."""
    if is_installed(model_name):
        return
    entry = lookup(model_name)
    size_hint = f" ({entry['ram_tier']} model)" if entry else ""
    console.print(
        f"[yellow]Configured model [bold]{model_name}[/bold]{size_hint} is not installed.[/yellow]"
    )
    console.print(f"[dim]Tip: run `locoder pull {model_name}` to download it separately.[/dim]")
    if not typer.confirm(f"Download '{model_name}' now?", default=True):
        raise typer.Exit(0)
    try:
        download(model_name, available_gb=_available_gb())
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


def start(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Override bind address (e.g. 0.0.0.0 to expose on LAN).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Override server port (single mode only).",
    ),
) -> None:
    """Start the llama-server and agent loop."""
    try:
        config: dict[str, Any] = read_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    if host is not None:
        config["inference"]["host"] = host
    if port is not None:
        config["inference"]["single"]["port"] = port

    mode: str = config["inference"].get("mode", "single")
    handles: list[ServerHandle] = []

    try:
        if mode == "dual":
            dual = config["inference"]["dual"]
            _ensure_installed(str(dual["planner"]["model"]))
            _ensure_installed(str(dual["executor"]["model"]))
            planner_h, executor_h = start_servers_dual(config)
            handles = [planner_h, executor_h]
            primary = planner_h
        else:
            _ensure_installed(str(config["inference"]["single"]["model"]))
            single_h = start_server(config)
            handles = [single_h]
            primary = single_h
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    for h in handles:
        _print_ready(h)

    workspace = Path.cwd()

    try:
        interactive_loop(config, primary, workspace, console)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        for h in handles:
            stop_server(h)
