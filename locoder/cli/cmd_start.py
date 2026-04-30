from __future__ import annotations

import socket
from pathlib import Path
from typing import Any  # noqa: F401 — used in config dict annotation below

import typer
from rich.console import Console

from locoder.agent.loop import interactive_loop
from locoder.config.manager import read_config
from locoder.hardware.detect import available_gb as _available_gb
from locoder.models.downloader import download, is_installed
from locoder.models.registry import lookup
from locoder.server.launcher import ServerHandle, start_server, stop_server

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


def start(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Override bind address (e.g. 0.0.0.0 to expose on LAN).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Override server port.",
    ),
) -> None:
    """Start the llama-server and agent loop."""
    try:
        config: dict[str, Any] = read_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    # Apply CLI overrides to the in-memory config so all subsystems see them.
    if host is not None:
        config["inference"]["host"] = host
    if port is not None:
        config["inference"]["single"]["port"] = port

    model_name: str = config["inference"]["single"]["model"]
    if not is_installed(model_name):
        entry = lookup(model_name)
        size_hint = f" ({entry['ram_tier']} model)" if entry else ""
        console.print(
            f"[yellow]Configured model [bold]{model_name}[/bold]{size_hint} "
            f"is not installed.[/yellow]"
        )
        console.print(
            f"[dim]Tip: edit ~/.locoder/config.toml to change the model, "
            f"or run `locoder pull {model_name}` to download it separately.[/dim]"
        )
        if not typer.confirm(f"Download '{model_name}' now?", default=True):
            raise typer.Exit(0)
        try:
            download(model_name, available_gb=_available_gb())
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None

    handle: ServerHandle
    try:
        handle = start_server(config)
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    _print_ready(handle)

    workspace = Path.cwd()

    try:
        interactive_loop(config, handle, workspace, console)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_server(handle)
