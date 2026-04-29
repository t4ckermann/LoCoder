from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from locoder.agent.loop import interactive_loop
from locoder.config.manager import read_config
from locoder.hardware.detect import available_gb as _available_gb
from locoder.models.downloader import download, is_installed
from locoder.models.registry import lookup
from locoder.server.launcher import start_server, stop_servers

console = Console()


def _required_models(mode: str, config: dict[str, Any]) -> list[str]:
    if mode == "single":
        return [config["inference"]["single"]["model"]]
    if mode == "hierarchical":
        return [
            config["inference"]["hierarchical"]["planner_model"],
            config["inference"]["hierarchical"]["executor_model"],
        ]
    console.print(f"[red]Unknown mode in config: {mode!r}[/red]")
    raise typer.Exit(1) from None


def start() -> None:
    """Start the llama-server and agent loop."""
    try:
        config = read_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    mode: str = config["inference"]["mode"]

    for model_name in _required_models(mode, config):
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

    try:
        handles = start_server(mode, config)
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    for handle in handles:
        console.print(
            f"[bold green]Server ready ({handle.role}) at "
            f"http://127.0.0.1:{handle.port}[/bold green]"
        )

    workspace = Path.cwd()

    try:
        interactive_loop(config, handles, workspace, console)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_servers(handles)
