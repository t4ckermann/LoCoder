from __future__ import annotations

import typer
from rich.console import Console

from locoder.config.manager import read_config
from locoder.models.downloader import download, is_installed
from locoder.server.launcher import start_server, stop_servers

console = Console()


def start() -> None:
    """Start the llama-server and agent loop."""
    try:
        config = read_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    mode: str = config["inference"]["mode"]

    if mode == "single":
        required_models = [config["inference"]["single"]["model"]]
    elif mode == "hierarchical":
        required_models = [
            config["inference"]["hierarchical"]["planner_model"],
            config["inference"]["hierarchical"]["executor_model"],
        ]
    else:
        console.print(f"[red]Unknown mode in config: {mode!r}[/red]")
        raise typer.Exit(1)

    from locoder.models.registry import lookup

    for model_name in required_models:
        if not is_installed(model_name):
            entry = lookup(model_name)
            size_hint = f" ({entry['ram_tier']} model)" if entry else ""
            console.print(
                f"[yellow]Configured model [bold]{model_name}[/bold]{size_hint} is not installed.[/yellow]"
            )
            console.print(
                f"[dim]Tip: edit ~/.locoder/config.toml to change the model, "
                f"or run `locoder pull {model_name}` to download it separately.[/dim]"
            )
            if not typer.confirm(f"Download '{model_name}' now?", default=True):
                raise typer.Exit(0)
            try:
                download(model_name)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1)

    try:
        handles = start_server(mode, config)
    except (RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    for handle in handles:
        console.print(
            f"[bold green]Server ready ({handle.role}) at "
            f"http://127.0.0.1:{handle.port}[/bold green]"
        )

    # TODO: launch agent loop (Phase 3)
    console.print("[dim]Agent loop not yet implemented. Press Ctrl-C to stop servers.[/dim]")

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_servers(handles)
