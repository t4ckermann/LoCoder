from __future__ import annotations

import typer
from rich.console import Console

from locoder.models.registry import refresh_registry

app = typer.Typer(help="Manage the model registry.")
console = Console()


def update() -> None:
    """Fetch the latest model registry from GitHub."""
    try:
        count = refresh_registry()
        console.print(f"[green]Registry updated: {count} models[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to update registry: {exc}[/red]")
        raise typer.Exit(1) from None


app.command("update")(update)
