from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from locoder.models.downloader import is_installed
from locoder.models.registry import load_registry, refresh_registry

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


def list_registry() -> None:
    """List all models available in the registry."""
    reg = load_registry()
    table = Table(title="Model Registry", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("RAM tier", style="yellow", justify="center")
    table.add_column("Params", style="white", justify="right")
    table.add_column("Default quant", style="white")
    table.add_column("Installed", style="green", justify="center")

    for name, entry in sorted(reg.items()):
        params = f"{entry.get('params_b', '?')}B"
        installed = "✓" if is_installed(name) else ""
        table.add_row(
            name,
            str(entry.get("ram_tier", "?")),
            params,
            str(entry.get("default_quant", "?")),
            installed,
        )

    console.print(table)


app.command("update")(update)
app.command("list")(list_registry)
app.command("ls")(list_registry)
