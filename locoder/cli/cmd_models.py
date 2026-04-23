from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from locoder.models.downloader import download as _download
from locoder.models.downloader import model_dir, remove as _remove

app = typer.Typer(help="Manage local models.")
console = Console()

_MODELS_DIR = Path("~/.locoder/models").expanduser()


def pull(
    model: str = typer.Argument(..., help="Model name from registry (e.g. qwen2.5-coder-7b)"),
    quant: str | None = typer.Option(None, "--quant", "-q", help="Quantization level (e.g. q4_k_m)"),
) -> None:
    """Download a model from HuggingFace."""
    try:
        path = _download(model, quant)
        console.print(f"[green]Downloaded to {path}[/green]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def list_models() -> None:
    """List locally installed models."""
    if not _MODELS_DIR.exists():
        console.print("[dim]No models installed yet. Run `locoder pull <model>`.[/dim]")
        return

    table = Table(title="Installed Models", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Size", style="green", justify="right")

    found = False
    for d in sorted(_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        for gguf in d.glob("*.gguf"):
            size_mb = gguf.stat().st_size / (1024 * 1024)
            table.add_row(d.name, gguf.name, f"{size_mb:.1f} MB")
            found = True

    if found:
        console.print(table)
    else:
        console.print("[dim]No models installed yet. Run `locoder pull <model>`.[/dim]")


def remove(
    model: str = typer.Argument(..., help="Model name to remove"),
) -> None:
    """Remove a locally installed model."""
    typer.confirm(f"Remove model '{model}'?", abort=True)
    try:
        _remove(model)
        console.print(f"[green]Removed '{model}'.[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def upgrade(
    old_model: str = typer.Argument(..., help="Model to replace (e.g. qwen2.5-coder-1.5b)"),
    new_model: str = typer.Argument(..., help="Model to download (e.g. gemma4-26b)"),
    quant: str | None = typer.Option(None, "--quant", "-q", help="Quantization for the new model"),
) -> None:
    """Download a better model, then offer to remove the old one."""
    from locoder.models.downloader import is_installed

    console.print(f"[bold]Downloading [cyan]{new_model}[/cyan]...[/bold]")
    try:
        path = _download(new_model, quant)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Downloaded to {path}[/green]")

    if not is_installed(old_model):
        console.print(f"[dim]'{old_model}' is not installed — nothing to remove.[/dim]")
        return

    if typer.confirm(f"\nRemove '{old_model}' to free up disk space?", default=False):
        try:
            _remove(old_model)
            console.print(f"[green]Removed '{old_model}'.[/green]")
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(
        f"\n[yellow]Remember to update ~/.locoder/config.toml:[/yellow]\n"
        f"  model = \"{new_model}\""
    )


# Sub-app commands (when accessed via `locoder models ...`)
app.command("pull")(pull)
app.command("list")(list_models)
app.command("ls")(list_models)
app.command("remove")(remove)
app.command("upgrade")(upgrade)
