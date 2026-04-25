from __future__ import annotations

import typer

from locoder import __version__
from locoder.cli.cmd_models import app as models_app
from locoder.cli.cmd_models import list_models, pull, remove, upgrade
from locoder.cli.cmd_registry import app as registry_app
from locoder.cli.cmd_setup import setup
from locoder.cli.cmd_start import start


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"locoder {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="locoder",
    help="Local-first coding agent powered by llama.cpp.",
    no_args_is_help=True,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


app.command()(setup)
app.command()(start)
app.add_typer(models_app, name="models")
app.add_typer(registry_app, name="registry")

# Convenience aliases at top level: pull, list, ls, remove, upgrade
app.command("pull")(pull)
app.command("list")(list_models)
app.command("ls")(list_models)
app.command("remove")(remove)
app.command("upgrade")(upgrade)
