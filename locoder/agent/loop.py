from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from locoder.agent.graph import run_agent
from locoder.models.client import active_model_name, supports_thinking
from locoder.server.launcher import ServerHandle

_HELP_TEXT = """
Available slash commands:
  /help    — show this message
  /status  — show model, mode, and thinking mode
  /think   — toggle deep thinking mode on/off for the current session
  /clear   — not yet supported (restart session to clear context)
  Ctrl-C   — stop servers and exit
""".strip()


def _print_status(
    config: dict[str, Any],
    handles: list[ServerHandle],
    console: Console,
    thinking_enabled: bool,
) -> None:
    mode = config["inference"]["mode"]
    console.print(f"[bold]Mode:[/bold] {mode}")
    for h in handles:
        console.print(f"[bold]{h.role}:[/bold] {active_model_name(config, h.role)}  port={h.port}")
    state = "[green]on[/green]" if thinking_enabled else "[dim]off[/dim]"
    console.print(f"[bold]Thinking:[/bold] {state}")


def interactive_loop(
    config: dict[str, Any],
    handles: list[ServerHandle],
    workspace: Path,
    console: Console,
) -> None:
    """Read-eval-print loop: accept tasks, run the agent, repeat."""
    console.print("\n[bold green]LoCoder ready.[/bold green] Type a task or [dim]/help[/dim].\n")

    model = active_model_name(config)
    thinking_enabled: bool = bool(config.get("agent", {}).get("thinking_mode", False))

    while True:
        try:
            task = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise

        if not task:
            continue

        if task == "/help":
            console.print(_HELP_TEXT)
            continue

        if task == "/status":
            _print_status(config, handles, console, thinking_enabled)
            continue

        if task == "/think":
            if not supports_thinking(model):
                console.print(f"[yellow]Thinking mode is not supported by {model!r}.[/yellow]")
            else:
                thinking_enabled = not thinking_enabled
                state = "[green]on[/green]" if thinking_enabled else "[dim]off[/dim]"
                console.print(f"Thinking mode: {state}")
            continue

        if task.startswith("/"):
            console.print(f"[yellow]Unknown command: {task!r}. Type /help.[/yellow]")
            continue

        try:
            run_agent(task, config, workspace, console, thinking_enabled)
        except Exception as exc:
            console.print(f"[red]Agent error: {exc}[/red]")

        console.print()
