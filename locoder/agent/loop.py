from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from rich.console import Console

from locoder.agent import history, rag
from locoder.agent.graph import run_agent
from locoder.models.client import (
    active_model_name,
    executor_model_name,
    planner_model_name,
    supports_thinking,
)
from locoder.server.launcher import ServerHandle

_HELP_TEXT = """
Available slash commands:
  /help     — show this message
  /status   — show model, mode, and thinking mode
  /think    — toggle deep thinking mode on/off for the current session
  /reindex  — re-index the workspace into the knowledge base
  /history  — show the last 5 task summaries from this session
  /clear    — clear persistent conversation history for this workspace
  Ctrl-C    — stop servers and exit
""".strip()


def _print_status(
    config: dict[str, Any],
    handle: ServerHandle,
    console: Console,
    thinking_enabled: bool,
) -> None:
    inf = config.get("inference", {})
    if inf.get("mode", "single") == "dual":
        dual = inf["dual"]
        console.print(
            f"[bold]Planner:[/bold] {planner_model_name(config)}  port={dual['planner']['port']}"
        )
        console.print(
            f"[bold]Executor:[/bold] {executor_model_name(config)}  port={dual['executor']['port']}"
        )
    else:
        console.print(f"[bold]Model:[/bold] {active_model_name(config)}  port={handle.port}")
    state = "[green]on[/green]" if thinking_enabled else "[dim]off[/dim]"
    console.print(f"[bold]Thinking:[/bold] {state}")


def _spawn_index(workspace: Path, config: dict[str, Any], console: Console) -> threading.Thread:
    """Start indexing in a daemon thread and return it."""

    def _run() -> None:
        try:
            rag.index_workspace(workspace, config, console)
        except Exception as exc:
            console.print(f"[dim][rag] Indexing error: {exc}[/dim]")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def interactive_loop(
    config: dict[str, Any],
    handle: ServerHandle,
    workspace: Path,
    console: Console,
) -> None:
    """Read-eval-print loop: accept tasks, run the agent, repeat."""
    console.print("\n[bold green]LoCoder ready.[/bold green] Type a task or [dim]/help[/dim].\n")
    console.print("[dim]Type /reindex to build the knowledge base before your first task.[/dim]\n")

    index_thread = threading.Thread(daemon=True)  # placeholder; replaced on /reindex

    model = planner_model_name(config)
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
            _print_status(config, handle, console, thinking_enabled)
            continue

        if task == "/think":
            if not supports_thinking(model):
                console.print(f"[yellow]Thinking mode is not supported by {model!r}.[/yellow]")
            else:
                thinking_enabled = not thinking_enabled
                state = "[green]on[/green]" if thinking_enabled else "[dim]off[/dim]"
                console.print(f"Thinking mode: {state}")
            continue

        if task == "/reindex":
            if index_thread.is_alive():
                console.print("[dim]Indexing already in progress...[/dim]")
            else:
                index_thread = _spawn_index(workspace, config, console)
            continue

        if task == "/history":
            summaries = history.recent_summaries(workspace)
            if summaries:
                for i, s in enumerate(summaries, 1):
                    console.print(f"  {i}. {s}")
            else:
                console.print("[dim]No history yet.[/dim]")
            continue

        if task == "/clear":
            history.clear(workspace)
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if task.startswith("/"):
            console.print(f"[yellow]Unknown command: {task!r}. Type /help.[/yellow]")
            continue

        try:
            run_agent(task, config, workspace, console, thinking_enabled)
        except Exception as exc:
            console.print(f"[red]Agent error: {exc}[/red]")

        console.print()
