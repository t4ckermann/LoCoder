from __future__ import annotations

import contextlib
import shlex
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console


def run_verify(
    written: list[str],
    workspace: Path,
    console: Console,
    verify_config: dict[str, Any],
) -> None:
    """Run configured checks (lint, type-check, tests, manual) on files written during the run."""
    if not written:
        return

    py_files = [f for f in written if f.endswith(".py")]
    abs_py_files = [str(workspace / f) if not Path(f).is_absolute() else f for f in py_files]

    if abs_py_files and verify_config.get("lint", True):
        console.print("[dim][verify] ruff check...[/dim]")
        r = subprocess.run(
            ["ruff", "check", "--fix", *abs_py_files],
            capture_output=True,
            cwd=str(workspace),
        )
        if r.returncode != 0:
            issues = r.stdout.decode(errors="replace")
            console.print(f"[yellow][verify] ruff issues:\n{issues}[/yellow]")

    if abs_py_files and verify_config.get("type_check", True):
        console.print("[dim][verify] mypy...[/dim]")
        r = subprocess.run(
            ["mypy", *abs_py_files],
            capture_output=True,
            cwd=str(workspace),
        )
        if r.returncode != 0:
            issues = r.stdout.decode(errors="replace")
            console.print(f"[yellow][verify] mypy issues:\n{issues}[/yellow]")

    if verify_config.get("tests", False):
        test_cmd_str = str(verify_config.get("test_command", "pytest"))
        console.print(f"[dim][verify] {test_cmd_str}...[/dim]")
        r = subprocess.run(
            shlex.split(test_cmd_str),
            capture_output=True,
            cwd=str(workspace),
        )
        if r.returncode != 0:
            out = (r.stdout + r.stderr).decode(errors="replace")
            console.print(f"[yellow][verify] test failures:\n{out}[/yellow]")
        else:
            console.print("[green][verify] all tests passed[/green]")

    if verify_config.get("manual", False):
        console.print("\n[bold yellow][verify] Manual review requested.[/bold yellow]")
        console.print("[dim]Review the changes above, then press Enter to continue...[/dim]")
        with contextlib.suppress(EOFError):
            input()
