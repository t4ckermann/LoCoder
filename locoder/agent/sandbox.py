from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from rich.console import Console

_SUPPORTED: dict[str, str] = {
    "python": sys.executable,
    "bash": "/bin/sh",
    "sh": "/bin/sh",
}

_GRACE_PERIOD = 5  # seconds between SIGTERM and SIGKILL


def _setrlimits() -> None:
    """Apply resource caps in the forked child; silently no-ops on Windows or permission errors."""
    try:
        import resource  # noqa: PLC0415 — Unix-only, imported lazily to avoid Windows failure

        for name, limit in (
            ("RLIMIT_FSIZE", 64 * 1024 * 1024),  # max file write: 64 MB
            ("RLIMIT_NPROC", 64),  # max child processes: fork-bomb guard
        ):
            rlimit = getattr(resource, name, None)
            if rlimit is not None:
                with contextlib.suppress(ValueError, OSError):
                    resource.setrlimit(rlimit, (limit, limit))
    except ImportError:
        pass


def _kill_process(proc: subprocess.Popen[bytes]) -> None:
    """Send SIGTERM, then SIGKILL after _GRACE_PERIOD if still alive."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_GRACE_PERIOD)
    except subprocess.TimeoutExpired:
        proc.kill()


def _kill_and_collect(proc: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    """Send SIGTERM→SIGKILL, then return whatever output was buffered."""
    _kill_process(proc)
    try:
        return proc.communicate()
    except Exception:  # noqa: BLE001
        return b"", b""


def _prompt_wait_or_abort(
    elapsed: int,
    extensions: int,
    max_extensions: int,
    console: Console,
) -> bool:
    """Prompt the user to extend or abort. Return True to keep waiting, False to abort."""
    console.print(f"\n[yellow]Code execution is taking longer than expected ({elapsed}s).[/yellow]")
    if max_extensions > 0:
        remaining = max_extensions - extensions
        console.print(
            f"  [bold][w][/bold] Wait another {elapsed}s  (extensions remaining: {remaining})"
        )
    else:
        console.print(f"  [bold][w][/bold] Wait another {elapsed}s")
    console.print("  [bold][a][/bold] Abort — let the agent try a different approach")
    try:
        choice = input("Choice [w/a]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return choice != "a"


def _build_cmd(interpreter: str, script: str, allow_network: bool) -> list[str]:
    """Return subprocess argv, wrapping with `unshare --net` on Linux when network is off."""
    base: list[str] = [interpreter, script]
    if allow_network or sys.platform != "linux" or shutil.which("unshare") is None:
        return base
    return ["unshare", "--net", *base]


def run_code(
    code: str,
    language: str,
    workspace: Path,
    *,
    timeout: int = 60,
    max_extensions: int = 10,
    allow_network: bool = False,
    console: Console,
) -> dict[str, Any]:
    """Execute *code* in a sandboxed subprocess inside *workspace*.

    Returns {"stdout": str, "stderr": str, "exit_code": int}.
    """
    lang = language.lower()
    interpreter = _SUPPORTED.get(lang)
    if interpreter is None:
        return {
            "stdout": "",
            "stderr": f"Unsupported language: {language!r}. Supported: {list(_SUPPORTED)}",
            "exit_code": 1,
        }

    network_enforced = (
        not allow_network and sys.platform == "linux" and shutil.which("unshare") is not None
    )
    if not allow_network and not network_enforced:
        console.print(
            "[dim][sandbox] Network isolation is advisory on this platform — "
            "use Docker for hard enforcement.[/dim]"
        )

    suffix = ".py" if lang == "python" else ".sh"
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as fh:
            fh.write(code)
            tmp = Path(fh.name)

        cmd = _build_cmd(interpreter, str(tmp), allow_network)
        popen_kwargs: dict[str, Any] = {
            "cwd": str(workspace),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _setrlimits

        proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603
        start = time.monotonic()
        extensions = 0

        while True:
            try:
                out_bytes, err_bytes = proc.communicate(timeout=timeout)
                return {
                    "stdout": out_bytes.decode(errors="replace"),
                    "stderr": err_bytes.decode(errors="replace"),
                    "exit_code": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - start)

                if max_extensions > 0 and extensions >= max_extensions:
                    out_bytes, err_bytes = _kill_and_collect(proc)
                    return {
                        "stdout": out_bytes.decode(errors="replace"),
                        "stderr": (
                            f"Execution aborted: maximum extensions ({max_extensions}) reached"
                            f" after {elapsed}s\n{err_bytes.decode(errors='replace')}"
                        ).strip(),
                        "exit_code": -1,
                    }

                if not _prompt_wait_or_abort(elapsed, extensions, max_extensions, console):
                    out_bytes, err_bytes = _kill_and_collect(proc)
                    return {
                        "stdout": out_bytes.decode(errors="replace"),
                        "stderr": (
                            f"Execution aborted by user after {elapsed}s\n"
                            f"{err_bytes.decode(errors='replace')}"
                        ).strip(),
                        "exit_code": -1,
                    }
                extensions += 1

    except OSError as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": -1}
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
