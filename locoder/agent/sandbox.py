from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_SUPPORTED: dict[str, str] = {
    "python": sys.executable,
    "bash": "/bin/sh",
    "sh": "/bin/sh",
}


def run_code(
    code: str,
    language: str,
    workspace: Path,
    timeout: int = 60,
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

    suffix = ".py" if lang == "python" else ".sh"

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as fh:
        fh.write(code)
        tmp = Path(fh.name)

    try:
        result = subprocess.run(
            [interpreter, str(tmp)],
            cwd=str(workspace),
            capture_output=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "exit_code": -1,
        }
    except OSError as exc:
        return {"stdout": "", "stderr": str(exc), "exit_code": -1}
    finally:
        tmp.unlink(missing_ok=True)
