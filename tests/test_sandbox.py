from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from locoder.agent.sandbox import run_code


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def console() -> MagicMock:
    return MagicMock()


class TestRunCode:
    def test_python_success(self, ws: Path, console: MagicMock) -> None:
        result = run_code("print('hello')", "python", ws, console=console)
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_unsupported_language_returns_error(self, ws: Path, console: MagicMock) -> None:
        result = run_code("", "ruby", ws, console=console)
        assert result["exit_code"] == 1
        assert "Unsupported" in result["stderr"]

    def test_nonzero_exit_code(self, ws: Path, console: MagicMock) -> None:
        result = run_code("import sys; sys.exit(42)", "python", ws, console=console)
        assert result["exit_code"] == 42

    def test_stderr_captured(self, ws: Path, console: MagicMock) -> None:
        result = run_code("import sys; sys.stderr.write('boom')", "python", ws, console=console)
        assert "boom" in result["stderr"]

    def test_stdout_and_stderr_both_captured(self, ws: Path, console: MagicMock) -> None:
        code = "import sys; print('out'); sys.stderr.write('err')"
        result = run_code(code, "python", ws, console=console)
        assert result["exit_code"] == 0
        assert "out" in result["stdout"]
        assert "err" in result["stderr"]

    def test_workspace_is_cwd(self, ws: Path, console: MagicMock) -> None:
        (ws / "marker.txt").write_text("present")
        result = run_code(
            "import os; print(os.path.exists('marker.txt'))", "python", ws, console=console
        )
        assert "True" in result["stdout"]

    def test_timeout_abort_on_user_choice(self, ws: Path, console: MagicMock) -> None:
        with patch("builtins.input", return_value="a"):
            result = run_code(
                "import time; time.sleep(100)",
                "python",
                ws,
                timeout=1,
                console=console,
            )
        assert result["exit_code"] == -1
        assert "aborted" in result["stderr"].lower()

    def test_timeout_wait_once_then_abort(self, ws: Path, console: MagicMock) -> None:
        responses = iter(["w", "a"])
        with patch("builtins.input", side_effect=responses):
            result = run_code(
                "import time; time.sleep(100)",
                "python",
                ws,
                timeout=1,
                console=console,
            )
        assert result["exit_code"] == -1
        assert "aborted" in result["stderr"].lower()

    def test_max_extensions_auto_abort(self, ws: Path, console: MagicMock) -> None:
        """After max_extensions is exhausted the process is killed automatically."""
        with patch("builtins.input", return_value="w"):
            result = run_code(
                "import time; time.sleep(100)",
                "python",
                ws,
                timeout=1,
                max_extensions=1,
                console=console,
            )
        assert result["exit_code"] == -1
        assert "maximum extensions" in result["stderr"]

    def test_allow_network_true_skips_advisory(self, ws: Path, console: MagicMock) -> None:
        run_code("print(1)", "python", ws, allow_network=True, console=console)
        # advisory message must NOT appear when network is explicitly allowed
        for call_args in console.print.call_args_list:
            text = str(call_args)
            assert "advisory" not in text.lower()
