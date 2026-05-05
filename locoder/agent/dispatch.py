from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from locoder.agent import sandbox, tools
from locoder.agent.schema import ToolCall

_MAX_OBS_CHARS = 8_000  # cap tool results appended to messages (~2k tokens)


def dispatch(
    call: ToolCall,
    workspace: Path,
    config: dict[str, Any],
    console: Console,
) -> str:
    """Execute a ToolCall and return its string result."""
    name = call.tool
    args = call.arguments
    if name == "read_file":
        return tools.read_file(str(args.get("path", "")), workspace)
    if name == "write_file":
        return tools.write_file(str(args.get("path", "")), str(args.get("content", "")), workspace)
    if name == "run_code":
        sb_cfg = config.get("sandbox", {})
        result = sandbox.run_code(
            str(args.get("code", "")),
            str(args.get("language", "python")),
            workspace,
            timeout=int(sb_cfg.get("execution_timeout", 60)),
            max_extensions=int(sb_cfg.get("max_extensions", 10)),
            allow_network=bool(sb_cfg.get("allow_network", False)),
            console=console,
        )
        return (
            f"exit_code: {result['exit_code']}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}"
        ).strip()
    if name == "list_directory":
        return tools.list_directory(str(args.get("path", ".")), workspace)
    if name == "search_codebase":
        return tools.search_codebase(
            str(args.get("query", "")),
            str(args.get("path", ".")),
            workspace,
        )
    if name == "search_knowledge_base":
        return tools.search_knowledge_base(str(args.get("query", "")), workspace, config)
    return f"Unknown tool: {name!r}"


def fmt_args(args: dict[str, Any]) -> str:
    """Format tool arguments for display, truncating long values."""
    parts: list[str] = []
    for k, v in args.items():
        s = repr(v)
        parts.append(f"{k}={s[:60]}{'...' if len(s) > 60 else ''}")
    return ", ".join(parts)
