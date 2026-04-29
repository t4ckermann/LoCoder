from __future__ import annotations

from pathlib import Path

from locoder.agent.schema import TOOL_NAMES

_TOOLS_DOC = """
read_file(path)               — read a file relative to the workspace root
write_file(path, content)     — write (or overwrite) a file; creates parent dirs
run_code(code, language)      — execute code ("python", "bash"); returns stdout/stderr/exit_code
list_directory(path)          — list entries in a directory
search_knowledge_base(query)  — semantic search over indexed codebase; prefer for concept/intent
                                queries when the index is current (no stale-file warning)
search_codebase(query, path)  — exact substring search; use for known symbols or when RAG is stale
""".strip()

_PLAN_FORMAT = """
Respond ONLY with a JSON object. No prose, no markdown fences.

To call a tool:
  {"action": "tool_call", "tool": "<name>", "arguments": {<key>: <value>, ...}}

To give a final answer:
  {"action": "answer", "content": "<your response to the user>"}

Available tools: """ + ", ".join(TOOL_NAMES)

_CLARIFY_FORMAT = """
Respond ONLY with a JSON object listing your assumptions:
  {"assumptions": ["<assumption 1>", "<assumption 2>", ...]}
""".strip()


def build_system_prompt(workspace: Path, thinking_prefix: str = "") -> str:
    prefix = f"{thinking_prefix}\n" if thinking_prefix else ""
    return (
        f"{prefix}"
        f"You are LoCoder, an expert software engineering agent running on the user's machine.\n"
        f"Workspace root: {workspace}\n\n"
        f"Available tools:\n{_TOOLS_DOC}\n\n"
        f"{_PLAN_FORMAT}"
    )


def build_clarify_prompt(task: str) -> str:
    return (
        f"The user asked: {task!r}\n\n"
        f"Before starting, state the key assumptions you will proceed with "
        f"(scope, approach, edge cases). Keep each assumption to one sentence.\n\n"
        f"{_CLARIFY_FORMAT}"
    )
