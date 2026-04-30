from __future__ import annotations

from pathlib import Path

from locoder.agent.schema import TOOL_NAMES

_TOOLS_DOC = """
read_file(path)               — read a file; accepts absolute paths or relative to workspace root
write_file(path, content)     — write (or overwrite) a file; creates parent dirs; absolute paths ok
run_code(code, language)      — execute code ("python", "bash"); returns stdout/stderr/exit_code
list_directory(path)          — list directory entries; absolute paths or relative to workspace root
search_knowledge_base(query)  — semantic search over indexed workspace; prefer for concept queries
search_codebase(query, path)  — exact substring search; absolute or relative path
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

_REVIEWER_FORMAT = """
Respond ONLY with a JSON object.
  {"verdict": "approved", "reason": "<brief>"}
  {"verdict": "revise",   "feedback": "<specific issues and what to fix>"}
""".strip()


def build_system_prompt(workspace: Path, thinking_prefix: str = "") -> str:
    prefix = f"{thinking_prefix}\n" if thinking_prefix else ""
    return (
        f"{prefix}"
        f"You are LoCoder, an expert software engineering agent running on the user's machine.\n"
        f"Workspace root: {workspace}\n"
        f"Use absolute paths to read or write files outside the workspace.\n\n"
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


def build_reviewer_prompt(task: str, answer: str, written_files: list[str]) -> str:
    files_str = "\n".join(written_files) if written_files else "(none)"
    return (
        f"You are reviewing a coding agent's completed work.\n\n"
        f"Task:\n{task}\n\n"
        f"Files written:\n{files_str}\n\n"
        f"Agent's final answer:\n{answer}\n\n"
        f"Is the work complete and correct? "
        f"Check for missing pieces, incorrect logic, or incomplete implementation.\n\n"
        f"{_REVIEWER_FORMAT}"
    )
