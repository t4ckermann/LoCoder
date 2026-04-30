from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Names of all callable tools — must stay in sync with tools.py and the system prompt.
TOOL_NAMES: list[str] = [
    "read_file",
    "write_file",
    "run_code",
    "list_directory",
    "search_codebase",
    "search_knowledge_base",
]

# JSON Schema for clarification response.
CLARIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["assumptions"],
    "additionalProperties": False,
}

# JSON Schema for plan-step response.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool_call", "answer"]},
        "tool": {"type": "string", "enum": TOOL_NAMES},
        "arguments": {"type": "object"},
        "content": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Answer:
    content: str


@dataclass(frozen=True)
class Review:
    verdict: str  # "approved" | "revise"
    reason: str = ""
    feedback: str = ""


def parse_plan(data: dict[str, Any]) -> ToolCall | Answer:
    """Parse a model response dict into a ToolCall or Answer."""
    if data.get("action") == "answer":
        return Answer(content=str(data.get("content", "")))
    return ToolCall(
        tool=str(data.get("tool", "")),
        arguments=dict(data.get("arguments") or {}),
    )


def parse_review(data: dict[str, Any]) -> Review:
    """Parse a model response dict into a Review. Defaults to approved on missing/invalid data."""
    return Review(
        verdict=str(data.get("verdict", "approved")),
        reason=str(data.get("reason", "")),
        feedback=str(data.get("feedback", "")),
    )
