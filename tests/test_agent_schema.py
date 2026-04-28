from __future__ import annotations

from locoder.agent.schema import Answer, ToolCall, parse_plan


def test_parse_plan_tool_call() -> None:
    data = {"action": "tool_call", "tool": "read_file", "arguments": {"path": "foo.py"}}
    result = parse_plan(data)
    assert isinstance(result, ToolCall)
    assert result.tool == "read_file"
    assert result.arguments == {"path": "foo.py"}


def test_parse_plan_answer() -> None:
    data = {"action": "answer", "content": "Done."}
    result = parse_plan(data)
    assert isinstance(result, Answer)
    assert result.content == "Done."


def test_parse_plan_defaults_to_tool_call_on_unknown_action() -> None:
    data = {"action": "unknown", "tool": "list_directory", "arguments": {}}
    result = parse_plan(data)
    assert isinstance(result, ToolCall)


def test_parse_plan_missing_arguments_defaults_to_empty() -> None:
    data = {"action": "tool_call", "tool": "list_directory"}
    result = parse_plan(data)
    assert isinstance(result, ToolCall)
    assert result.arguments == {}
