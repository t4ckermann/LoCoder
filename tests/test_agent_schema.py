from __future__ import annotations

from locoder.agent.schema import Answer, Review, ToolCall, parse_plan, parse_review


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


def test_parse_review_approved() -> None:
    data = {"verdict": "approved", "reason": "looks good"}
    result = parse_review(data)
    assert isinstance(result, Review)
    assert result.verdict == "approved"
    assert result.reason == "looks good"


def test_parse_review_revise() -> None:
    data = {"verdict": "revise", "feedback": "missing error handling in the main function"}
    result = parse_review(data)
    assert isinstance(result, Review)
    assert result.verdict == "revise"
    assert result.feedback == "missing error handling in the main function"


def test_parse_review_defaults_to_approved_on_empty_data() -> None:
    result = parse_review({})
    assert result.verdict == "approved"
    assert result.reason == ""
    assert result.feedback == ""


def test_parse_review_unknown_verdict_preserved() -> None:
    result = parse_review({"verdict": "maybe"})
    assert result.verdict == "maybe"
