from __future__ import annotations

import contextlib
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, StateGraph
from openai import InternalServerError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from typing_extensions import TypedDict

from locoder.agent import history, prompts, sandbox, tools
from locoder.agent.schema import Answer, Review, ToolCall, parse_plan, parse_review
from locoder.models.client import (
    active_model_name,
    executor_model_name,
    get_executor_client,
    get_planner_client,
    get_sync_client,
    planner_model_name,
    thinking_prefix,
)

_MAX_ITERATIONS = 30
_MAX_REVIEWS = 2
_MAX_OBS_CHARS = 8_000  # cap tool results appended to messages (~2k tokens)
_MAX_CONTEXT_MESSAGES = 40  # system + first task + last N; prevents KV-cache RAM blowup


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    task: str
    iterations: int
    written_files: list[str]
    done: bool
    answer: str
    pending_tool: dict[str, Any]  # {"tool": str, "arguments": dict} when a call is pending
    last_observation: str  # output from the last tool dispatch
    review_count: int  # number of reviewer cycles completed


def _trim_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep system msg + first user task + most recent exchanges to stay within context window."""
    if len(messages) <= _MAX_CONTEXT_MESSAGES:
        return messages
    head = messages[:2]  # system prompt + first user task
    tail = messages[-(_MAX_CONTEXT_MESSAGES - 2) :]
    return head + tail


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output (Qwen3/DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction; falls back to wrapping plain text as an answer."""
    with contextlib.suppress(json.JSONDecodeError):
        return cast(dict[str, Any], json.loads(text))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            return cast(dict[str, Any], json.loads(text[start : end + 1]))
    return {"action": "answer", "content": text}


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call llama-server and return a parsed dict.

    Uses plain-text completion (no JSON grammar enforcement) so that models which
    emit <think>…</think> blocks before JSON don't fail.  Strips thinking blocks
    before attempting to parse, and falls back to a best-effort JSON extraction.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=0.2,
            max_tokens=4096,
        )
    except InternalServerError as exc:
        raise RuntimeError(
            "Inference server returned 500. If this keeps happening try "
            'flash_attn = "off" or a smaller ctx_size in .locoder.toml, then restart.'
        ) from exc
    raw = _strip_thinking((resp.choices[0].message.content or "{}").strip())
    return _extract_json(raw)


def _dispatch(
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


def _verify_written_files(
    written: list[str],
    workspace: Path,
    console: Console,
    verify_config: dict[str, Any],
) -> None:
    """Run configured checks on files written during the agent run."""
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


def make_graph(
    config: dict[str, Any],
    workspace: Path,
    console: Console,
    thinking_mode: bool | None = None,
) -> Any:
    """Build and compile the agent LangGraph state machine."""
    dual_mode: bool = config.get("inference", {}).get("mode", "single") == "dual"

    if dual_mode:
        p_client = get_planner_client(config)
        e_client = get_executor_client(config)
        p_model = planner_model_name(config)
        e_model = executor_model_name(config)
    else:
        p_client = get_sync_client(config)
        e_client = p_client
        p_model = active_model_name(config)
        e_model = p_model

    if thinking_mode is None:
        thinking_enabled: bool = bool(config.get("agent", {}).get("thinking_mode", False))
    else:
        thinking_enabled = thinking_mode

    p_prefix = thinking_prefix(p_model, thinking_enabled)
    e_prefix = thinking_prefix(e_model, thinking_enabled)
    reviewer_system = (
        "[REVIEWER]\n"
        "You are a code review agent. You review work produced by a coding agent and decide "
        "whether it satisfies the user's request. Be concise and specific in your feedback."
    )
    reviewer_enabled: bool = bool(config.get("agent", {}).get("reviewer_enabled", False))
    verify_config: dict[str, Any] = config.get("verify", {})

    # /think-style prefixes (Qwen3) must appear in the last user message, not the system prompt.
    # <|think|>-style prefixes (Gemma4) go in the system prompt as-is.
    p_user_suffix = p_prefix if p_prefix.startswith("/") else ""
    e_user_suffix = e_prefix if e_prefix.startswith("/") else ""
    p_sys_prefix = "" if p_prefix.startswith("/") else p_prefix
    e_sys_prefix = "" if e_prefix.startswith("/") else e_prefix
    planner_system = "[PLANNER]\n" + prompts.build_system_prompt(workspace, p_sys_prefix)
    executor_system = "[EXECUTOR]\n" + prompts.build_system_prompt(workspace, e_sys_prefix)

    def _with_system(
        msgs: list[dict[str, Any]], system: str, user_suffix: str = ""
    ) -> list[dict[str, Any]]:
        if msgs and msgs[0].get("role") == "system":
            result: list[dict[str, Any]] = [{"role": "system", "content": system}, *msgs[1:]]
        else:
            result = [{"role": "system", "content": system}, *msgs]
        if user_suffix:
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("role") == "user":
                    result[i] = {**result[i], "content": f"{result[i]['content']}\n{user_suffix}"}
                    break
        return result

    def invoke_planner(messages: list[dict[str, Any]]) -> dict[str, Any]:
        prepared = _trim_context(_with_system(messages, planner_system, p_user_suffix))
        return _call_llm(p_client, p_model, prepared)

    def invoke_executor(messages: list[dict[str, Any]]) -> dict[str, Any]:
        prepared = _trim_context(_with_system(messages, executor_system, e_user_suffix))
        return _call_llm(e_client, e_model, prepared)

    def invoke_reviewer(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _call_llm(p_client, p_model, _trim_context(_with_system(messages, reviewer_system)))

    # --- clarify node ---
    def clarify_node(state: AgentState) -> AgentState:
        clarify_messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompts.build_clarify_prompt(state["task"])},
        ]
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            data = invoke_planner(clarify_messages)
        assumptions: list[str] = data.get("assumptions", [])

        if assumptions:
            console.print("\n[bold cyan][clarify][/bold cyan] Proceeding with these assumptions:")
            for i, a in enumerate(assumptions, 1):
                console.print(f"  {i}. {a}")
        else:
            console.print("[dim][clarify] No assumptions generated — proceeding.[/dim]")

        console.print("\n[dim]Press Enter to proceed, or type a correction:[/dim] ", end="")
        correction = input().strip()

        task = state["task"]
        if correction:
            task = f"{task}\n\nCorrection from user: {correction}"
            console.print("[dim]Task updated with your correction.[/dim]")

        seed_messages: list[dict[str, Any]] = [
            {"role": "user", "content": task},
        ]
        return {**state, "task": task, "messages": seed_messages}

    # --- plan node ---
    def plan_node(state: AgentState) -> AgentState:
        if state["iterations"] >= _MAX_ITERATIONS:
            console.print("[yellow]Maximum iterations reached — stopping.[/yellow]")
            return {**state, "done": True, "answer": "Stopped: maximum iterations reached."}

        label = f"[dim]Working... (step {state['iterations'] + 1})[/dim]"
        with console.status(label, spinner="dots"):
            data = invoke_executor(state["messages"])
        step = parse_plan(data)

        if isinstance(step, Answer):
            return {**state, "done": True, "answer": step.content}

        return {**state, "pending_tool": {"tool": step.tool, "arguments": step.arguments}}

    # --- act node ---
    def act_node(state: AgentState) -> AgentState:
        tool_name = state["pending_tool"]["tool"]
        arguments = dict(state["pending_tool"].get("arguments") or {})
        call = ToolCall(tool=tool_name, arguments=arguments)
        console.print(f"[bold blue][act][/bold blue]  {call.tool}({_fmt_args(call.arguments)})")
        result = _dispatch(call, workspace, config, console)
        return {**state, "last_observation": result}

    # --- observe node ---
    def observe_node(state: AgentState) -> AgentState:
        tool_name = state["pending_tool"]["tool"]
        arguments = dict(state["pending_tool"].get("arguments") or {})

        obs = state["last_observation"]
        if len(obs) > _MAX_OBS_CHARS:
            total = len(state["last_observation"])
            obs = obs[:_MAX_OBS_CHARS] + f"\n... (truncated — {total} chars total)"

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps(
                {"action": "tool_call", "tool": tool_name, "arguments": arguments}
            ),
        }
        tool_msg: dict[str, Any] = {
            "role": "user",
            "content": f"[Tool result: {tool_name}]\n{obs}",
        }

        written = list(state["written_files"])
        if tool_name == "write_file":
            path = str(arguments.get("path", ""))
            if path and path not in written:
                written.append(path)

        return {
            **state,
            "messages": [*state["messages"], assistant_msg, tool_msg],
            "iterations": state["iterations"] + 1,
            "written_files": written,
            "pending_tool": {},
            "last_observation": "",
        }

    # --- reviewer node ---
    def reviewer_node(state: AgentState) -> AgentState:
        if state["review_count"] >= _MAX_REVIEWS:
            console.print("[dim][review] Max reviews reached — accepting.[/dim]")
            return state

        review_messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompts.build_reviewer_prompt(
                    state["task"], state["answer"], state["written_files"]
                ),
            }
        ]
        with console.status("[dim]Reviewing...[/dim]", spinner="dots"):
            data = invoke_reviewer(review_messages)
        review: Review = parse_review(data)

        if review.verdict != "revise":
            console.print(f"[green][review] Approved: {review.reason or 'work looks good'}[/green]")
            return {**state, "review_count": state["review_count"] + 1}

        console.print(f"[yellow][review] Revision requested: {review.feedback}[/yellow]")
        feedback_msg: dict[str, Any] = {
            "role": "user",
            "content": (
                f"[Reviewer] The previous answer was not satisfactory.\n"
                f"{review.feedback}\n"
                f"Please revise your work and try again."
            ),
        }
        return {
            **state,
            "done": False,
            "answer": "",
            "review_count": state["review_count"] + 1,
            "messages": [*state["messages"], feedback_msg],
        }

    # --- verify node ---
    def verify_node(state: AgentState) -> AgentState:
        if state["written_files"]:
            _verify_written_files(state["written_files"], workspace, console, verify_config)
        return state

    def _route_plan(state: AgentState) -> str:
        if not state["done"]:
            return "act"
        return "reviewer" if reviewer_enabled else "verify"

    def _route_reviewer(state: AgentState) -> str:
        return "verify" if state["done"] else "plan"

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("clarify")
    graph.add_edge("clarify", "plan")
    graph.add_conditional_edges(
        "plan", _route_plan, {"act": "act", "reviewer": "reviewer", "verify": "verify"}
    )
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "plan")
    graph.add_conditional_edges("reviewer", _route_reviewer, {"verify": "verify", "plan": "plan"})
    graph.add_edge("verify", END)
    return graph.compile()


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool arguments for display, truncating long values."""
    parts: list[str] = []
    for k, v in args.items():
        s = repr(v)
        parts.append(f"{k}={s[:60]}{'...' if len(s) > 60 else ''}")
    return ", ".join(parts)


# Exported type for callers that want to inject a custom model function (Phase 9).
InvokeModel = Callable[[list[dict[str, Any]]], dict[str, Any]]


def run_agent(
    task: str,
    config: dict[str, Any],
    workspace: Path,
    console: Console,
    thinking_mode: bool | None = None,
) -> None:
    """Run the agent graph for a single user task."""
    app = make_graph(config, workspace, console, thinking_mode)
    prior_messages = history.load(workspace)
    initial: AgentState = {
        "messages": prior_messages,
        "task": task,
        "iterations": 0,
        "written_files": [],
        "done": False,
        "answer": "",
        "pending_tool": {},
        "last_observation": "",
        "review_count": 0,
    }
    final: AgentState = app.invoke(initial)
    if final.get("answer"):
        console.print(f"\n[bold green]Done.[/bold green] {final['answer']}")
    history.save(workspace, final["messages"])
