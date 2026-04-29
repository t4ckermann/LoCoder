from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, StateGraph
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from typing_extensions import TypedDict

from locoder.agent import prompts, sandbox, tools
from locoder.agent.schema import Answer, ToolCall, parse_plan
from locoder.models.client import (
    active_model_name,
    get_sync_client,
    thinking_prefix,
)

_MAX_ITERATIONS = 30


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    task: str
    iterations: int
    written_files: list[str]
    done: bool
    answer: str


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call llama-server with json_object response format; return parsed dict."""
    resp = client.chat.completions.create(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    try:
        return cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError:
        return {"action": "answer", "content": raw}


def _dispatch(
    call: ToolCall,
    workspace: Path,
    config: dict[str, Any],
) -> str:
    """Execute a ToolCall and return its string result."""
    name = call.tool
    args = call.arguments
    if name == "read_file":
        return tools.read_file(str(args.get("path", "")), workspace)
    if name == "write_file":
        return tools.write_file(str(args.get("path", "")), str(args.get("content", "")), workspace)
    if name == "run_code":
        result = sandbox.run_code(
            str(args.get("code", "")),
            str(args.get("language", "python")),
            config,
            workspace,
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
    return f"Unknown tool: {name!r}"


def _verify_written_files(written: list[str], workspace: Path, console: Console) -> None:
    """Run ruff + mypy on any Python files written during the agent run."""
    py_files = [f for f in written if f.endswith(".py")]
    if not py_files:
        return

    abs_files = [str(workspace / f) for f in py_files]

    console.print("[dim][verify] ruff check...[/dim]")
    r = subprocess.run(
        ["ruff", "check", "--fix", *abs_files],
        capture_output=True,
        cwd=str(workspace),
    )
    if r.returncode != 0:
        out = r.stdout.decode(errors="replace")
        console.print(f"[yellow][verify] ruff issues:\n{out}[/yellow]")

    console.print("[dim][verify] mypy...[/dim]")
    r = subprocess.run(
        ["mypy", *abs_files],
        capture_output=True,
        cwd=str(workspace),
    )
    if r.returncode != 0:
        out = r.stdout.decode(errors="replace")
        console.print(f"[yellow][verify] mypy issues:\n{out}[/yellow]")


def make_graph(
    config: dict[str, Any],
    workspace: Path,
    console: Console,
    thinking_mode: bool | None = None,
) -> Any:
    """Build and compile the agent LangGraph state machine."""
    client = get_sync_client(config)
    model = active_model_name(config)
    if thinking_mode is None:
        thinking_enabled: bool = bool(config.get("agent", {}).get("thinking_mode", False))
    else:
        thinking_enabled = thinking_mode
    t_prefix = thinking_prefix(model, thinking_enabled)
    system_prompt = prompts.build_system_prompt(workspace, t_prefix)

    # --- clarify node ---
    def clarify_node(state: AgentState) -> AgentState:
        clarify_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts.build_clarify_prompt(state["task"])},
        ]
        data = _call_llm(client, model, clarify_messages)
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

        # Seed the conversation history used by plan_node
        seed_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        return {**state, "task": task, "messages": seed_messages}

    # --- plan node ---
    def plan_node(state: AgentState) -> AgentState:
        if state["iterations"] >= _MAX_ITERATIONS:
            console.print("[yellow]Maximum iterations reached — stopping.[/yellow]")
            return {**state, "done": True, "answer": "Stopped: maximum iterations reached."}

        console.print(f"[dim][plan] thinking... (step {state['iterations'] + 1})[/dim]")
        data = _call_llm(client, model, state["messages"])
        step = parse_plan(data)

        if isinstance(step, Answer):
            return {**state, "done": True, "answer": step.content}

        # It's a ToolCall — add assistant's intent to messages
        console.print(f"[bold blue][act][/bold blue]  {step.tool}({_fmt_args(step.arguments)})")
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps(
                {"action": "tool_call", "tool": step.tool, "arguments": step.arguments}
            ),
        }
        result = _dispatch(step, workspace, config)
        tool_msg: dict[str, Any] = {
            "role": "user",
            "content": f"[Tool result: {step.tool}]\n{result}",
        }

        written = list(state["written_files"])
        if step.tool == "write_file":
            path = str(step.arguments.get("path", ""))
            if path and path not in written:
                written.append(path)

        return {
            **state,
            "messages": [*state["messages"], assistant_msg, tool_msg],
            "iterations": state["iterations"] + 1,
            "written_files": written,
            "done": False,
        }

    # --- verify node ---
    def verify_node(state: AgentState) -> AgentState:
        if state["written_files"]:
            _verify_written_files(state["written_files"], workspace, console)
        return state

    def _route(state: AgentState) -> str:
        return "verify" if state["done"] else "plan"

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("clarify")
    graph.add_edge("clarify", "plan")
    graph.add_conditional_edges("plan", _route, {"plan": "plan", "verify": "verify"})
    graph.add_edge("verify", END)
    return graph.compile()


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool arguments for display, truncating long values."""
    parts: list[str] = []
    for k, v in args.items():
        s = repr(v)
        parts.append(f"{k}={s[:60]}{'...' if len(s) > 60 else ''}")
    return ", ".join(parts)


def run_agent(
    task: str,
    config: dict[str, Any],
    workspace: Path,
    console: Console,
    thinking_mode: bool | None = None,
) -> None:
    """Run the agent graph for a single user task."""
    app = make_graph(config, workspace, console, thinking_mode)
    initial: AgentState = {
        "messages": [],
        "task": task,
        "iterations": 0,
        "written_files": [],
        "done": False,
        "answer": "",
    }
    final: AgentState = app.invoke(initial)
    if final.get("answer"):
        console.print(f"\n[bold green]Done.[/bold green] {final['answer']}")
