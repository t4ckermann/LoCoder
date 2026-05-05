from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from openai import OpenAI
from rich.console import Console
from typing_extensions import TypedDict

from locoder.agent import history, prompts
from locoder.agent.dispatch import _MAX_OBS_CHARS, dispatch, fmt_args
from locoder.agent.llm import _MAX_CONTEXT_MESSAGES, _trim_context, call_llm  # noqa: F401
from locoder.agent.schema import Answer, Review, ToolCall, parse_plan, parse_review
from locoder.agent.verify import run_verify
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

# Re-export for callers that import these constants from graph.
__all__ = ["AgentState", "InvokeModel", "make_graph", "run_agent"]


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


# Exported type for callers that want to inject a custom model function.
InvokeModel = Callable[[list[dict[str, Any]]], dict[str, Any]]


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

    def _invoke(
        client: OpenAI, model: str, msgs: list[dict[str, Any]], system: str, suffix: str = ""
    ) -> dict[str, Any]:
        return call_llm(client, model, _trim_context(_with_system(msgs, system, suffix)))

    def clarify_node(state: AgentState) -> AgentState:
        clarify_messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompts.build_clarify_prompt(state["task"])},
        ]
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            data = _invoke(p_client, p_model, clarify_messages, planner_system, p_user_suffix)
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

        return {**state, "task": task, "messages": [{"role": "user", "content": task}]}

    def plan_node(state: AgentState) -> AgentState:
        if state["iterations"] >= _MAX_ITERATIONS:
            console.print("[yellow]Maximum iterations reached — stopping.[/yellow]")
            return {**state, "done": True, "answer": "Stopped: maximum iterations reached."}

        label = f"[dim]Working... (step {state['iterations'] + 1})[/dim]"
        with console.status(label, spinner="dots"):
            data = _invoke(e_client, e_model, state["messages"], executor_system, e_user_suffix)
        step = parse_plan(data)

        if isinstance(step, Answer):
            return {**state, "done": True, "answer": step.content}
        return {**state, "pending_tool": {"tool": step.tool, "arguments": step.arguments}}

    def act_node(state: AgentState) -> AgentState:
        tool_name = state["pending_tool"]["tool"]
        arguments = dict(state["pending_tool"].get("arguments") or {})
        call = ToolCall(tool=tool_name, arguments=arguments)
        console.print(f"[bold blue][act][/bold blue]  {call.tool}({fmt_args(call.arguments)})")
        result = dispatch(call, workspace, config, console)
        return {**state, "last_observation": result}

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
            data = _invoke(p_client, p_model, review_messages, reviewer_system)
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

    def verify_node(state: AgentState) -> AgentState:
        if state["written_files"]:
            run_verify(state["written_files"], workspace, console, verify_config)
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
