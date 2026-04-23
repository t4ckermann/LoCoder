# Phase 9: Multi-Agent Extensions (Future)

> **Scope**: This phase is not part of the v1 implementation. It is a design note that imposes one architectural constraint on v1 to ensure the multi-agent upgrade is additive rather than a rewrite.

---

## v1 Architectural Constraint

The single-agent loop in v1 **must** route all model calls and step transitions through a LangGraph state machine — not through inline if/else logic in the main loop. This is the only v1 constraint this phase imposes.

Concretely:

- The LangGraph graph in v1 has one node per ReAct phase: `clarify → plan → act → observe → verify`
- Each node calls a single function: `invoke_model(prompt, context)` — the model identity is resolved there, not in the graph
- Adding a second agent later means adding nodes and edges to the graph, and splitting `invoke_model` into `invoke_planner` / `invoke_executor` — no changes to the existing node logic

If this constraint is violated (e.g. the ReAct loop is a hand-rolled `while` loop with direct model calls), the multi-agent upgrade requires unpicking the entire control flow.

---

## Future Multi-Agent Patterns

Once the single-agent loop is stable and validated, LangGraph makes the following extensions straightforward:

- **Planner agent** (larger model) decomposes tasks into subtasks and owns the clarification session
- **Coder agent** (specialized coding model) handles code generation and file writes
- **Reviewer agent** (same or different model) validates code quality and correctness after verification
- **Orchestrator** routes tasks between agents via LangGraph edges based on current state

This mirrors how professional engineering teams work and can be added incrementally — one agent at a time — without touching existing node logic.

---

## Criteria for Moving to Phase 9

Phase 9 becomes active when all of the following are true:

- Single-agent loop is reliable across at least 3 different models
- Clarification, ReAct, verification, and sandbox are all implemented and tested
- The LangGraph state machine constraint above has been respected in v1
