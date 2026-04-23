# Phase 9: Multi-Agent Extensions (Future)

Once the single-agent loop is stable, LangGraph makes it straightforward to extend to multi-agent patterns:

- **Planner agent** (larger model) decomposes tasks into subtasks
- **Coder agent** (specialized coding model) implements each subtask
- **Reviewer agent** (same or different model) validates code quality and correctness
- **Orchestrator** routes tasks between agents via a LangGraph state machine

This hierarchical structure mirrors how professional engineering teams work and can be implemented incrementally.
