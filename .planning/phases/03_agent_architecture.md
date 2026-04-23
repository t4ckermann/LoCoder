# Phase 3: Agent Architecture

## Clarification Session (Pre-Execution)

Before any code is written or tools are invoked, the agent runs a **structured clarification loop** with the user. This surfaces ambiguities, missing constraints, and edge cases that would otherwise cause silent failures or require expensive re-work mid-task.

```
User Request
    ↓
[clarify] — Identify ambiguities, implicit assumptions, missing info
    ↓
State assumptions + start countdown
    ↓
User approves / corrects → confirmed task
    ↓
[plan] phase begins
```

The clarification step asks about:
- **Scope**: Which files, modules, or systems are in/out of scope?
- **Constraints**: Language version, dependencies allowed, performance requirements?
- **Edge cases**: How should the agent handle errors, empty inputs, missing files?
- **Output expectations**: Should code be tested? Documented? What style/conventions?
- **Destructive actions**: Any files or data that must not be touched?

The questions are batched into a single message — not spread across multiple turns — to keep the interaction tight. The agent then immediately states the assumptions it will proceed with, and starts a countdown:

```
I have a few questions, but I'll proceed with these assumptions in 10 seconds
unless you reply:

  1. Scope: I'll only modify `src/auth/login.py` and its test file.
  2. Edge cases: I'll raise `ValueError` on empty input.
  3. Style: I'll follow the existing code style (no docstrings unless present).

[Proceeding in 10s — reply to change anything]
```

If the user replies before the countdown expires, the agent reads the correction, updates its assumptions, and confirms before proceeding. If no reply comes, it proceeds with the stated assumptions. This keeps simple tasks fast while still giving the user a meaningful checkpoint.

**Countdown duration**: 10 seconds default, configurable in `.locoder.toml` under `[agent] clarification_timeout = 10`. Set to `0` to disable the countdown and always wait for explicit confirmation.

**Implementation note**: This phase uses the planner model in hierarchical mode. The confirmed assumptions are stored in the conversation context and referenced throughout the execution phase to avoid drift.

---

## The Agent Loop (ReAct)

After clarification, the agent runs a LangGraph state machine with the following nodes:

```
[clarify] → confirmed task
    ↓
[plan]    — Reason about the next step; select a tool and arguments
    ↓
[act]     — Invoke the selected tool
    ↓
[observe] — Process the tool output; decide whether to loop or finish
    ↓
loop back to [plan] until the task is complete
    ↓
[verify]  — Post-change verification pass (see Phase 7)
```

llama.cpp enforces structured output at the **sampling level** using GBNF grammars or JSON Schema enforcement — the model is physically constrained to emit valid tool calls. This is the **primary mechanism LoCoder relies on**, because native chat-template tool calling is inconsistent across models (e.g. Qwen2.5-Coder emits tool calls as plain text content without grammar enforcement). Grammar enforcement sidesteps model-specific inconsistencies entirely.

---

## Tool Calling Implementation

Tool calling works via two mechanisms in llama.cpp:

**Option A — JSON Schema enforcement (recommended):**
```python
# Pass a schema to llama-server; tokens are sampled to match it
response_format = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": ["read_file", "write_file", "run_code", "list_directory", "search_codebase", "search_knowledge_base"]},
            "arguments": {"type": "object"}
        },
        "required": ["tool", "arguments"]
    }
}
```

**Option B — OpenAI tool use format:**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    }
]
```

**Model compatibility notes:**

| Model | Native tool calling | Notes |
| :--- | :--- | :--- |
| Mistral-Nemo-Instruct | ✅ Reliable | Best native support in this size range |
| Gemma 4 (all variants) | ⚠️ Buggy | Streaming drops tool calls; array params serialize incorrectly. Use grammar enforcement + non-streaming. See Phase 2 for details. |
| Qwen2.5-Coder-Instruct | ❌ Unreliable | Emits tool calls as plain text without `--jinja`; use grammar enforcement |
| DeepSeek-Coder-V2-Instruct | ❌ Unreliable | Rely on grammar enforcement |
| Phi-4 | ❌ Needs flag | Requires `--jinja` flag; use grammar enforcement |
| CodeLlama-Instruct | ❌ None | No native support; grammar enforcement only |
| Any base model (non-instruct) | ❌ Incompatible | Do not use |

LoCoder defaults to grammar enforcement (Option A) for all models to guarantee consistency regardless of which model is loaded. Non-streaming inference is used for all tool calls.

---

## Core Tools to Implement

```python
def read_file(path: str) -> str:
    """Read a file from the workspace."""

def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""

def run_code(code: str, language: str = "python") -> dict:
    """Execute code in an isolated subprocess. Returns stdout, stderr, exit_code.
    Sandboxed per Phase 8 — soft timeout, workspace-scoped, no shell=True."""

def list_directory(path: str) -> list[str]:
    """List files in a directory."""

def search_codebase(query: str, path: str = ".") -> list[dict]:
    """Search for a pattern in files (ripgrep-backed)."""

def search_knowledge_base(query: str) -> list[str]:
    """Retrieve relevant context from the RAG vector store."""
```

Sandbox behaviour for `run_code` is defined in Phase 8 — soft timeout with user prompt, workspace path validation, no `shell=True`.

---

## Interactive CLI (Terminal UI)

The agent runs as an interactive CLI loop — similar in feel to Claude Code. After `locoder start` brings up the server, the user types requests at a prompt and the agent responds inline in the terminal.

```
> refactor the auth module to use JWT instead of session cookies

[clarify] I'll proceed with these assumptions in 10s unless you reply:
  1. Scope: only src/auth/ and its tests
  2. Keep existing session-based tests as reference, add new JWT tests
  3. Use PyJWT library (already in requirements.txt)

[plan] reading current auth implementation...
[act]  read_file("src/auth/session.py")
[act]  write_file("src/auth/jwt.py", ...)
[verify] ruff check — no errors
        mypy — no errors

Done. 3 files changed. Run `pytest tests/auth/` to verify.
>
```

**Slash commands** available in the interactive loop (Phase 5):

| Command | Effect |
| :--- | :--- |
| `/clear` | Reset conversation context |
| `/status` | Show model, mode, context usage % |
| `/help` | List available commands |

The interactive loop is implemented in Phase 3 as part of the agent loop. The `locoder start` command currently stubs this with a placeholder.

---

## Prompt Engineering

The system prompt defines the agent's behavior contract:

```
You are LoCoder, an expert software engineering agent.
Your environment: [describe workspace, language, constraints]
You have access to the following tools: [tool list]

When solving a task:
1. Think step by step before acting.
2. Use the minimum number of tool calls needed.
3. Always verify your work by reading back files or running code.
4. If execution fails, reason about the error before retrying.

Respond ONLY with valid JSON matching the tool call schema, or a final answer prefixed with ANSWER:.
```
