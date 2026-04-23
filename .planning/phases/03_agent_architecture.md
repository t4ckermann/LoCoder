# Phase 3: Agent Architecture

## Step 4: Clarification Session (Pre-Execution)

Before any code is written or tools are invoked, the agent runs a **structured clarification loop** with the user. This surfaces ambiguities, missing constraints, and edge cases that would otherwise cause silent failures or require expensive re-work mid-task.

```
User Request
    ↓
[Analyze] — Identify ambiguities, implicit assumptions, missing info
    ↓
[Clarify] — Ask the user targeted questions (one round, batched)
    ↓
[Confirm] — Restate understanding and proposed approach
    ↓
User approves → proceed / User corrects → re-confirm
    ↓
Execution phase begins
```

The clarification step asks about:
- **Scope**: Which files, modules, or systems are in/out of scope?
- **Constraints**: Language version, dependencies allowed, performance requirements?
- **Edge cases**: How should the agent handle errors, empty inputs, missing files?
- **Output expectations**: Should code be tested? Documented? What style/conventions?
- **Destructive actions**: Any files or data that must not be touched?

The questions are batched into a single message — not spread across multiple turns — to keep the interaction tight. The agent then restates its understanding before proceeding, giving the user a final checkpoint.

**Implementation note**: This phase uses the larger/planner model if a hierarchical setup is running. The clarification is stored in the conversation context and referenced throughout the execution phase to avoid drift.

---

## Step 5: The Agent Loop (ReAct)

After the clarification session, the core loop is **ReAct** (Reasoning + Acting):

```
Confirmed Task
    ↓
[Thought] — Plan the next step
    ↓
[Action] — Select and invoke a tool
    ↓
[Observation] — Process the tool output
    ↓
Loop until [Answer] is ready
```

llama.cpp enforces structured output at the **sampling level** using GBNF grammars or JSON Schema enforcement — the model is physically constrained to emit valid tool calls. This is the **primary mechanism LoCoder relies on**, because native chat-template tool calling is inconsistent across models (e.g. Qwen2.5-Coder emits tool calls as plain text content without grammar enforcement). Grammar enforcement sidesteps model-specific inconsistencies entirely.

## Step 6: Tool Calling Implementation

Tool calling works via two mechanisms in llama.cpp:

**Option A — JSON Schema enforcement (recommended):**
```python
# Pass a schema to llama-server; tokens are sampled to match it
response_format = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": ["read_file", "write_file", "run_code", "search"]},
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
- Reliable native tool calling (chat template): Mistral-Nemo-Instruct, Mistral-7B-Instruct
- Grammar enforcement required (GBNF/JSON Schema): Qwen2.5-Coder-Instruct, DeepSeek-Coder-V2-Instruct, Phi-4 (also needs `--jinja`)
- Grammar enforcement required, no native support: CodeLlama-Instruct
- Base models (non-instruct): **incompatible** — do not use

LoCoder defaults to grammar enforcement (Option A) for all models to guarantee consistency regardless of which model is loaded.

## Step 7: Core Tools to Implement

```python
def read_file(path: str) -> str:
    """Read a file from the workspace."""

def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""

def run_code(code: str, language: str = "python") -> dict:
    """Execute code in an isolated subprocess. Returns stdout, stderr, exit_code."""

def list_directory(path: str) -> list[str]:
    """List files in a directory."""

def search_codebase(query: str, path: str = ".") -> list[dict]:
    """Search for a pattern in files (ripgrep-backed)."""

def search_knowledge_base(query: str) -> list[str]:
    """Retrieve relevant context from the RAG vector store."""
```

`run_code` must sandbox execution. Start with subprocess + timeout + resource limits. A future phase can move to a proper sandbox (e.g., Firejail, gVisor, or Docker).

## Step 8: Prompt Engineering

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
