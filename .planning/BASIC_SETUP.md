# LoCoder — Architecture & Implementation Plan

> **Core premise**: A local-first coding agent system powered by `llama.cpp` directly (not Ollama). The goal is maximum performance on consumer hardware, full control over quantization and inference parameters, and a clean agent loop that works from a Raspberry Pi to a workstation with a high-end GPU.

---

## Why llama.cpp over Ollama

Ollama internally wraps `llama.cpp`, but adds overhead and restricts several parameters. Choosing `llama.cpp` directly gives us:

| Concern | llama.cpp | Ollama |
| :--- | :--- | :--- |
| Context window | Up to 131,072 tokens (model-dependent) | Capped at ~11,288 by default |
| Hardware support | CUDA, Metal, Vulkan, ROCm, CPU | CUDA, Metal only |
| Quantization control | Full GGUF control (Q2_K → Q8_0) | Abstracted away |
| Inference tuning | Fine-grained threads, batch, KV cache | Limited |
| Disk footprint | ~90 MB binary | Larger runtime |
| Tool calling | GBNF grammar + JSON Schema enforcement | Partially supported |
| Speculative decoding | Native support (2–2.5× speedup) | Not exposed |

**Decision**: LoCoder runs `llama-server` (the built-in HTTP server shipped with llama.cpp) and communicates with it through its OpenAI-compatible API. This decouples the inference backend from the agent code entirely — any OpenAI-compatible client library works without modification.

---

## Phase 1: Infrastructure Setup

### Step 1: Build / Install llama.cpp

Clone and build `llama.cpp` with the appropriate backend for the target machine:

```bash
# GPU (CUDA)
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON && cmake --build build -j$(nproc)

# CPU-only (universal fallback)
cmake -B build && cmake --build build -j$(nproc)
```

Key binaries produced:
- `llama-server` — the HTTP inference server (primary dependency)
- `llama-quantize` — for re-quantizing GGUF models locally
- `llama-bench` — to benchmark a model/hardware combination

### Step 2: Start the Inference Server

`llama-server` exposes an OpenAI-compatible REST API at `http://localhost:8080/v1`.

```bash
./llama-server \
  --model models/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  --n-gpu-layers auto \          # Offload all possible layers to GPU
  --threads 8 \                  # Match exact physical core count
  --ctx-size 32768 \             # Large context for codebase tasks
  --batch-size 512 \             # Prompt processing batch
  --flash-attn auto \            # Enable Flash Attention where supported
  --parallel 4 \                 # Concurrent request slots (server mode)
  --port 8080
```

**Critical tuning notes:**
- `--threads` must equal the **physical** core count (not logical/hyperthreaded). This is the single biggest cause of slow inference.
- `--n-gpu-layers auto` will fill VRAM greedily; reduce the number if you get OOM errors.
- `--ctx-size` scales KV cache linearly — only set as large as your tasks require.
- `--flash-attn` gives a ~10–20% speedup on supported hardware with no quality cost.

### Step 3: Verify the Server

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "print hello world in Python"}]}'
```

---

## Phase 2: Model Strategy

> **Full model-by-RAM-tier recommendations belong in the README.** This section covers the selection logic. The README will list specific GGUF filenames, download links, and memory requirements per tier (4 GB / 8 GB / 16 GB / 32 GB+).

### Quantization Selection

For coding tasks, quality degrades more noticeably under aggressive quantization than in general chat. The recommended quantization levels:

| Level | Quality | Use case |
| :--- | :--- | :--- |
| Q2_K | Poor for code | Emergency / extremely low RAM |
| Q3_K_M | Acceptable | Sub-4 GB devices, small models only |
| Q4_K_M | Good | Default for CPU-only inference |
| **Q5_K_M** | **Excellent** | **Recommended default for coding** |
| Q6_K | Near-lossless | When VRAM headroom exists |
| Q8_0 | Near-lossless | Highest quality, near full precision |

### Hierarchical Model Strategy

Inspired by the original plan but updated for reality:

1. **Planner model** — larger model handles task decomposition, tool selection, debugging strategy
2. **Executor model** — smaller specialized coder handles code generation steps

Both models run as separate `llama-server` instances on different ports, or the same server is called with different system prompts depending on the task. At smaller RAM tiers, a single model does both roles.

### Recommended Model Families (to be detailed in README)

- **Qwen2.5-Coder** (0.5B → 32B) — best overall performance per parameter, excellent tool calling, 128K context support in larger variants
- **DeepSeek-Coder-V2** (Lite and full) — trained on 1.17T code tokens, Fill-in-the-Middle (FIM) support, MoE architecture
- **CodeLlama** (7B → 34B) — Meta's established baseline, strong instruction following
- **Phi-4** (3.8B) — Microsoft's small model punches above its weight for reasoning and code

---

## Phase 3: Agent Architecture

### Step 4: Clarification Session (Pre-Execution)

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

### Step 5: The Agent Loop (ReAct)

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

llama.cpp enforces structured output at the **sampling level** using GBNF grammars or JSON Schema enforcement — the model is physically constrained to emit valid tool calls. This eliminates an entire class of parsing failures common with prompt-only approaches.

### Step 6: Tool Calling Implementation

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

Models that reliably support tool use: Qwen2.5-Coder (all sizes), DeepSeek-Coder-V2, CodeLlama-Instruct, Mistral-Instruct.

### Step 7: Core Tools to Implement

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

### Step 8: Prompt Engineering

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

---

## Phase 4: Framework Stack

### Chosen Stack

```
llama-server (llama.cpp)        ← inference backend
      ↓ OpenAI-compatible API
openai Python client             ← transport layer (no vendor lock-in)
      ↓
LangChain + LangGraph            ← tool binding, memory, orchestration
      ↓
Custom Agent Loop                ← LoCoder agent logic
```

**Why this stack:**

- `llama-server` exposes a standard API — the agent code is backend-agnostic
- `openai` client is the de-facto standard; switching models requires only changing `base_url`
- LangChain provides mature RAG, tool integration, and memory components
- LangGraph handles multi-step, branching, and multi-agent workflows as a state machine
- The custom agent loop sits on top and drives the ReAct pattern

### Python Dependencies

```toml
[dependencies]
openai = ">=1.0"              # OpenAI-compatible client
langchain = ">=0.3"           # Core agent framework
langchain-community = ">=0.3" # Community integrations
langgraph = ">=0.2"           # Graph-based orchestration
llama-cpp-python = ">=0.3"    # Optional: in-process inference (no server needed)
chromadb = ">=0.5"            # Vector store for RAG
sentence-transformers = "*"   # Embedding model for RAG
```

`llama-cpp-python` is listed as optional. It allows running inference directly in-process (useful for edge devices or tight integration), while `llama-server` + `openai` client is the default for flexibility.

---

## Phase 5: Memory and Context

### Short-Term Memory (Context Window)

The conversation history (Thought → Action → Observation chains) lives in the model's context window. With `llama.cpp` and a 32K+ context, this is sufficient for complex multi-step tasks.

Use a sliding window or summarization strategy when context grows too long.

### Long-Term Memory (RAG)

For codebase-aware context retrieval:

1. **Chunk** project files (functions, classes, modules)
2. **Embed** using a local embedding model (e.g., `nomic-embed-text` via `llama-server`'s `/v1/embeddings` endpoint)
3. **Store** in ChromaDB (local vector database, no external service)
4. **Retrieve** the top-k most relevant chunks when the agent starts a new subtask

This lets the agent answer questions about large codebases without loading every file into context.

---

## Phase 6: Performance Optimization

### Priority-Ordered Tuning Steps

1. **Match `--threads` to physical cores** — most impactful single change for CPU inference
2. **Maximize `--n-gpu-layers`** — fill VRAM, reduce it only if OOM
3. **Use Q5_K_M quantization** for coding models (best quality/size tradeoff)
4. **Enable `--flash-attn`** — free ~10–20% speedup on supported hardware
5. **Set `--ctx-size` conservatively** — KV cache scales with context; only allocate what you need
6. **Enable `--parallel N`** for server mode (N = number of concurrent agent threads)
7. **Speculative decoding** (advanced) — run a small draft model alongside the main model for 1.6–2.5× speedup; requires 2× the model loading but is the fastest single improvement for generation throughput

### Expected Throughput by Hardware Tier

| Hardware | Estimated tok/sec | Practical model |
| :--- | :---: | :--- |
| CPU only (8-core) | 5–20 | Qwen2.5-Coder-1.5B Q4_K_M |
| 8 GB VRAM (e.g. RTX 3060) | 30–50 | Qwen2.5-Coder-7B Q5_K_M |
| 16 GB VRAM (e.g. RTX 4060 Ti) | 60–100 | Qwen2.5-Coder-14B Q5_K_M |
| 24 GB VRAM (e.g. RTX 4090) | 120–150 | Qwen2.5-Coder-32B Q4_K_M |
| 48 GB+ (e.g. dual 24 GB) | 200+ | DeepSeek-Coder-V2 Q5_K_M |

---

## Phase 7: Post-Change Verification

After every file write or code generation step, the agent runs a **verification pass** before reporting success to the user. The goal is to catch problems at the source rather than surface them as runtime failures later.

### Detection Strategy

The agent auto-detects which tools are appropriate by inspecting the workspace:

| Signal | Inferred tools |
| :--- | :--- |
| `pyproject.toml` / `setup.cfg` with `[tool.ruff]` | `ruff check`, `ruff format --check` |
| `.flake8` / `setup.cfg` `[flake8]` | `flake8` |
| `pyproject.toml` `[tool.black]` | `black --check` |
| `pyproject.toml` `[tool.mypy]` | `mypy` |
| `.eslintrc*` / `eslint.config.*` | `eslint` |
| `prettier.config.*` / `.prettierrc*` | `prettier --check` |
| `Cargo.toml` | `cargo clippy`, `cargo fmt --check` |
| `go.mod` | `go vet`, `gofmt -l` |
| No config found | Fall back to language-default (e.g. `ruff` for Python) |

Detection runs once at session start and the result is cached. The agent does not install missing tools — it reports them as unavailable and skips gracefully.

### Verification Loop

```
[Code written / file changed]
    ↓
[Lint] — Run detected linter(s) on changed files only
    ↓
Errors? → [Thought] reason about the error → [Fix] → re-lint (max 3 retries)
    ↓
[Format check] — Run formatter in check mode
    ↓
Formatting needed? → Apply formatter automatically (no retry needed)
    ↓
[Type check] — Run type checker if configured (mypy, tsc, etc.)
    ↓
All clear → mark step complete and report to user
```

Linting is scoped to **changed files only** where the tool supports it (e.g. `ruff check path/to/file.py`), avoiding noise from pre-existing issues in untouched code. Pre-existing violations in unchanged files are surfaced as a one-time warning at session start, not treated as agent failures.

### Retry Limit and Escalation

If lint errors persist after 3 fix attempts, the agent stops retrying and escalates to the user with:
- The exact error output
- The last attempted fix
- A question asking whether to override, skip, or take a different approach

This prevents infinite loops on genuinely ambiguous style conflicts or misconfigured linters.

---

## Phase 8: Code Execution Safety (Sandbox)

Code execution is the highest-risk surface in any coding agent. Mitigation layers:

1. **Subprocess isolation**: `subprocess.run` with `timeout=30`, `resource.setrlimit` for CPU/memory caps
2. **Workspace sandboxing**: Agent can only read/write files under a designated workspace directory
3. **No network access** by default during code execution (can be toggled per session)
4. **Future**: Container-based execution (Docker or gVisor) for full isolation

Never pass user-controlled strings directly to `shell=True`. All tool arguments are validated against JSON Schema before execution.

---

## Phase 9: Multi-Agent Extensions (Future)

Once the single-agent loop is stable, LangGraph makes it straightforward to extend to multi-agent patterns:

- **Planner agent** (larger model) decomposes tasks into subtasks
- **Coder agent** (specialized coding model) implements each subtask
- **Reviewer agent** (same or different model) validates code quality and correctness
- **Orchestrator** routes tasks between agents via a LangGraph state machine

This hierarchical structure mirrors how professional engineering teams work and can be implemented incrementally.

---

## Summary: Recommended Stack

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| Inference backend | `llama-server` (llama.cpp) | Maximum performance, full hardware support |
| API interface | OpenAI-compatible (`/v1/`) | Framework-agnostic, swap models freely |
| Python client | `openai` library | Standard, well-maintained |
| Agent framework | LangChain + LangGraph | Mature tooling, graph-based orchestration |
| In-process inference | `llama-cpp-python` (optional) | Edge devices, tight integration |
| Vector store | ChromaDB | Local, no external service needed |
| Embeddings | `nomic-embed-text` via llama-server | Local, no external API |
| Quantization target | Q5_K_M (coding) / Q4_K_M (fallback) | Best quality/size tradeoff for code |
| Code execution | Sandboxed subprocess → Docker | Safety-first, incrementally hardened |
