# Phase 4: Framework Stack

## Chosen Stack

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

**Architectural constraint (from Phase 9):**

The v1 ReAct loop **must** be implemented as a LangGraph state machine — not a hand-rolled `while` loop with inline model calls. Each phase (`clarify → plan → act → observe → verify`) is a graph node. Model calls are resolved inside each node through a single `invoke_model()` function. This keeps the graph topology decoupled from model identity, making the multi-agent upgrade in Phase 9 additive (new nodes + edges) rather than a rewrite.

## Python Dependencies

```toml
[dependencies]
openai = ">=1.0,<2.0"                    # OpenAI-compatible client
langchain = ">=0.3,<2.0"                 # Core agent framework
langchain-community = ">=0.3,<2.0"       # Community integrations
langgraph = ">=1.0,<2.0"                 # Graph-based orchestration (current: 1.x)
llama-cpp-python = ">=0.3"               # Optional: in-process inference (no server needed)
chromadb = ">=0.5,<1.0"                  # Vector store for RAG
fastembed = ">=0.3,<1.0"                 # Embeddings — runs nomic-embed-text locally, no server needed
huggingface-hub = ">=0.23,<1.0"          # Model downloads for `locoder pull`
rich = ">=13.0,<14.0"                    # CLI progress bars and output formatting
typer = ">=0.12,<1.0"                    # CLI framework for locoder commands
tomli-w = ">=1.0"                        # Writing config.toml on setup
psutil = ">=6.0,<7.0"                    # CPU core count + RAM detection during setup
pathspec = ">=0.12,<1.0"                 # .gitignore pattern parsing for RAG file exclusion
```

`llama-cpp-python` is listed as optional. It allows running inference directly in-process (useful for edge devices or tight integration), while `llama-server` + `openai` client is the default for flexibility.

**Embedding note**: Use `fastembed` with the `nomic-embed-text` model for generating embeddings. It runs fully in-process (no separate server), downloads the model on first use, and integrates directly with ChromaDB. Do **not** use `LlamaCppEmbeddings` with ChromaDB — it produces severely degraded similarity search quality.
