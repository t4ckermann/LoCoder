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

## Python Dependencies

```toml
[dependencies]
openai = ">=1.0,<2.0"                    # OpenAI-compatible client
langchain = ">=0.3,<2.0"                 # Core agent framework
langchain-community = ">=0.3,<2.0"       # Community integrations
langgraph = ">=1.0,<2.0"                 # Graph-based orchestration (current: 1.x)
llama-cpp-python = ">=0.3"               # Optional: in-process inference (no server needed)
chromadb = ">=0.5,<1.0"                  # Vector store for RAG
sentence-transformers = ">=3.0,<4.0"     # Embeddings for RAG (do NOT use LlamaCppEmbeddings)
```

`llama-cpp-python` is listed as optional. It allows running inference directly in-process (useful for edge devices or tight integration), while `llama-server` + `openai` client is the default for flexibility.

**Embedding note**: Use `sentence-transformers` (e.g. `all-MiniLM-L6-v2` or `nomic-embed-text`) for generating embeddings to store in ChromaDB. Do **not** use `LlamaCppEmbeddings` with ChromaDB — it produces severely degraded similarity search quality, causing correct results to rank near the bottom.
