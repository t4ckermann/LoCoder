# Phase 5: Memory and Context

## Short-Term Memory (Context Window)

The conversation history (`[plan] → [act] → [observe]` chains) lives in the model's context window. With `llama.cpp` and a 32K+ context, this is sufficient for complex multi-step tasks.

### Summarization (auto-compaction)

When the context reaches **80% of `--ctx-size`**, LoCoder triggers a compaction pass:

1. The oldest turns (everything before the most recent 25% of the window) are extracted
2. A single summarization call is made — using the planner model in hierarchical mode, or the same model in single mode — with the prompt: *"Summarize the work done so far, decisions made, and current state, concisely."*
3. The extracted turns are replaced with the summary in the conversation history
4. The agent continues from the compressed state

This costs one extra inference call per compaction but preserves the gist of earlier work, preventing the agent from repeating steps or contradicting earlier decisions.

**Config** (under `[agent]`):

```toml
[agent]
clarification_timeout = 10
context_compaction_threshold = 0.80   # Fraction of ctx_size that triggers summarization
```

### Manual Context Reset (`/clear`)

The user can clear the entire conversation context at any time with the `/clear` command in the LoCoder CLI:

```
> /clear
Context cleared. Starting fresh session.
```

`/clear` wipes the in-memory conversation history and resets the agent to its initial system prompt. It does **not** affect the RAG index, the filesystem, or any code already written. Useful when a task has gone off-track and a clean start is preferable to summarizing a broken state.

**Available slash commands** (to be expanded as features are added):

| Command | Effect |
| :--- | :--- |
| `/clear` | Reset conversation context |
| `/status` | Show current model, mode, context usage % |
| `/help` | List available slash commands |

## Long-Term Memory (RAG)

For codebase-aware context retrieval:

1. **Chunk** project files (functions, classes, modules)
2. **Embed** using `fastembed` with the `nomic-embed-text` model. It runs fully in-process, requires no separate server, and downloads the model on first use (~270 MB). Do **not** use `LlamaCppEmbeddings` — it degrades similarity search quality severely.
3. **Store** in ChromaDB (local vector database, no external service)
4. **Retrieve** the top-k most relevant chunks when the agent starts a new subtask

This lets the agent answer questions about large codebases without loading every file into context.

### Indexing — When and How

The workspace is indexed automatically every time `locoder start` is run, before the agent becomes interactive. Indexing is incremental: each file's `mtime` (last modified timestamp) is compared against what was stored in ChromaDB on the previous run. Only new or changed files are re-chunked and re-embedded — unchanged files are skipped entirely.

**Indexing flow on startup:**

```
locoder start
    ↓
Scan workspace files (respects .gitignore and [rag] exclude patterns)
    ↓
For each file: compare mtime vs. stored mtime in ChromaDB metadata
    ↓
Changed / new → re-chunk → re-embed → upsert into ChromaDB
Unchanged     → skip
Deleted       → remove from ChromaDB
    ↓
Agent becomes interactive
```

**Config options** (under `[rag]` in `config.toml`):

```toml
[rag]
embeddings_model = "nomic-embed-text"
vector_store_dir = "~/.locoder/vectorstore"
exclude = ["**/.git", "**/node_modules", "**/__pycache__", "**/dist", "**/*.lock"]
chunk_size = 512        # Tokens per chunk
chunk_overlap = 64      # Token overlap between consecutive chunks
top_k = 5              # Chunks retrieved per query
```

For very large repos the startup scan adds a few seconds — acceptable for a coding agent session that will run for minutes to hours.
