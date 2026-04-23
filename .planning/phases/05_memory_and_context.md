# Phase 5: Memory and Context

## Short-Term Memory (Context Window)

The conversation history (Thought → Action → Observation chains) lives in the model's context window. With `llama.cpp` and a 32K+ context, this is sufficient for complex multi-step tasks.

Use a sliding window or summarization strategy when context grows too long.

## Long-Term Memory (RAG)

For codebase-aware context retrieval:

1. **Chunk** project files (functions, classes, modules)
2. **Embed** using `sentence-transformers` (e.g. `nomic-embed-text` or `all-MiniLM-L6-v2`) running via the Python library directly. Do **not** route embeddings through `llama-server`'s `/v1/embeddings` endpoint and feed them into ChromaDB — `LlamaCppEmbeddings` degrades similarity search quality severely. Use `sentence-transformers` as a standalone embedder.
3. **Store** in ChromaDB (local vector database, no external service)
4. **Retrieve** the top-k most relevant chunks when the agent starts a new subtask

This lets the agent answer questions about large codebases without loading every file into context.
