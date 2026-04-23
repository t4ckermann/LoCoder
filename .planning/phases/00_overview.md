# LoCoder — Overview & Rationale

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

## Recommended Stack Summary

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| Inference backend | `llama-server` (llama.cpp) | Maximum performance, full hardware support — managed automatically by LoCoder |
| API interface | OpenAI-compatible (`/v1/`) | Framework-agnostic, swap models freely |
| Python client | `openai` library | Standard, well-maintained |
| Agent framework | LangChain + LangGraph | Mature tooling, graph-based orchestration |
| In-process inference | `llama-cpp-python` (optional) | Edge devices, tight integration |
| Vector store | ChromaDB | Local, no external service needed |
| Embeddings | `nomic-embed-text` via `fastembed` | Local, no external API, no separate server |
| Quantization target | Q5_K_M (coding) / Q4_K_M (fallback) | Best quality/size tradeoff for code |
| Code execution | Sandboxed subprocess → Docker | Safety-first, incrementally hardened |
