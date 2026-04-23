# Phase 2: Model Strategy

> **Full model-by-RAM-tier recommendations belong in the README.** This section covers the selection logic. The README will list specific GGUF filenames, download links, and memory requirements per tier (4 GB / 8 GB / 16 GB / 32 GB+).

## Quantization Selection

For coding tasks, quality degrades more noticeably under aggressive quantization than in general chat. The recommended quantization levels:

| Level | Quality | Use case |
| :--- | :--- | :--- |
| Q2_K | Poor for code | Emergency / extremely low RAM |
| Q3_K_M | Acceptable | Sub-4 GB devices, small models only |
| Q4_K_M | Good | Default for CPU-only inference |
| **Q5_K_M** | **Excellent** | **Recommended default for coding** |
| Q6_K | Near-lossless | When VRAM headroom exists |
| Q8_0 | Near-lossless | Highest quality, near full precision |

## Instruct Variants Only

**Base models cannot perform tool calling or follow agentic instructions reliably.** Always use the `-instruct` or `-it` GGUF variant. Base model GGUFs are for fine-tuning or completion-only use cases and will not work with the agent loop.

## Hierarchical Model Strategy

1. **Planner model** — larger model handles task decomposition, tool selection, debugging strategy
2. **Executor model** — smaller specialized coder handles code generation steps

Both models run as separate `llama-server` instances on different ports, or the same server is called with different system prompts depending on the task. At smaller RAM tiers, a single model does both roles.

## Recommended Model Families (to be detailed in README)

- **Qwen2.5-Coder-Instruct** (0.5B → 32B) — best code quality per parameter; 128K context in larger variants. Native chat-template tool calling is unreliable without the `--jinja` flag; use GBNF grammar enforcement instead (see Phase 3).
- **DeepSeek-Coder-V2-Instruct** (Lite: 16B active / 236B MoE) — trained on 1.17T code tokens, Fill-in-the-Middle (FIM) support; tool calling via GGUF not fully documented — rely on grammar enforcement.
- **Mistral-Nemo-Instruct** (12B) — most reliable native tool/function calling of any model in this size range via llama.cpp; excellent general reasoning; good planner model candidate.
- **CodeLlama-Instruct** (7B → 34B) — instruction following only; does **not** natively support structured tool calling without a fine-tuned adapter. Only viable when grammar enforcement is used.
- **Phi-4** (14B) — Microsoft's reasoning-focused model; requires `--jinja` flag for tool calling. Note: Phi-4 is 14B, not 3.8B.
- **Phi-3.5-mini-Instruct** (3.8B) — small footprint, reasonable code quality; best choice when RAM is the primary constraint.
