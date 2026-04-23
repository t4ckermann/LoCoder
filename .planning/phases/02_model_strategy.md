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

## Model Mode

LoCoder supports two modes, switched via `mode` in `config.toml`. `locoder setup` auto-detects hardware and sets a default; the user can override at any time.

### `mode = "single"`
One model handles everything: clarification, planning, code generation, and verification reasoning. Simpler, lower RAM requirement. Recommended for machines under ~20 GB combined RAM/VRAM.

### `mode = "hierarchical"`
Two models with separate roles, each running as its own `llama-server` instance on its own port (`planner_port` and `executor_port` in `config.toml`):

1. **Planner model** (`planner_model`, default port `8081`) — handles the clarification session, task decomposition, tool selection, and debugging strategy. Should be a strong reasoning model (e.g. Mistral-Nemo-Instruct).
2. **Executor model** (`executor_model`, default port `8082`) — handles code generation, file writes, and direct tool invocations. Should be a strong coding model (e.g. Qwen2.5-Coder-7B-Instruct).

The orchestrator routes each step to the appropriate model based on the current ReAct phase. Recommended for machines with > 20 GB RAM/VRAM.

Each server instance uses the shared `[inference.server_args]` block by default. Per-model overrides can be set under `[inference.server_args.planner]` and `[inference.server_args.executor]` — only keys that differ from the shared block need to be specified.

## Recommended Model Families (registered in `registry.json`, detailed in README)

- **Qwen2.5-Coder-Instruct** (0.5B → 32B) — best code quality per parameter; 128K context in larger variants. Native chat-template tool calling is unreliable without the `--jinja` flag; use GBNF grammar enforcement instead (see Phase 3).
- **DeepSeek-Coder-V2-Instruct** (Lite: 16B active / 236B MoE) — trained on 1.17T code tokens, Fill-in-the-Middle (FIM) support; tool calling via GGUF not fully documented — rely on grammar enforcement.
- **Mistral-Nemo-Instruct** (12B) — most reliable native tool/function calling of any model in this size range via llama.cpp; excellent general reasoning; good planner model candidate.
- **CodeLlama-Instruct** (7B → 34B) — instruction following only; does **not** natively support structured tool calling without a fine-tuned adapter. Only viable when grammar enforcement is used.
- **Phi-4** (14B) — Microsoft's reasoning-focused model; requires `--jinja` flag for tool calling. Note: Phi-4 is 14B, not 3.8B.
- **Phi-3.5-mini-Instruct** (3.8B) — small footprint, reasonable code quality; best choice when RAM is the primary constraint.
