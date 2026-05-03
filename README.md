# LoCoder

Local-first coding agent powered by [llama.cpp](https://github.com/ggerganov/llama.cpp). Runs entirely on your machine — no API keys, no cloud.

> **Status:** Phase 9 complete. Multi-agent reviewer node (LangGraph `[REVIEWER]` role) and global filesystem access — the agent can now read and write files at any absolute path on the machine, not just within the workspace directory.

---

## How it works

1. `locoder setup` detects your hardware and downloads a pre-built `llama-server` binary.
2. `locoder pull <model>` downloads the right model quantization for your RAM.
3. `locoder start` launches the local server and drops you into an interactive agent session.

---

## Requirements

- Python 3.11+
- macOS, Linux, or Windows (x86-64 / ARM64)
- At least 4 GB RAM (8 GB recommended)

No GPU required. Apple Silicon unified memory counts as VRAM.

---

## Installation

**Option A — `uv` (recommended)**

```bash
uv pip install -e .
```

**Option B — standard virtualenv**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Quick start

```bash
locoder setup                      # detect hardware, install llama-server
locoder pull qwen2.5-coder-7b     # download model (auto-picks quant)
locoder start                      # launch server + agent session
```

---

## Commands

### `locoder setup`
Detects CPU/RAM/VRAM, downloads `llama-server`, writes `.locoder.toml`.

### `locoder pull <model> [--quant <q>]`
Downloads a model GGUF from HuggingFace. Auto-selects quantization without `--quant`.

### `locoder models list`
Lists locally installed models with file size. Alias: `locoder ls`.

### `locoder models remove <model>`
Removes an installed model (with confirmation prompt).

### `locoder models upgrade <old> <new>`
Downloads a new model, offers to remove the old one.

### `locoder registry list`
Lists all registry models with RAM tier, params, and install status.

### `locoder registry update`
Fetches the latest registry from GitHub.

### `locoder start [--host HOST] [--port PORT]`
Starts the llama-server and drops into the interactive agent loop.

- `--host 0.0.0.0` exposes the server on all network interfaces (LAN access). Defaults to `127.0.0.1`.
- `--port 9090` overrides the configured port. Both flags patch the in-memory config so the agent client connects to the correct address.

**Session commands:**

| Command | Effect |
|---|---|
| `/help` | Show available commands |
| `/status` | Show model, port, and thinking mode |
| `/think` | Toggle deep thinking mode |
| `/reindex` | Re-index workspace into the knowledge base |
| `/history` | Show last 5 task summaries |
| `/clear` | Clear persistent conversation history |
| `Ctrl-C` | Stop server and exit |

---

## Model catalog

| Model | RAM needed | Params | Notes |
|---|---|---|---|
| `phi-3.5-mini` | 4 GB | 3.8B | Best ultra-low-RAM option |
| `qwen2.5-coder-1.5b` | 4 GB | 1.5B | Fastest, smallest footprint |
| `qwen3-4b` | 4 GB | 4B | Native tool calling; thinking mode |
| `gemma4-e2b` | 4 GB | 5.1B | 2.3B active params; thinking mode |
| `qwen2.5-coder-7b` | 8 GB | 7.6B | Recommended default |
| `qwen3-8b` | 8 GB | 8B | Strong reasoning + coding; thinking mode |
| `gemma4-e4b` | 8 GB | 8.0B | 4.5B active; thinking mode |
| `deepseek-coder-v2-lite` | 8 GB | 16B | FIM support |
| `mistral-nemo` | 12 GB | 12.2B | Best native tool calling |
| `phi-4` | 16 GB | 14B | Strong reasoning |
| `qwen2.5-coder-14b` | 16 GB | 14.8B | Best code quality in mid-tier |
| `qwen3-coder-next` | 64 GB | 79.7B | Qwen3 80B coder; sharded download (4 parts); thinking mode |

**Quantization:** `q5_k_m` is the recommended default. Use `q4_k_m` for CPU-only. Use `q8_0` for near-lossless quality.

---

## Inference mode

### Single (default)

One `llama-server` process, one port. Three roles share the same model via distinct system-prompt prefixes:

- `[PLANNER]` — clarification and assumptions
- `[EXECUTOR]` — ReAct tool-call loop
- `[REVIEWER]` — quality gate before verification (up to 2 cycles; default off)

### Dual

Two `llama-server` processes on separate ports — one thinking-capable model for planning/review, one coding-specialized model for execution.

**RAM requirement:** both model weights must fit in memory simultaneously alongside the OS (~6 GB) and the fastembed embedding model (~500 MB). On a 24 GB machine two 7–8B models exceed the budget. Practical combinations:

| Planner | Executor | Total weights | Min RAM |
|---|---|---|---|
| `qwen3-4b` (2.5 GB) | `qwen2.5-coder-7b` (5.3 GB) | ~7.8 GB | 16 GB |
| `qwen3-8b` (5.5 GB) | `qwen2.5-coder-7b` (5.3 GB) | ~10.8 GB | 32 GB |
| `gemma4-e4b` (5.5 GB) | `qwen2.5-coder-14b` (9.5 GB) | ~15 GB | 48 GB |

Set `mode = "dual"` in `[inference]` and add `[inference.dual.planner]` / `[inference.dual.executor]` sections (see Config below). Run `locoder pull` for both models before starting.

On machines with enough RAM but only one large model downloaded, single mode with `qwen3-8b` + `thinking_mode = true` is the recommended alternative — it handles both planning and coding in one process.

Enable the reviewer with `reviewer_enabled = true` in `[agent]` config (default off).

---

## Config

`locoder setup` writes `.locoder.toml` in the current directory (gitignored). Override with `LOCODER_CONFIG`.

```toml
[inference]
llama_server_bin = "..."
mode = "single"   # "single" (default) or "dual"

[inference.single]
model = "qwen2.5-coder-7b"
port = 8080

# ── Dual-model mode ──────────────────────────────────────────────────────────
# Uncomment and set mode = "dual" above to use separate models per role.
# Both models must be pulled first: locoder pull qwen3-8b && locoder pull qwen2.5-coder-7b
# [inference.dual.planner]
# model = "qwen3-8b"           # thinking-capable model for planning and review
# port = 8082
#
# [inference.dual.executor]
# model = "qwen2.5-coder-7b"   # coding-specialized model for tool-call execution
# port = 8083
# ─────────────────────────────────────────────────────────────────────────────

# server_args are auto-tuned by `locoder setup` based on detected hardware
[inference.server_args]
threads = 8          # physical CPU cores
ctx_size = 32768     # 8192 / 32768 / 65536 based on RAM tier
parallel = 4         # concurrent request slots (1 for CPU-only)
ngl = 9999           # GPU layers (0 for CPU-only)
flash_attn = "on"

# Optional: speculative decoding for 1.6–2.5× speedup
# Both models must share the same tokenizer family.
# [inference.speculative]
# enabled = true
# model_draft = "qwen2.5-coder-1.5b"   # small draft model (must be pulled first)
# draft_max = 8

[models]
dir = "~/.locoder/models"

[agent]
thinking_mode = true
reviewer_enabled = false  # set true to enable the [REVIEWER] quality-gate node

[rag]
top_k = 5
chunk_size = 512
chunk_overlap = 64
embed_batch_size = 16   # chunks per embedding batch; reduce if OOM during indexing
exclude = ["**/.git", "**/node_modules", "**/__pycache__"]

[sandbox]
execution_timeout = 60   # seconds before the wait/abort prompt appears
max_extensions = 10      # max times the user can choose [w]; 0 = unlimited
allow_network = false    # set true to allow outbound connections from code

# Per-project verification — what the agent runs after writing files
[verify]
lint = true          # ruff check + auto-fix on written .py files
type_check = true    # mypy on written .py files
tests = false        # run test_command after each task
test_command = "pytest"
manual = false       # pause and wait for Enter before continuing
```

---

## Project layout

```
locoder/
  cli/        # typer commands
  config/     # TOML read/write
  hardware/   # CPU/RAM/VRAM detection
  models/     # registry, downloader, quant selector, OpenAI-compat client
  server/     # llama-server install and launcher
  agent/      # ReAct loop, tools, sandbox, LangGraph state machine
  data/       # bundled registry.json
```
