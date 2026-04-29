# LoCoder

Local-first coding agent powered by [llama.cpp](https://github.com/ggerganov/llama.cpp). Runs entirely on your machine — no API keys, no cloud.

> **Status:** Phase 5 complete. CLI, hardware detection, model management, server launcher, agent loop, RAG knowledge base (ChromaDB + fastembed), and persistent conversation history are all working.

---

## How it works

1. `locoder setup` detects your hardware and downloads a pre-built `llama-server` binary.
2. `locoder pull <model>` downloads the right model quantization for your RAM.
3. `locoder start` launches the local server and drops you into an interactive agent session — no internet required after that.

---

## Requirements

- Python 3.11+
- macOS, Linux, or Windows (x86-64 / ARM64)
- At least 4 GB RAM (8 GB recommended)

No GPU required. Apple Silicon unified memory counts as VRAM.

---

## Installation

**macOS and Linux both require a virtual environment** — modern system Python (Homebrew, Ubuntu, etc.) blocks global `pip install` by default (PEP 668).

### Option A — `uv` (recommended, fastest)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles the venv automatically.

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux

# Install LoCoder
uv pip install -e .
```

### Option B — standard virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -e .
```

After activation, `locoder` is on your PATH for the current shell session. Re-run `source .venv/bin/activate` in new terminals.

### Option C — pipx (install as an app, no manual venv)

```bash
pipx install .
```

`pipx` manages its own isolated environment. Install it with `brew install pipx` (macOS) or `pip install --user pipx` (Linux).

### Option D — `setup.sh` (for local development)

A convenience script that creates the venv and installs LoCoder in one step:

```bash
./setup.sh
source .venv/bin/activate
```

After activation, `locoder` is on your PATH for the current shell session.

---

## Quick start

```bash
# 1. Detect hardware, install llama-server, write .locoder.toml
locoder setup

# 2. Download a model (auto-selects quantization for your RAM)
locoder pull qwen2.5-coder-7b

# 3. Start the server
locoder start
```

---

## Commands

### `locoder setup`
Detects CPU cores, RAM, and VRAM. Downloads a pre-built `llama-server` binary if one is not on PATH. Writes `.locoder.toml` in the current directory with sensible defaults for your hardware.

### `locoder pull <model> [--quant <q>]`
Downloads a model GGUF from HuggingFace. Without `--quant`, automatically picks the best quantization that fits your available RAM.

```bash
locoder pull qwen2.5-coder-7b           # auto quant
locoder pull qwen2.5-coder-7b -q q4_k_m # explicit quant
```

### `locoder models list`
Lists locally installed models with file size.

```bash
locoder models list   # or: locoder ls
```

### `locoder models remove <model>`
Removes an installed model (with confirmation prompt).

### `locoder models upgrade <old> <new>`
Downloads a new model, then offers to remove the old one to free disk space.

```bash
locoder models upgrade qwen2.5-coder-1.5b qwen2.5-coder-7b
```

### `locoder registry list`
Lists all models available in the registry with RAM tier, parameter count, and install status.

```bash
locoder registry list   # or: locoder registry ls
```

### `locoder registry update`
Fetches the latest registry from GitHub and saves it to `~/.locoder/registry.json`.

### `locoder start`
Starts the llama-server subprocess(es) and the interactive agent loop. Type tasks at the `>` prompt; the agent clarifies assumptions, plans, executes tools, and verifies results.

**Slash commands inside the session:**

| Command | Effect |
|---|---|
| `/help` | Show available commands |
| `/status` | Show current model, mode, ports, and thinking mode |
| `/think` | Toggle deep thinking mode on/off for the current session |
| `/reindex` | Re-index the workspace into the knowledge base |
| `/history` | Show the last 5 task summaries from this workspace |
| `/clear` | Clear persistent conversation history for this workspace |
| `Ctrl-C` | Stop servers and exit |

**Agent config** (in `.locoder.toml`):

```toml
[agent]
clarification_timeout = 10   # seconds shown in clarify prompt (informational)
thinking_mode = true          # enable <|think|> prefix for Gemma 4 models
```

**RAG config** (controls knowledge base indexing):

```toml
[rag]
embeddings_model = "nomic-ai/nomic-embed-text-v1.5"  # fastembed model (downloaded once, ~250 MB)
vector_store_dir = "~/.locoder/vectorstore"  # where ChromaDB persists the index
chunk_size = 512      # words per chunk
chunk_overlap = 64    # overlap between consecutive chunks
top_k = 5             # results returned per query
exclude = [           # glob patterns to skip during indexing
  "**/.git",
  "**/node_modules",
  "**/__pycache__",
  "**/dist",
  "**/*.lock",
]
```

The agent prefers `search_knowledge_base` (semantic) for concept/intent queries and falls back to `search_codebase` (exact grep) for known symbols or when indexed files are stale. A stale-file warning is shown inline when source files have changed since the last index.

**Sandbox config** (controls `run_code` tool):

```toml
[sandbox]
execution_timeout = 60    # seconds before subprocess is killed
allow_network = false     # outbound network access during code execution
```

---

## Model catalog

Pick based on your available RAM. "Installed RAM" means combined RAM+VRAM on Apple Silicon.

| Model | RAM needed | Params | Notes |
|---|---|---|---|
| `phi-3.5-mini` | 4 GB | 3.8B | Best ultra-low-RAM option |
| `qwen2.5-coder-1.5b` | 4 GB | 1.5B | Fastest, smallest footprint |
| `qwen3-4b` | 4 GB | 4B | Qwen3 dense; native tool calling; thinking mode |
| `gemma4-e2b` | 4 GB | 5.1B | 2.3B active params; thinking mode |
| `qwen2.5-coder-7b` | 8 GB | 7.6B | Recommended default; excellent code quality |
| `qwen3-8b` | 8 GB | 8B | Qwen3 dense; strong reasoning + coding; thinking mode |
| `gemma4-e4b` | 8 GB | 8.0B | 4.5B active; thinking mode; planner default |
| `deepseek-coder-v2-lite` | 8 GB | 16B | FIM support; strong code model |
| `mistral-nemo` | 12 GB | 12.2B | Best native tool calling; good planner |
| `phi-4` | 16 GB | 14B | Strong reasoning; needs `--jinja` for tool calls |
| `qwen2.5-coder-14b` | 16 GB | 14.8B | Best code quality in mid-tier |

### Quantization guide

| Quant | Quality | Notes |
|---|---|---|
| `q8_0` | Near-lossless | Highest quality, largest file |
| `q6_k` | Near-lossless | Good when VRAM headroom exists |
| **`q5_k_m`** | **Excellent** | **Recommended default for coding** |
| `q4_k_m` | Good | Default for CPU-only inference |
| `q3_k_m` | Acceptable | Sub-4 GB devices only |
| `q2_k` | Poor | Emergency / extremely low RAM |

---

## Inference mode

LoCoder runs a single `llama-server` process on one port. The agent serialises its planner and executor phases through the same endpoint, passing accumulated context in the message history between calls. Each phase receives a distinct system-prompt prefix (`[PLANNER]` / `[EXECUTOR]`) so the model adapts its behaviour without requiring two separate servers.

This replaces the earlier two-port hierarchical design, which doubled RAM usage with no meaningful quality benefit.

---

## Config

`locoder setup` writes `.locoder.toml` in the current directory (gitignored). The global fallback is `~/.locoder/config.toml`. Override the path with `LOCODER_CONFIG`.

Key settings:

```toml
[inference]
llama_server_bin = "..."  # path to llama-server

[inference.single]
model = "qwen2.5-coder-7b"
port = 8080

[models]
dir = "~/.locoder/models"
```

---

## Project layout

```
locoder/
  cli/        # typer commands
  config/     # TOML read/write
  hardware/   # CPU/RAM/VRAM detection, port allocation
  models/     # registry, downloader, quant selector, OpenAI-compat client
  server/     # llama-server install and launcher
  agent/      # ReAct agent loop, tools, sandbox, LangGraph state machine
  data/       # bundled registry.json
```
