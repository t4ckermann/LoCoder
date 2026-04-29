# LoCoder

Local-first coding agent powered by [llama.cpp](https://github.com/ggerganov/llama.cpp). Runs entirely on your machine — no API keys, no cloud.

> **Status:** Phase 3 complete. CLI, hardware detection, model management, server launcher, and agent loop are all working.

---

## Requirements

- Python 3.11+
- macOS, Linux, or Windows (x86-64 / ARM64)
- At least 4 GB RAM (8 GB recommended)

No GPU required. Apple Silicon unified memory counts as VRAM.

---

## Installation

```bash
pip install -e .
```

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
| `/status` | Show current model, mode, and ports |
| `Ctrl-C` | Stop servers and exit |

**Agent config** (in `.locoder.toml`):

```toml
[agent]
clarification_timeout = 10   # seconds shown in clarify prompt (informational)
thinking_mode = true          # enable <|think|> prefix for Gemma 4 models
```

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
| `gemma4-e2b` | 4 GB | 5.1B | 2.3B active params; thinking mode |
| `qwen2.5-coder-7b` | 8 GB | 7.6B | Recommended default; excellent code quality |
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

## Inference modes

LoCoder supports two modes, set automatically by `locoder setup` based on your RAM. You can override `mode` in `.locoder.toml` at any time.

### `mode = "single"` (< 20 GB)
One model handles everything. Lower RAM requirement, simpler setup.

### `mode = "hierarchical"` (≥ 20 GB)
Two models running on separate ports:
- **Planner** (default: `gemma4-e4b`, port 8081) — task decomposition and reasoning
- **Executor** (default: `qwen2.5-coder-7b`, port 8082) — code generation

---

## Config

`locoder setup` writes `.locoder.toml` in the current directory (gitignored). The global fallback is `~/.locoder/config.toml`. Override the path with `LOCODER_CONFIG`.

Key settings:

```toml
[inference]
mode = "single"           # "single" or "hierarchical"
llama_server_bin = "..."  # path to llama-server

[inference.single]
model = "qwen2.5-coder-7b"
port = 8080

[inference.hierarchical]
planner_model = "gemma4-e4b"
planner_port = 8081
executor_model = "qwen2.5-coder-7b"
executor_port = 8082

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
