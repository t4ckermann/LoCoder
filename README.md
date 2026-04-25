# LoCoder

A local-first coding agent powered by [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Runs entirely on your machine — no API keys, no cloud, no data leaving your network.

## Requirements

- Python 3.11+
- macOS, Linux, or Windows
- 4 GB+ RAM (8 GB+ recommended)

## Install

```bash
pip install -e .
```

## Quick start

```bash
# 1. Run setup in your project directory.
#    Detects hardware, downloads llama-server, and writes .locoder.toml.
cd your-project
locoder setup

# 2. Download a model (first time only — stored globally in ~/.locoder/models/).
locoder pull qwen2.5-coder-7b

# 3. Start the agent.
locoder start
```

## Config

LoCoder uses a **project-local** config file — `.locoder.toml` in your working directory. It is gitignored so machine-specific paths and ports don't end up in version control.

```
your-project/
├── .locoder.toml          ← gitignored, created by `locoder setup`
├── .locoder.toml.example  ← commit this as a template for teammates
└── ...
```

Copy `.locoder.toml.example` from this repo as a starting point, or let `locoder setup` generate it from your hardware.

**Resolution order** — LoCoder looks for config in:
1. `LOCODER_CONFIG` env var (explicit path)
2. `.locoder.toml` in the current directory
3. `~/.locoder/config.toml` (global fallback)

Models and the `llama-server` binary are always stored globally in `~/.locoder/` and shared across all projects.

## Models

| Registry name | Size | RAM (Q4_K_M) | Best for |
| :--- | :--- | :--- | :--- |
| `qwen2.5-coder-1.5b` | 1.5B | ~1 GB | Low-RAM / quick tasks |
| `qwen2.5-coder-7b` | 7B | ~5 GB | General coding (default) |
| `qwen2.5-coder-14b` | 14B | ~9 GB | Higher quality coding |
| `mistral-nemo` | 12B | ~7 GB | Planning / reasoning |
| `deepseek-coder-v2-lite` | 16B active | ~10 GB | Code-focused MoE |
| `gemma4-e2b` | 2.3B active (5.1B total) | ~3 GB | Fastest; constrained systems |
| `gemma4-e4b` | 4.5B active (8B total) | ~5 GB | **Recommended default** — text, image, audio |
| `gemma4-26b` | 3.8B active (25.2B total, MoE) | ~17 GB | High-VRAM; 256K context |
| `gemma4-31b` | 30.7B | ~18 GB | Best quality; high-VRAM only |

### Gemma 4 notes

**Why `gemma4-e4b` over `gemma4-26b`**: the 26B is MoE — it loads 25 GB of weights even though only 3.8B are active per token, causing slow memory bandwidth. The E4B loads only 8 GB total, responds significantly faster, and also supports audio. Use 26B/31B only when quality clearly outweighs latency.

**Thinking mode**: Gemma 4 supports chain-of-thought reasoning via the `<|think|>` token, injected per-request in the chat API. This is a Phase 3 feature — not yet available in the app.

```bash
locoder pull qwen2.5-coder-7b          # download
locoder pull gemma4-26b --quant q4_k_m # specific quantization
locoder list                            # show installed
locoder upgrade qwen2.5-coder-7b gemma4-26b  # download better, remove old
locoder remove qwen2.5-coder-7b        # remove
```

## All commands

```
locoder setup              Detect hardware, install llama-server, write .locoder.toml
locoder start              Start server and agent loop
locoder pull <model>       Download a model
locoder list / ls          List installed models
locoder remove <model>     Remove an installed model
locoder upgrade <old> <new>  Download new model, offer to remove old
locoder registry update    Fetch latest model registry from GitHub
locoder --version          Show version
```

## Inference modes

**Single** (default for < 20 GB RAM): one model handles everything.

**Hierarchical** (> 20 GB RAM, or set manually): separate planner and executor models on different ports. Set `mode = "hierarchical"` in `.locoder.toml` and configure `planner_model` / `executor_model`.
