# Phase 1: Infrastructure Setup

## Step 1: Build / Install llama.cpp

Clone and build `llama.cpp` with the appropriate backend for the target machine:

```bash
# GPU (CUDA)
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# Apple Silicon (Metal)
cmake -B build -DGGML_METAL=ON && cmake --build build -j$(nproc)

# CPU-only (universal fallback)
cmake -B build && cmake --build build -j$(nproc)
```

Key binaries produced:
- `llama-server` — the HTTP inference server (primary dependency)
- `llama-quantize` — for re-quantizing GGUF models locally
- `llama-bench` — to benchmark a model/hardware combination

LoCoder stores the path to the `llama-server` binary in its config (see Step 3). Users do not invoke it directly.

---

## Step 2: Model Management (`locoder pull`)

LoCoder manages model downloads automatically. Users never need to find or download GGUF files manually.

### Storage location

All models are stored in `~/.locoder/models/<model-id>/`. Example:

```
~/.locoder/
└── models/
    ├── qwen2.5-coder-7b-instruct/
    │   └── qwen2.5-coder-7b-instruct-q5_k_m.gguf
    └── qwen2.5-coder-1.5b-instruct/
        └── qwen2.5-coder-1.5b-instruct-q5_k_m.gguf
```

### CLI commands

```bash
# First-run hardware detection — writes ~/.locoder/config.toml
locoder setup

# Start the agent in the current directory
# If configured models are not installed, downloads them automatically before starting
locoder start

# List locally installed models (both aliases work)
locoder list
locoder ls

# Download a model manually (fetches recommended GGUF quantization by default)
locoder pull qwen2.5-coder-7b

# Download a specific quantization
locoder pull qwen2.5-coder-7b --quant q4_k_m

# Remove a locally installed model
locoder remove qwen2.5-coder-7b

# Update the built-in model registry from the LoCoder GitHub repo
locoder registry update
```

> **Model selection is always done via `config.toml`** — edit `[inference.single] model` or `[inference.hierarchical] planner_model` / `executor_model` directly. There is no `locoder use` command.

### Model registry

LoCoder ships with a built-in registry file (`registry.json`) that maps short model names to their HuggingFace repo, recommended GGUF filename, and quantization per RAM tier. Example entry:

```json
{
  "qwen2.5-coder-7b": {
    "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "default_quant": "q5_k_m",
    "filename": "qwen2.5-coder-7b-instruct-{quant}.gguf",
    "ram_tier": "8gb"
  }
}
```

Downloads are fetched from HuggingFace Hub using the `huggingface-hub` Python library. Progress is shown with a progress bar. The registry can be updated with `locoder registry update` (fetches latest from the LoCoder GitHub repo).

---

## Step 3: Configuration

LoCoder reads a config file at `~/.locoder/config.toml`:

```toml
[inference]
llama_server_bin = "/path/to/llama-server"   # Set automatically during setup
host = "127.0.0.1"

# Model mode: "single" or "hierarchical"
# "single"       — one model handles everything (planning, clarification, code generation)
# "hierarchical" — separate planner and executor models (requires more RAM)
# locoder setup detects available RAM/VRAM and writes a sensible default here.
mode = "single"

[inference.single]
model = "qwen2.5-coder-7b"                   # Active model for single mode
port = 8080                                  # llama-server port — change if 8080 is already in use

[inference.hierarchical]
planner_model = "mistral-nemo"               # Larger model: clarification, planning, debugging
planner_port = 8081                          # llama-server port for the planner model
executor_model = "qwen2.5-coder-7b"          # Coding model: code generation, file writes
executor_port = 8082                         # llama-server port for the executor model

# Shared server args — apply to all llama-server instances unless overridden below
[inference.server_args]
threads = 8          # Physical core count — set automatically via CPU detection
ctx_size = 32768
batch_size = 512
ubatch_size = 512
flash_attn = "on"
parallel = 4
ngl = 9999           # Set automatically based on detected GPU VRAM; reduce if OOM

# Per-model overrides for hierarchical mode (optional — omit keys to inherit from shared above)
# Useful when planner and executor have different context or memory requirements.
[inference.server_args.planner]
ctx_size = 16384     # Planner typically needs less context than the executor
parallel = 2

[inference.server_args.executor]
ctx_size = 32768     # Executor handles large file reads and code generation
parallel = 4

[models]
dir = "~/.locoder/models"

[agent]
clarification_timeout = 10             # Seconds before proceeding with stated assumptions (0 = always wait)
context_compaction_threshold = 0.80    # Fraction of ctx_size that triggers summarization

[sandbox]
execution_timeout = 60    # Seconds before prompting the user to wait or abort
max_extensions = 10       # Max times the user can extend (0 = unlimited)
allow_network = false     # Network access during code execution — off by default

[rag]
embeddings_model = "nomic-embed-text"
vector_store_dir = "~/.locoder/vectorstore"
exclude = ["**/.git", "**/node_modules", "**/__pycache__", "**/dist", "**/*.lock"]
chunk_size = 512      # Tokens per chunk
chunk_overlap = 64    # Overlap between consecutive chunks
top_k = 5             # Chunks retrieved per search_knowledge_base call
```

On first run, `locoder setup` detects hardware and writes sensible defaults into `config.toml`. Detection is intentionally lightweight — no heavy dependencies:

- **CPU cores**: `psutil.cpu_count(logical=False)` → sets `threads`
- **RAM**: `psutil.virtual_memory().total` → informs model mode default
- **NVIDIA VRAM**: one `nvidia-smi --query-gpu=memory.total --format=csv,noheader` subprocess call → sets `ngl` and informs model mode
- **Apple Silicon**: checks `sysctl hw.memsize` → treats unified memory as VRAM
- **Port availability**: attempts `socket.bind` on each default port (8080, 8081, 8082); if already in use, increments by 1 until a free port is found and writes the result to config. Warns the user which ports were reassigned.

**Auto mode selection thresholds** (written to `mode` in config, always overridable):

| Available RAM / VRAM | Default mode | Reasoning |
| :--- | :--- | :--- |
| < 10 GB | `single` (small model) | Not enough headroom for two models |
| 10–20 GB | `single` (mid model) | One good model fits; two would be too slow |
| > 20 GB | `hierarchical` (suggested) | Enough headroom to run planner + executor |

The user can always override `mode` in `config.toml` regardless of what setup detected.

---

## Step 4: Start the Inference Server

LoCoder starts and stops `llama-server` automatically when the agent starts/exits. The server is managed as a subprocess — users never need to run it manually.

### Auto-install on startup

When `locoder start` is run, LoCoder checks whether the model(s) defined in `config.toml` are present in `~/.locoder/models/`. If any are missing, it downloads them automatically before launching the server:

```
locoder start
    ↓
Read config — determine mode (single / hierarchical)
    ↓
For each configured model:
    Present in ~/.locoder/models/? → continue
    Missing → look up in registry.json → download from HuggingFace Hub
    Not in registry? → exit with error: "Unknown model '<name>'. Run `locoder registry update` or add it to registry.json."
    ↓
All models ready → start llama-server instance(s) → agent becomes interactive
```

The download uses the same `huggingface-hub` flow as `locoder pull`, with a progress bar. If the download fails (no internet, rate limit, etc.) LoCoder exits with a clear error rather than starting with a broken state.

For debugging or advanced use, the server can be started manually. Use the port from your config (`[inference.single] port`, default 8080; or `planner_port` / `executor_port` for hierarchical):

```bash
# Single mode
./llama-server \
  --model ~/.locoder/models/qwen2.5-coder-7b-instruct/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  -ngl 9999 \
  --threads 8 \
  --ctx-size 32768 \
  --batch-size 512 \
  --ubatch-size 512 \
  --flash-attn on \
  --parallel 4 \
  --port 8080   # match [inference.single] port in config

# Hierarchical mode — run once per model on its own port
./llama-server --model ~/.locoder/models/mistral-nemo-instruct/... --port 8081 ...
./llama-server --model ~/.locoder/models/qwen2.5-coder-7b-instruct/... --port 8082 ...
```

**Critical tuning notes:**
- `--threads` must equal the **physical** core count (not logical/hyperthreaded). This is the single biggest cause of slow inference.
- `-ngl` (alias `--n-gpu-layers`) takes a **number only**. Use `9999` to offload everything; llama.cpp clamps it to the model's actual layer count. Reduce if you hit OOM.
- `--flash-attn` accepts `on`, `off`, or `auto` (default). `on` is explicit and safe.
- `--ctx-size` scales KV cache linearly — only set as large as your tasks require.

**Optional: Speculative decoding** (1.6–2.5× throughput gain)

Run a small draft model alongside the main model. Both must share the same tokenizer:

```bash
./llama-server \
  --model ~/.locoder/models/qwen2.5-coder-7b-instruct/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  --model-draft ~/.locoder/models/qwen2.5-coder-1.5b-instruct/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf \
  --draft-max 8 \
  -ngl 9999 \
  --threads 8
```

---

## Step 5: Verify the Server

```bash
# Replace 8080 with whatever port is in [inference.single] port in your config
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "print hello world in Python"}]}'
```
