# Phase 1: Infrastructure Setup ✅

> **Status: complete** — all steps below are implemented and working as of v0.2.0.

---

## Step 1: Install llama.cpp (`locoder setup`)

`locoder setup` downloads a pre-built `llama-server` binary automatically — no cmake, no compiler required.

**Auto-install flow:**
1. Check for a managed binary at `~/.locoder/bin/llama-server` → use it
2. Check `PATH` for `llama-server` → use it
3. Fetch the latest release from the [llama.cpp GitHub releases API](https://api.github.com/repos/ggerganov/llama.cpp/releases/latest)
4. Pick the correct asset for the current platform/arch (skips CUDA/ROCm/Vulkan variants by default so the binary runs on any machine without GPU drivers)
5. Extract the full archive to `~/.locoder/bin/` (binary + shared libraries)
6. Create `.dylib` compatibility symlinks on macOS (the binary's RPATH uses short names like `libllama.0.dylib`; the archive ships `libllama.0.0.8902.dylib`)
7. Validate with `llama-server --version`

**Platform → asset mapping:**

| Platform | Asset keyword |
| :--- | :--- |
| macOS ARM64 (Apple Silicon) | `macos-arm64` |
| macOS x86_64 | `macos-x86_64` |
| Linux x86_64 | `ubuntu-x64` |
| Linux ARM64 | `ubuntu-arm64` |
| Windows x64 | `win-cpu-x64` |

The binary and all shared libraries land in `~/.locoder/bin/` — shared across all projects. `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` is set automatically when launching the server so the shared libs are found at runtime.

**Advanced / manual build** (only needed for custom backends like CUDA or ROCm):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# GPU (CUDA)
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)

# Apple Silicon (Metal) — pre-built binary already uses Metal
cmake -B build -DGGML_METAL=ON && cmake --build build -j$(nproc)

# CPU-only
cmake -B build && cmake --build build -j$(nproc)
```

Then point `llama_server_bin` in `.locoder.toml` at your custom binary.

---

## Step 2: Model Management

LoCoder manages model downloads automatically. Users never need to find or download GGUF files manually.

### Storage location

Models are stored globally in `~/.locoder/models/<model-id>/` — shared across all projects on the machine:

```
~/.locoder/
├── bin/
│   ├── llama-server
│   └── lib*.dylib          ← shared libraries (macOS)
└── models/
    ├── qwen2.5-coder-7b/
    │   └── qwen2.5-coder-7b-instruct-q5_k_m.gguf
    └── gemma4-26b/
        └── google_gemma-4-26B-A4B-it-Q4_K_M.gguf
```

### CLI commands

```bash
# First-run setup — detects hardware, installs llama-server, writes .locoder.toml
locoder setup

# Start the agent (downloads missing models automatically, with confirm prompt)
locoder start

# List locally installed models
locoder list
locoder ls

# Download a model (fetches default quantization from registry)
locoder pull qwen2.5-coder-7b

# Download a specific quantization
locoder pull qwen2.5-coder-7b --quant q4_k_m

# Remove a locally installed model
locoder remove qwen2.5-coder-7b

# Upgrade: download better model, prompt to remove old one on success
locoder upgrade qwen2.5-coder-1.5b qwen2.5-coder-7b

# Fetch latest model registry from GitHub
locoder registry update
```

> **Model selection is via `.locoder.toml`** — edit `[inference.single] model` or `[inference.hierarchical] planner_model` / `executor_model`. There is no `locoder use` command.

### `locoder upgrade <old-model> <new-model>`

1. Download `<new-model>` (same as `locoder pull`)
2. On success, prompt: `"Remove '<old-model>' to free up disk space? [y/N]"`
3. If confirmed, remove the old model directory
4. Print a reminder to update `.locoder.toml`

The remove step only runs after a fully successful download. The config is never modified automatically.

### Model registry

LoCoder ships a bundled `registry.json` (`locoder/data/registry.json`) mapping short names to HuggingFace repos, filenames, and RAM tiers. Filename templates support two placeholders:

- `{quant}` — lowercase (used by Qwen official repos, e.g. `q5_k_m`)
- `{QUANT}` — uppercase (used by bartowski repos, e.g. `Q5_K_M`)

```json
{
  "qwen2.5-coder-7b": {
    "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "default_quant": "q5_k_m",
    "filename": "qwen2.5-coder-7b-instruct-{quant}.gguf",
    "ram_tier": "8gb"
  },
  "gemma4-26b": {
    "repo": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
    "default_quant": "q4_k_m",
    "filename": "google_gemma-4-26B-A4B-it-{QUANT}.gguf",
    "ram_tier": "24gb"
  }
}
```

A user override at `~/.locoder/registry.json` takes precedence over the bundled file. `locoder registry update` fetches the latest from GitHub and writes the override.

---

## Step 3: Configuration

Config is **project-local** — `.locoder.toml` in the working directory. It is gitignored so machine-specific paths and ports don't end up in version control. A committed `.locoder.toml.example` serves as the template.

**Resolution order:**
1. `LOCODER_CONFIG` env var (explicit override)
2. `.locoder.toml` in the current directory ← written by `locoder setup`
3. `~/.locoder/config.toml` (global fallback)

```
your-project/
├── .locoder.toml          ← gitignored, machine-specific
├── .locoder.toml.example  ← committed, template for teammates
```

**Full config structure** (generated by `locoder setup`):

```toml
[inference]
llama_server_bin = "~/.locoder/bin/llama-server"
host = "127.0.0.1"
mode = "single"   # "single" or "hierarchical"

[inference.single]
model = "qwen2.5-coder-7b"
port = 8080

[inference.hierarchical]
planner_model  = "mistral-nemo"
planner_port   = 8081
executor_model = "qwen2.5-coder-7b"
executor_port  = 8082

[inference.server_args]
threads     = 8       # physical cores — set by locoder setup
ctx_size    = 32768
batch_size  = 512
ubatch_size = 512
flash_attn  = "on"    # accepts "on", "off", or "auto"
parallel    = 4
ngl         = 9999    # 9999 = offload all layers; 0 = CPU-only

[inference.server_args.planner]
ctx_size = 16384
parallel = 2

[inference.server_args.executor]
ctx_size = 32768
parallel = 4

[models]
dir = "~/.locoder/models"

[agent]
clarification_timeout        = 10
context_compaction_threshold = 0.80

[sandbox]
execution_timeout = 60
max_extensions    = 10
allow_network     = false

[rag]
embeddings_model = "nomic-embed-text"
vector_store_dir = "~/.locoder/vectorstore"
exclude          = ["**/.git", "**/node_modules", "**/__pycache__", "**/dist", "**/*.lock"]
chunk_size       = 512
chunk_overlap    = 64
top_k            = 5
```

**Hardware detection** (runs during `locoder setup`):

- **CPU cores**: `psutil.cpu_count(logical=False)` → `threads`
- **RAM**: `psutil.virtual_memory().total`
- **NVIDIA VRAM**: `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
- **Apple Silicon**: `sysctl -n hw.memsize` → unified memory treated as VRAM
- **Ports**: `socket.bind` attempted on 8080/8081/8082; increments until free

**Auto mode selection thresholds:**

| Effective RAM/VRAM | Mode | Model hint |
| :--- | :--- | :--- |
| < 10 GB | `single` | `small` (qwen2.5-coder-1.5b) |
| 10–20 GB | `single` | `mid` (qwen2.5-coder-7b) |
| > 20 GB | `hierarchical` | `large` (qwen2.5-coder-14b) |

The user can always override `mode` in `.locoder.toml`.

---

## Step 4: Start the Inference Server

`locoder start` manages the full startup sequence:

```
locoder start
    ↓
Read .locoder.toml — determine mode (single / hierarchical)
    ↓
For each configured model:
    Installed in ~/.locoder/models/? → continue
    Missing → confirm with user → download from HuggingFace
    Not in registry? → exit with error
    ↓
Start llama-server subprocess(es)
    ↓
Poll /health every 0.5s — timeout 60s
    ↓
"Server ready at http://127.0.0.1:<port>"
    ↓
# TODO: agent loop (Phase 3)
```

`DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` is set to `~/.locoder/bin/` before spawning so managed binaries find their shared libraries without system-level install.

**Critical tuning notes:**
- `--threads` = physical core count (not hyperthreaded). Biggest single factor for CPU speed.
- `-ngl 9999` offloads all layers; llama.cpp clamps to model's actual count. Reduce if OOM.
- `--flash-attn on/off/auto` — always pass a value; it is not a boolean flag.
- `--ctx-size` scales KV cache linearly — set conservatively.

**Optional: Speculative decoding** (1.6–2.5× throughput gain)

Both models must share the same tokenizer:

```bash
~/.locoder/bin/llama-server \
  --model ~/.locoder/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  --model-draft ~/.locoder/models/qwen2.5-coder-1.5b/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf \
  --draft-max 8 \
  -ngl 9999 \
  --threads 10
```

---

## Step 5: Verify the Server

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "print hello world in Python"}]}'
```
