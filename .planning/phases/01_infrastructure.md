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

## Step 2: Start the Inference Server

`llama-server` exposes an OpenAI-compatible REST API at `http://localhost:8080/v1`.

```bash
./llama-server \
  --model models/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  -ngl 9999 \                    # Offload all layers to GPU (large number clamps to model max)
  --threads 8 \                  # Match exact physical core count
  --ctx-size 32768 \             # Large context for codebase tasks
  --batch-size 512 \             # Prompt processing batch
  --ubatch-size 512 \            # Physical micro-batch for prompt eval
  --flash-attn on \              # Enable Flash Attention
  --parallel 4 \                 # Concurrent request slots (server mode)
  --port 8080
```

**Critical tuning notes:**
- `--threads` must equal the **physical** core count (not logical/hyperthreaded). This is the single biggest cause of slow inference.
- `-ngl` (alias `--n-gpu-layers`) takes a **number only** — `auto` is not a valid value. Use `9999` to offload everything; llama.cpp clamps it to the model's actual layer count. Reduce if you hit OOM.
- `--flash-attn` accepts `on`, `off`, or `auto` (default). `on` is explicit and safe.
- `--ctx-size` scales KV cache linearly — only set as large as your tasks require.

**Optional: Speculative decoding** (1.6–2.5× throughput gain)

Run a small draft model alongside the main model. Both must share the same tokenizer:

```bash
./llama-server \
  --model models/qwen2.5-coder-7b-instruct-q5_k_m.gguf \
  --model-draft models/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf \
  --draft-max 8 \
  -ngl 9999 \
  --threads 8
```

## Step 3: Verify the Server

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "print hello world in Python"}]}'
```
