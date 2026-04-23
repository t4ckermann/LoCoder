# Phase 6: Performance Optimization

## Priority-Ordered Tuning Steps

1. **Match `--threads` to physical cores** — most impactful single change for CPU inference
2. **Maximize `-ngl`** — fill VRAM, reduce it only if OOM
3. **Use Q5_K_M quantization** for coding models (best quality/size tradeoff)
4. **Enable `--flash-attn`** — free ~10–20% speedup on supported hardware
5. **Set `--ctx-size` conservatively** — KV cache scales with context; only allocate what you need
6. **Enable `--parallel N`** for server mode (N = number of concurrent agent threads)
7. **Speculative decoding** (advanced) — run a small draft model alongside the main model for 1.6–2.5× speedup. Enable with `--model-draft <draft.gguf>` and `--draft-max 8`. Both models must share the same tokenizer. Example: Qwen2.5-Coder-7B as main + Qwen2.5-Coder-1.5B as draft.

## Expected Throughput by Hardware Tier

| Hardware | Estimated tok/sec | Practical model |
| :--- | :---: | :--- |
| CPU only (8-core) | 5–20 | Qwen2.5-Coder-1.5B Q4_K_M |
| 8 GB VRAM (e.g. RTX 3060) | 30–50 | Qwen2.5-Coder-7B Q5_K_M |
| 16 GB VRAM (e.g. RTX 4060 Ti) | 60–100 | Qwen2.5-Coder-14B Q5_K_M |
| 24 GB VRAM (e.g. RTX 4090) | 120–150 | Qwen2.5-Coder-32B Q4_K_M |
| 48 GB+ (e.g. dual 24 GB) | 200+ | DeepSeek-Coder-V2 Q5_K_M |
