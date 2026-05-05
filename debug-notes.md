# LoCoder Debug Notes — 2026-05-01

## Session summary

Branch: `phase-9-multi-agent-extension`
Model: `qwen3-8b`, single mode, Metal (Mac), `ngl = 9999`

---

## Issue 1: 500 "Compute error" on all agent requests — FIXED

### Symptoms
Every prompt ("hello?", "can you generate code?") immediately returned:
```
Agent error: Error code: 500 - {'error': {'code': 500, 'message': 'Compute error.', 'type': 'server_error'}}
```

### Root causes found

1. **`response_format={"type": "json_object"}` breaks Qwen3 on llama.cpp**
   JSON grammar enforcement in llama-server fails when Qwen3 tries to emit `<think>…</think>` tokens before the JSON. The grammar rejects `<think>` as the first token → 500.

2. **`flash_attn = "on"` caused compute errors on Metal**
   Metal Flash Attention for Qwen3-8B was causing NaN/compute errors during the forward pass. Disabling it (`flash_attn = "off"`) was required.

3. **The fallback retry also failed**
   The uncommitted diff added a fallback (retry without `response_format`) but both calls were failing, so the second 500 propagated uncaught.

### Fixes applied (committed)

- Removed `response_format=json_object` entirely — no grammar enforcement.
- Added `_strip_thinking()`: strips `<think>…</think>` blocks (Qwen3/DeepSeek-R1) before JSON parsing.
- Added `_extract_json()`: best-effort JSON extraction — tries direct parse, then finds `{…}` in free-form text, then wraps as answer.
- Added `max_tokens=4096` to cap generation and prevent Qwen3 thinking chains from exhausting ctx_size.
- Better error message on 500 mentioning flash_attn and ctx_size as likely config culprits.
- User set `flash_attn = "off"` in `.locoder.toml`.

---

## Issue 2: Agent stuck on indexing — OPEN

### Symptoms
After disabling flash_attn and restarting, `/reindex` (or startup auto-index) hangs for a very long time.

### Relevant change in uncommitted diff (`locoder/agent/rag.py`)
```python
# was 100, changed to 16
batch = 16
```
The fastembed batch size was reduced from 100 → 16. This makes indexing ~6× slower for large repos. With 63 files and 5 chunks, it should still be fast. Check if the hang is in the embedding model download, fastembed initialisation, or in the batch loop itself.

### Things to check tomorrow
- [ ] Is the hang in `_load_embedder()` (first-run model download)?
      fastembed downloads `nomic-ai/nomic-embed-text-v1.5` to `/tmp/fastembed_cache/` on first use.
      Check: `ls /tmp/fastembed_cache/`
- [ ] Does indexing complete eventually, just very slowly?
      The rag.py change moved indexing to a background thread (`_spawn_index`).
      If the thread is still alive, `/reindex` will print "Indexing already in progress...".
- [ ] Is batch=16 the right value? Was it reduced to fix an OOM during embedding?
      If so, consider making it configurable via `[rag] embed_batch_size` in `.locoder.toml`.
- [ ] With flash_attn off, try a simple prompt — does the agent work now?

### Relevant files
- `locoder/agent/rag.py` — indexing logic, `_load_embedder`, `index_workspace`
- `locoder/agent/loop.py` — `_spawn_index`, `/reindex` handler
- `.locoder.toml` — `[rag]` section: `embeddings_model`, `vector_store_dir`, `exclude`

---

## Config state at end of session

```toml
[inference.server_args]
flash_attn = "off"   # changed from "on" — was causing Compute error with Qwen3 on Metal

[agent]
thinking_mode = true   # NOTE: /think prefix is added to system message, which Qwen3 ignores
                       # To properly enable Qwen3 thinking, /think must be appended to the
                       # LAST USER message. This is a known limitation — thinking_mode=true
                       # is effectively a no-op for qwen3 until this is fixed.
reviewer_enabled = true
```

## Known limitation: Qwen3 thinking mode is a no-op

`_THINKING_PREFIXES["qwen3-8b"] = "/think"` is added to the **system** message.
Qwen3's chat template only processes `/think`/`/no_think` in the **last user message**.
So `thinking_mode = true` has no effect on Qwen3 currently.

To fix: in `make_graph` in `graph.py`, for models where the prefix is `/think` (not `<|think|>`),
append the prefix to the last user message in `_with_system` instead of prepending to the system prompt.

---

## Version
Bumped `0.9.9` → `0.9.10` at end of session.
