# LoCoder — Claude Code Instructions

---

## ⛔ MANDATORY: Run before marking any task complete

> These are non-negotiable. Do not say a task is done until every command below exits 0.

```bash
# 1. Lint and auto-fix
ruff check locoder/ tests/ --fix

# 2. Format check — flag diffs to the user, do NOT silently reformat
ruff format --check locoder/ tests/

# 3. Type check
mypy locoder/

# 4. Tests
pytest
```

If dev tools are not installed:
```bash
# Inside an active venv or with uv:
pip install -e ".[dev]"
# or: uv pip install -e ".[dev]"
```

---

## ⛔ MANDATORY: Bump version in pyproject.toml

After every phase completion or any user-visible change (new feature, new command, new model, behaviour change, bug fix), increment the version in `pyproject.toml` before finishing:

- **Phase completion** → bump minor: `0.4.0` → `0.5.0`
- **Feature or fix within a phase** → bump patch: `0.5.0` → `0.5.1`
- **Never leave the version the same** after delivering working changes to the user.

```bash
# Verify you actually changed it
grep "^version" pyproject.toml
```

---

## ⛔ MANDATORY: Keep README.md current

After every task that adds, removes, or changes user-visible behaviour (commands, flags, config keys, models, modes), update `README.md` before finishing. Specifically:

- New CLI command or flag → add it to the Commands section
- Model added/removed from `registry.json` → update the model catalog table
- New config key → add it to the Config section
- Inference mode or behaviour change → update the relevant section

Do **not** update the README for internal refactors, test additions, or type annotation fixes that have no user-visible effect.

**Keep README concise.** Every update is also an opportunity to remove or condense existing content. Prefer one-liners over paragraphs; prefer a single table row over a prose description. If a section grows beyond what a new user needs in 30 seconds, cut it.

---

## ⛔ MANDATORY: Sanity checklist

Check every item before finishing. Fail = do not proceed.

| # | Rule | How to verify |
|---|------|---------------|
| 1 | `from __future__ import annotations` is the first non-comment line in every `.py` file | `grep -rL 'from __future__' locoder/**/*.py` must be empty |
| 2 | All imports are at the top of the file — no late imports except under `TYPE_CHECKING` | Scan the file; ruff `E402` catches violations |
| 3 | Every public function has an explicit return type annotation | `mypy --strict` catches missing ones |
| 4 | No bare `dict` / `list` in signatures — use `dict[str, Any]`, `list[str]`, etc. | mypy reports these |
| 5 | No unused imports | ruff `F401` |
| 6 | No duplicated constants — import the source of truth, never redefine | Manual review |
| 7 | `raise typer.Exit(N) from None` in all CLI except-blocks — no raw tracebacks to user | ruff `B904` |
| 8 | No bare `dict` or `list` in variable annotations — always parameterize: `dict[str, Any]` | ruff `PYI001` / manual |
| 9 | No `Any` for known types — use `Console \| None`, `TextEmbedding`, etc. under `TYPE_CHECKING` when the type comes from a lazy import | mypy |
| 10 | ISP: functions accept only the data they need — never pass the full config blob to a function that uses one field; extract the value before calling | Manual review |
| 11 | Shared cross-module string constants go in `locoder/constants.py` — never repeat a string in two files that cannot import each other | Manual review |
| 12 | Test functions must have explicit return type annotations (`-> None`) and typed fixtures | mypy on `tests/` |
| 13 | `pytest.raises(AttributeError)` not `pytest.raises(Exception)` — always use the specific exception | ruff `B017` |

---

## Testing

Tests live in `tests/`. Run with `pytest` (configured in `pyproject.toml`).

**What to test:**
- Pure logic only — `selector.py`, `registry.py`, `launcher.build_argv`, `client.py` helpers. These have no I/O.
- Structural properties (frozen dataclasses, required registry keys) belong in tests too.
- Do **not** test functions that download files, spawn subprocesses, or hit real ports — those are integration concerns.

**What not to mock:** the bundled `registry.json` is package data and safe to read in tests without mocking.

---

## Design principles

### SOLID

Apply these to every new module, class, or function. Each principle maps to a concrete Python pattern.

#### S — Single Responsibility
One module, class, or function does one thing.

```python
# BAD — download() also updates the registry
def download(name: str) -> Path:
    refresh_registry()          # not its job
    ...

# GOOD — separate concerns
def download(name: str) -> Path: ...        # locoder/models/downloader.py
def refresh_registry() -> int: ...          # locoder/models/registry.py
```

**Python patterns**: one public function per logical operation; modules grouped by concern (`hardware/`, `models/`, `server/`); private helpers prefixed `_`.

---

#### O — Open/Closed
Open for extension, closed for modification. Add new behaviour without editing existing code.

```python
# BAD — add a new platform by editing _detect_asset_keyword()
def _detect_asset_keyword() -> str:
    if system == "darwin": return "macos-arm64"
    if system == "linux":  return "ubuntu-x64"
    raise RuntimeError("unsupported")   # must edit here to add BSD

# GOOD — dispatch table; extend by adding an entry
_PLATFORM_KEYWORDS: dict[tuple[str, str], str] = {
    ("darwin", "arm64"):   "macos-arm64",
    ("darwin", "x86_64"):  "macos-x86_64",
    ("linux",  "x86_64"):  "ubuntu-x64",
    ("linux",  "aarch64"): "ubuntu-arm64",
    ("windows","x86_64"):  "win-avx2-x64",
}
```

**Python patterns**: dispatch tables (`dict` keyed on enums/tuples); `functools.singledispatch` for type-based dispatch; `Protocol` for plug-in interfaces.

---

#### L — Liskov Substitution
Subtypes must be substitutable for their base type without breaking callers.

```python
# Use Protocol instead of ABC so any conforming object works
from typing import Protocol

class ProgressSink(Protocol):
    def update(self, completed: int, total: int | None) -> None: ...

# Any callable with this shape works — no inheritance needed
def download(name: str, progress: ProgressSink | None = None) -> Path: ...
```

**Python patterns**: `typing.Protocol` (structural subtyping) over abstract base classes; `dataclass(frozen=True)` for value objects that must not be mutated by callers; avoid mutable default arguments.

---

#### I — Interface Segregation
Callers should not depend on interfaces they don't use. Keep signatures narrow.

```python
# BAD — caller must provide a full config dict even if it only needs bin_path
def build_argv(config: dict[str, Any], model_path: Path) -> list[str]: ...

# GOOD — accept only the exact data needed
def build_argv(
    llama_server_bin: str,
    model_path: Path,
    port: int,
    args: dict[str, object],
) -> list[str]: ...
```

**Python patterns**: explicit keyword arguments over `**kwargs`; narrow `TypedDict` or `dataclass` for structured inputs; avoid passing entire config blobs through internal functions.

---

#### D — Dependency Inversion
High-level modules should not depend on low-level details; both depend on abstractions.

```python
# BAD — launcher directly imports a concrete downloader
from locoder.models.downloader import model_dir   # tight coupling

# GOOD — accept a resolver callable; test with a fake
from collections.abc import Callable
from pathlib import Path

def start_server(
    mode: str,
    config: dict[str, Any],
    resolve_gguf: Callable[[str], Path] = _resolve_gguf,   # default impl injected
) -> list[ServerHandle]: ...
```

**Python patterns**: inject dependencies as `Callable` defaults or `Protocol` parameters; use `importlib` for truly optional heavy dependencies; avoid module-level side effects that prevent substitution in tests.

---

### KISS — Keep It Simple

> The simplest correct solution is always preferred over a clever one.

Rules (in priority order):

1. **Flat over nested** — prefer a sequence of `if` guards that return early over deeply nested blocks.
   ```python
   # BAD
   def resolve(hw):
       if hw.vram_gb is not None:
           if hw.vram_gb > 20:
               return "hierarchical"
           else:
               return "single"
       else:
           return "single"

   # GOOD
   def resolve(hw: HardwareInfo) -> str:
       if hw.vram_gb is not None and hw.vram_gb > 20:
           return "hierarchical"
       return "single"
   ```

2. **No premature abstraction** — three similar lines is better than a helper function nobody asked for. Extract only when a third caller appears.

3. **No speculative features** — implement what the current task requires. No `# future: ...` scaffolding.

4. **Standard library first** — reach for `pathlib`, `tomllib`, `dataclasses`, `urllib.request` before adding a dependency.

5. **One obvious way** — if there are two equally valid implementations, pick the one a reader can understand in 10 seconds without context.

6. **Dataclasses for data, functions for logic** — do not create classes just to group functions. Use a module for that.
   ```python
   # BAD — class with no state
   class HardwareDetector:
       def detect(self) -> HardwareInfo: ...

   # GOOD — plain function
   def detect() -> HardwareInfo: ...
   ```

---

## Project conventions

- Python 3.11+, `from __future__ import annotations` everywhere.
- `ruff` (line-length 100) is the single linter + formatter — do not add `black`, `isort`, or `flake8`.
- `@dataclass(frozen=True)` for immutable value objects (`HardwareInfo`); plain `@dataclass` for mutable handles (`ServerHandle`).
- CLI output via `rich.console.Console` only — never `print()`. Errors `[red]`, warnings `[yellow]`, success `[green]`.
- `raise typer.Exit(1) from None` in CLI except-blocks — the `from None` suppresses chaining (ruff B904).
- `BaseException` catch in `downloader.py` is intentional (partial-file cleanup on Ctrl-C) — leave it.
- Bare `except Exception: pass` in `hardware/detect.py` is intentional (optional tools like `nvidia-smi`) — leave it.
- `thinking_mode = true` in `.locoder.toml` enables the `<|think|>` prefix for Gemma 4 models (see `locoder/models/client.py:thinking_prefix`).
- Platform-specific stdlib modules (`resource`, `fcntl`) must be imported **inside** a `try/except ImportError` block, not at module level — they fail on Windows. Annotate with `# noqa: PLC0415`.
- Agent sandbox config lives under `[sandbox]` in `.locoder.toml`: `execution_timeout` (int seconds, default 60), `max_extensions` (int, 0 = unlimited interactive prompts, default 10), `allow_network` (bool, default false). Any change to these keys is user-visible and requires a README + version bump.
- Sandbox kill order: SIGTERM → wait `_GRACE_PERIOD` seconds → SIGKILL. Network isolation uses `unshare --net` on Linux (hard); advisory-only on macOS/Windows. Resource caps (`RLIMIT_FSIZE` 64 MB, `RLIMIT_NPROC` 64) are applied in the forked child via `preexec_fn`.

---

## Architecture overview

```
locoder/
  cli/          # typer commands — I/O and orchestration only, zero business logic
  config/       # TOML read/write, path resolution
  hardware/     # CPU/RAM/VRAM detection, port allocation
  models/       # Registry, HuggingFace download, install state
  server/       # llama-server subprocess launch, health polling, binary install
  data/         # Bundled registry.json (package data)
```

**Cross-module import rules** (enforced by architecture, not a linter — verify manually):

| Module | May import from |
|--------|----------------|
| `cli/` | everything |
| `server/` | `models/`, `config/` |
| `models/` | `data/` only (within `locoder/`) |
| `config/` | `hardware/` |
| `hardware/` | nothing within `locoder/` |

> `models/` must not import from `hardware/`. If a function needs available RAM, accept it as a
> plain `float` parameter — the `cli/` layer calls `hardware.detect.available_gb()` and passes it in.

Cycles are forbidden. If you need to break a cycle, introduce a `Protocol` in the lower module and depend on that instead of the concrete upper module.

---

## Phase status

| Phase | Status | What was delivered |
|-------|--------|--------------------|
| 1 — Infrastructure | ✅ Complete | CLI scaffold, hardware detection, model registry, server launcher |
| 2 — Model strategy | ✅ Complete | Quant selector, HuggingFace downloader, model management commands, OpenAI-compat client |
| 3 — Agent architecture | ✅ Complete | LangGraph ReAct loop (clarify → plan → verify), tool sandbox, interactive CLI |
| 4 — Framework stack | ✅ Complete | Dependency set locked, `.gitignore`-aware `search_codebase` via `pathspec` |
| 5 — Memory & context | ✅ Complete | ChromaDB RAG, `fastembed` embeddings, persistent conversation history |
| 6 — Single-server roles | ✅ Complete | Dropped hierarchical two-port mode; single server with `[PLANNER]`/`[EXECUTOR]` system-prompt prefixes; ChromaDB telemetry suppressed. |
| 7 — Post-change verification | ✅ Complete | `verify` node in agent graph runs ruff/mypy/pytest/manual on written files; per-project `[verify]` config; setup asks verification preferences. `--host`/`--port` flags added as a minor addition. |
| 8 — Code execution safety | ✅ Complete | Soft timeout with interactive wait/abort prompt; `max_extensions` cap; network isolation via `unshare --net` on Linux (advisory on macOS/Windows); Unix resource caps (RLIMIT_FSIZE, RLIMIT_NPROC); SIGTERM → SIGKILL grace period on abort. |

### Design note — Phase 6: single-server role model

**Problem with the current hierarchical two-port mode:**
Running two separate `llama-server` processes simultaneously doubles RAM usage and adds operational complexity (two health checks, two ports, two model downloads) without a proportional benefit. `llama-server` does not support hot model swapping, so the only way to use two *different* models concurrently is two processes — but this is rarely worth the cost.

**New approach:**
- One `llama-server` process, one port, one model.
- The agent graph serialises its phases: planner call → context accumulated in `messages` → executor call on the same endpoint with the full prior context prepended.
- Role differentiation is achieved via distinct system-prompt sections (prefixed `[PLANNER]` / `[EXECUTOR]`) passed as the `system` message at each call, not via separate processes.
- The existing `invoke_model` closure in `graph.py` is already the right abstraction point: Phase 6 replaces it with two closures (`invoke_planner`, `invoke_executor`) that hit the same client but inject different system prompts.
- The `mode = "hierarchical"` config key and `start_server` multi-launch path are removed; `mode = "single"` becomes the only inference mode.

**Cross-module impact:**
- `server/launcher.py`: remove `hierarchical` branch from `start_server`; remove `planner_port` / `executor_port` from config schema.
- `config/manager.py`: remove `[inference.hierarchical]` defaults.
- `agent/graph.py`: split `invoke_model` into `invoke_planner` / `invoke_executor`; each injects a role-specific system-prompt prefix.
- `cli/cmd_start.py` and `agent/loop.py`: update status display; no more multi-handle rendering.
- `README.md`: remove the Inference modes section; document the single-server role model.
