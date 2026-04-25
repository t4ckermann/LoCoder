# LoCoder — Claude Code Instructions

---

## ⛔ MANDATORY: Run before marking any task complete

> These are non-negotiable. Do not say a task is done until every command below exits 0.

```bash
# 1. Lint and auto-fix
ruff check locoder/ --fix

# 2. Format check — flag diffs to the user, do NOT silently reformat
ruff format --check locoder/

# 3. Type check
mypy locoder/
```

If `ruff` or `mypy` are not installed:
```bash
pip install -e ".[dev]"
```

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

Cycles are forbidden. If you need to break a cycle, introduce a `Protocol` in the lower module and depend on that instead of the concrete upper module.
