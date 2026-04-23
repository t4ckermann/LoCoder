# Phase 7: Post-Change Verification

After every file write or code generation step, the agent runs a **verification pass** before reporting success to the user. The goal is to catch problems at the source rather than surface them as runtime failures later.

## Detection Strategy

The agent auto-detects which tools are appropriate by inspecting the workspace:

| Signal | Inferred tools |
| :--- | :--- |
| `pyproject.toml` / `setup.cfg` with `[tool.ruff]` | `ruff check`, `ruff format --check` |
| `.flake8` / `setup.cfg` `[flake8]` | `flake8` |
| `pyproject.toml` `[tool.black]` | `black --check` |
| `pyproject.toml` `[tool.mypy]` | `mypy` |
| `.eslintrc*` / `eslint.config.*` | `eslint` |
| `prettier.config.*` / `.prettierrc*` | `prettier --check` |
| `Cargo.toml` | `cargo clippy`, `cargo fmt --check` |
| `go.mod` | `go vet`, `gofmt -l` |
| No config found | Fall back to language-default (e.g. `ruff` for Python) |

Detection runs once at session start and the result is cached. The agent does not install missing tools — it reports them as unavailable and skips gracefully.

## Verification Loop

```
[Code written / file changed]
    ↓
[Lint] — Run detected linter(s) on changed files only
    ↓
Errors? → [Thought] reason about the error → [Fix] → re-lint (max 3 retries)
    ↓
[Format check] — Run formatter in check mode
    ↓
Formatting needed? → Apply formatter automatically (no retry needed)
    ↓
[Type check] — Run type checker if configured (mypy, tsc, etc.)
    ↓
All clear → mark step complete and report to user
```

Linting is scoped to **changed files only** where the tool supports it (e.g. `ruff check path/to/file.py`), avoiding noise from pre-existing issues in untouched code. Pre-existing violations in unchanged files are surfaced as a one-time warning at session start, not treated as agent failures.

## Retry Limit and Escalation

If lint errors persist after 3 fix attempts, the agent stops retrying and escalates to the user with:
- The exact error output
- The last attempted fix
- A question asking whether to override, skip, or take a different approach

This prevents infinite loops on genuinely ambiguous style conflicts or misconfigured linters.
