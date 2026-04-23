# Phase 7: Post-Change Verification

After the agent completes an entire task and all file writes are done, it runs a single **verification pass** before reporting success to the user. The goal is to catch problems at the source rather than surface them as runtime failures later.

Verification is deliberately deferred to task completion — not triggered after each individual file write — to avoid false positives from partially-written state (e.g. mypy failing on an import that points to a file not yet written).

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
Task complete — all files written
    ↓
[Lint] — Run detected linter(s) on all files changed during the task
    ↓
Errors? → [Thought] reason about the error → [Fix] → re-lint (max 3 retries)
    ↓
[Format check] — Run formatter in check mode across all changed files
    ↓
Formatting needed? → Apply formatter automatically (no retry needed)
    ↓
[Type check] — Run type checker if configured (mypy, tsc, etc.) across all changed files
    ↓
All clear → report success to user
```

Verification is scoped to **files changed during the task** — linters are invoked with explicit file paths, not the whole project. This avoids noise from pre-existing issues in untouched code. Pre-existing violations in unchanged files are surfaced as a one-time warning at session start, not treated as agent failures.

## Retry Limit and Escalation

If lint errors persist after 3 fix attempts, the agent stops retrying and escalates to the user with:
- The exact error output
- The last attempted fix
- A question asking whether to override, skip, or take a different approach

This prevents infinite loops on genuinely ambiguous style conflicts or misconfigured linters.
