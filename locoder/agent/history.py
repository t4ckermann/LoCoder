from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

_HISTORY_DIR = Path("~/.locoder/history")
_MAX_TURNS = 200
_SEED_TURNS = 20  # tasks to inject into new session context


def _path(workspace: Path) -> Path:
    key = hashlib.sha1(str(workspace.resolve()).encode()).hexdigest()[:16]
    return _HISTORY_DIR.expanduser() / f"{key}.jsonl"


def load(workspace: Path) -> list[dict[str, Any]]:
    """Return messages from the last _SEED_TURNS tasks, oldest-first."""
    p = _path(workspace)
    if not p.exists():
        return []
    try:
        raw_turns: list[list[dict[str, Any]]] = []
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw_turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        messages: list[dict[str, Any]] = []
        for turn in raw_turns[-_SEED_TURNS:]:
            messages.extend(turn)
        return messages
    except OSError:
        return []


def save(workspace: Path, messages: list[dict[str, Any]]) -> None:
    """Append task messages as a JSONL line; trim history to _MAX_TURNS."""
    p = _path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    turns: list[str] = []
    if p.exists():
        with contextlib.suppress(OSError):
            turns = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
    turns.append(json.dumps(messages, ensure_ascii=False))
    if len(turns) > _MAX_TURNS:
        turns = turns[-_MAX_TURNS:]
    p.write_text("\n".join(turns) + "\n", encoding="utf-8")


def clear(workspace: Path) -> None:
    p = _path(workspace)
    if p.exists():
        p.unlink()


def recent_summaries(workspace: Path, n: int = 5) -> list[str]:
    """Return the first user message from each of the last n tasks."""
    p = _path(workspace)
    if not p.exists():
        return []
    summaries: list[str] = []
    try:
        lines = [ln for ln in p.read_text(errors="replace").splitlines() if ln.strip()]
        for line in lines[-n:]:
            try:
                msgs: list[dict[str, Any]] = json.loads(line)
                for msg in msgs:
                    if msg.get("role") == "user":
                        summaries.append(str(msg.get("content", ""))[:120])
                        break
            except (json.JSONDecodeError, AttributeError):
                continue
    except OSError:
        pass
    return summaries
