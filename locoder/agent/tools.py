from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec


def _resolve(path: str, workspace: Path) -> Path:
    """Resolve path. Absolute paths accepted as-is; relative paths must stay within workspace."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    target = (workspace / path).resolve()
    if not target.is_relative_to(workspace.resolve()):
        raise ValueError(f"Path {path!r} escapes the workspace")
    return target


def _display_path(p: Path, workspace: Path) -> str:
    """Return workspace-relative path if inside workspace, otherwise the absolute path."""
    try:
        return str(p.relative_to(workspace.resolve()))
    except ValueError:
        return str(p)


def read_file(path: str, workspace: Path) -> str:
    try:
        return _resolve(path, workspace).read_text(errors="replace")
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: file not found: {path!r}"
    except OSError as exc:
        return f"Error reading file: {exc}"


def write_file(path: str, content: str, workspace: Path) -> str:
    try:
        p = _resolve(path, workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path!r}"
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error writing file: {exc}"


def list_directory(path: str, workspace: Path) -> str:
    try:
        p = _resolve(path, workspace)
        if not p.is_dir():
            return f"Error: {path!r} is not a directory"
        entries = sorted(
            _display_path(child, workspace) + ("/" if child.is_dir() else "")
            for child in p.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error: {exc}"


def search_codebase(query: str, path: str, workspace: Path) -> str:
    """Case-insensitive substring search across text files, capped at 50 matches."""
    try:
        if path and path not in {".", ""}:
            root = _resolve(path, workspace)
        else:
            root = workspace.resolve()
    except ValueError as exc:
        return f"Error: {exc}"

    gitignore = workspace / ".gitignore"
    spec: pathspec.PathSpec | None = None
    if gitignore.is_file():
        try:
            patterns = gitignore.read_text(errors="replace").splitlines()
            spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except OSError:
            pass

    lines: list[str] = []
    try:
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            display = _display_path(file_path, workspace)
            # Only apply workspace .gitignore to files inside the workspace.
            if spec is not None and not Path(display).is_absolute() and spec.match_file(display):
                continue
            try:
                text = file_path.read_text(errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    lines.append(f"{display}:{lineno}: {line.rstrip()}")
                    if len(lines) >= 50:
                        lines.append("... (truncated at 50 matches)")
                        return "\n".join(lines)
    except OSError as exc:
        return f"Error: {exc}"

    return "\n".join(lines) if lines else f"No matches for {query!r}"


def search_knowledge_base(query: str, workspace: Path, config: dict[str, Any]) -> str:
    """Semantic search over the indexed codebase using ChromaDB + fastembed."""
    from locoder.agent import rag

    return rag.search(query, config, workspace)
