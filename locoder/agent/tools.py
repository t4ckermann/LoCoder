from __future__ import annotations

from pathlib import Path

import pathspec


def _validated(path: str, workspace: Path) -> Path:
    """Resolve *path* within *workspace*; raise ValueError if it escapes."""
    target = (workspace / path).resolve()
    if not target.is_relative_to(workspace.resolve()):
        raise ValueError(f"Path {path!r} is outside the workspace")
    return target


def read_file(path: str, workspace: Path) -> str:
    try:
        return _validated(path, workspace).read_text(errors="replace")
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: file not found: {path!r}"
    except OSError as exc:
        return f"Error reading file: {exc}"


def write_file(path: str, content: str, workspace: Path) -> str:
    try:
        p = _validated(path, workspace)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path!r}"
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error writing file: {exc}"


def list_directory(path: str, workspace: Path) -> str:
    try:
        p = _validated(path, workspace)
        if not p.is_dir():
            return f"Error: {path!r} is not a directory"
        entries = sorted(
            str(child.relative_to(workspace)) + ("/" if child.is_dir() else "")
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
            root = _validated(path, workspace)
        else:
            root = workspace.resolve()
    except ValueError as exc:
        return f"Error: {exc}"

    gitignore = workspace / ".gitignore"
    spec: pathspec.PathSpec[pathspec.Pattern] | None = None
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
            rel = str(file_path.relative_to(workspace))
            if spec is not None and spec.match_file(rel):
                continue
            try:
                text = file_path.read_text(errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    lines.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(lines) >= 50:
                        lines.append("... (truncated at 50 matches)")
                        return "\n".join(lines)
    except OSError as exc:
        return f"Error: {exc}"

    return "\n".join(lines) if lines else f"No matches for {query!r}"


def search_knowledge_base(query: str) -> str:
    """Stub — RAG vector store not yet implemented (Phase 5)."""
    return "(knowledge base not yet available)"
