from __future__ import annotations

from pathlib import Path
from typing import Any

from locoder.agent.rag import _chunk_text, _collect_files, _collection_name


def test_collection_name_stable(tmp_path: Path) -> None:
    name = _collection_name(tmp_path)
    assert name == _collection_name(tmp_path)
    assert name.startswith("ws_")
    assert len(name) == 3 + 16  # "ws_" + 16 hex chars


def test_chunk_text_basic() -> None:
    words = list(range(10))
    text = " ".join(str(w) for w in words)
    chunks = _chunk_text(text, chunk_size=4, overlap=1)
    # Each chunk has at most 4 words; step = 3
    assert len(chunks) > 1
    assert all(len(c.split()) <= 4 for c in chunks)


def test_chunk_text_empty() -> None:
    assert _chunk_text("", chunk_size=512, overlap=64) == []


def test_chunk_text_smaller_than_chunk() -> None:
    chunks = _chunk_text("hello world", chunk_size=512, overlap=64)
    assert chunks == ["hello world"]


def test_collect_files_excludes_gitignore(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1")
    (tmp_path / "ignore.pyc").write_text("bytecode")
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    config: dict[str, Any] = {"rag": {"exclude": []}}
    files = _collect_files(tmp_path, config)
    names = {f.name for f in files}
    assert "keep.py" in names
    assert "ignore.pyc" not in names


def test_collect_files_exclude_pattern(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "bundle.js").write_text("built")
    config: dict[str, Any] = {"rag": {"exclude": ["**/dist/**"]}}
    files = _collect_files(tmp_path, config)
    names = {f.name for f in files}
    assert "app.py" in names
    assert "bundle.js" not in names
