from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locoder.agent.tools import (
    list_directory,
    read_file,
    search_codebase,
    search_knowledge_base,
    write_file,
)


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    return tmp_path


class TestReadFile:
    def test_reads_existing_file(self, ws: Path) -> None:
        (ws / "hello.txt").write_text("hello world")
        assert read_file("hello.txt", ws) == "hello world"

    def test_missing_file_returns_error(self, ws: Path) -> None:
        result = read_file("nope.txt", ws)
        assert result.startswith("Error:")

    def test_path_traversal_rejected(self, ws: Path) -> None:
        result = read_file("../../etc/passwd", ws)
        assert result.startswith("Error:")


class TestWriteFile:
    def test_creates_file(self, ws: Path) -> None:
        result = write_file("out.txt", "content", ws)
        assert "Written" in result
        assert (ws / "out.txt").read_text() == "content"

    def test_creates_parent_dirs(self, ws: Path) -> None:
        write_file("a/b/c.txt", "x", ws)
        assert (ws / "a" / "b" / "c.txt").exists()

    def test_path_traversal_rejected(self, ws: Path) -> None:
        result = write_file("../../evil.txt", "x", ws)
        assert result.startswith("Error:")


class TestListDirectory:
    def test_lists_entries(self, ws: Path) -> None:
        (ws / "a.py").write_text("")
        (ws / "b.py").write_text("")
        result = list_directory(".", ws)
        assert "a.py" in result
        assert "b.py" in result

    def test_non_directory_returns_error(self, ws: Path) -> None:
        (ws / "file.txt").write_text("")
        result = list_directory("file.txt", ws)
        assert result.startswith("Error:")

    def test_path_traversal_rejected(self, ws: Path) -> None:
        result = list_directory("../../", ws)
        assert result.startswith("Error:")


class TestSearchCodebase:
    def test_finds_match(self, ws: Path) -> None:
        (ws / "main.py").write_text("def hello():\n    pass\n")
        result = search_codebase("hello", ".", ws)
        assert "main.py" in result
        assert "hello" in result

    def test_no_match_message(self, ws: Path) -> None:
        (ws / "main.py").write_text("nothing here\n")
        result = search_codebase("xyzzy", ".", ws)
        assert "No matches" in result

    def test_path_traversal_rejected(self, ws: Path) -> None:
        result = search_codebase("x", "../../", ws)
        assert result.startswith("Error:")


class TestSearchKnowledgeBase:
    def test_returns_not_indexed_when_no_store(self, tmp_path: Path) -> None:
        config: dict[str, Any] = {"rag": {"vector_store_dir": str(tmp_path / "nonexistent")}}
        result = search_knowledge_base("anything", tmp_path, config)
        assert isinstance(result, str)
        assert "not yet indexed" in result.lower()
