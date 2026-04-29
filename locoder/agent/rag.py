from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pathspec

if TYPE_CHECKING:
    from chromadb.api.types import IncludeEnum

_MAX_FILES_WARN = 5_000
_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _fastembed_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "fastembed_cache"


# chromadb Metadata values must be str | int | float | bool
_MetaValue = str | int | float | bool
_Metadata = dict[str, _MetaValue]


def _collection_name(workspace: Path) -> str:
    return "ws_" + hashlib.sha1(str(workspace.resolve()).encode()).hexdigest()[:16]


def _collect_files(workspace: Path, config: dict[str, Any]) -> list[Path]:
    gitignore = workspace / ".gitignore"
    git_spec: pathspec.PathSpec | None = None
    if gitignore.is_file():
        with contextlib.suppress(OSError):
            git_spec = pathspec.PathSpec.from_lines(
                "gitwildmatch", gitignore.read_text(errors="replace").splitlines()
            )

    exclude_patterns: list[str] = config.get("rag", {}).get("exclude", [])
    exc_spec: pathspec.PathSpec | None = (
        pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns) if exclude_patterns else None
    )

    files: list[Path] = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(workspace))
        if git_spec and git_spec.match_file(rel):
            continue
        if exc_spec and exc_spec.match_file(rel):
            continue
        files.append(p)
    return files


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += step
    return chunks


def _load_embedder(model_name: str, console: Any | None, _retry: bool = False) -> Any:
    """Return a TextEmbedding instance, auto-clearing a partial cache on first failure."""
    from fastembed import TextEmbedding

    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:
        msg = str(exc)
        is_missing = any(kw in msg for kw in ("NoSuchFile", "File doesn't exist", "doesn't exist"))
        if not _retry and is_missing:
            model_slug = "models--" + model_name.replace("/", "--")
            partial_dir = _fastembed_cache_dir() / model_slug
            if partial_dir.exists():
                shutil.rmtree(partial_dir, ignore_errors=True)
            if console is not None:
                console.print(
                    "[yellow][rag] Partial model cache cleared — re-downloading...[/yellow]"
                )
            return _load_embedder(model_name, console, _retry=True)
        raise


def index_workspace(workspace: Path, config: dict[str, Any], console: Any | None = None) -> None:
    """Chunk and upsert all workspace files into ChromaDB. Imports are deferred."""
    # Late imports — ChromaDB adds ~1-2s startup cost; don't pay it at module load.
    import chromadb

    rag_cfg: dict[str, Any] = config.get("rag", {})
    vector_store_dir = Path(
        str(rag_cfg.get("vector_store_dir", "~/.locoder/vectorstore"))
    ).expanduser()
    chunk_size: int = int(rag_cfg.get("chunk_size", 512))
    overlap: int = int(rag_cfg.get("chunk_overlap", 64))
    embed_model_name: str = str(rag_cfg.get("embeddings_model", _EMBED_MODEL))

    files = _collect_files(workspace, config)

    if len(files) > _MAX_FILES_WARN and console is not None:
        console.print(
            f"[yellow][rag] {len(files):,} files found — indexing may take a while. "
            "Add patterns to [rag] exclude in .locoder.toml to skip build artefacts.[/yellow]"
        )

    model_slug = "models--" + embed_model_name.replace("/", "--")
    if not (_fastembed_cache_dir() / model_slug).exists() and console is not None:
        console.print(
            "[dim][rag] First run: downloading fastembed embedding model (~250 MB). "
            "This happens once — subsequent starts are instant.[/dim]"
        )

    if console is not None:
        console.print(
            f"[dim][rag] Indexing {len(files)} files (model: {embed_model_name})...[/dim]"
        )

    embedder = _load_embedder(embed_model_name, console)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_store_dir))
    collection = client.get_or_create_collection(_collection_name(workspace))

    documents: list[str] = []
    metadatas: list[_Metadata] = []
    ids: list[str] = []

    for file_path in files:
        try:
            text = file_path.read_text(errors="replace")
        except OSError:
            continue
        rel = str(file_path.relative_to(workspace))
        mtime: float = file_path.stat().st_mtime
        for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
            documents.append(chunk)
            metadatas.append({"file": rel, "chunk": i, "mtime": mtime})
            ids.append(f"{rel}::chunk{i}")

    if not documents:
        return

    batch = 100
    for start in range(0, len(documents), batch):
        batch_docs = documents[start : start + batch]
        raw_embeddings = [list(e) for e in embedder.embed(batch_docs)]
        # chromadb accepts Sequence[float] per chunk; cast silences the ndarray overload mismatch
        embeddings: list[list[float]] = [[float(v) for v in vec] for vec in raw_embeddings]
        collection.upsert(
            ids=ids[start : start + batch],
            documents=batch_docs,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas[start : start + batch],  # type: ignore[arg-type]
        )

    if console is not None:
        console.print(f"[dim][rag] Done — {len(documents)} chunks from {len(files)} files.[/dim]")


def search(query: str, config: dict[str, Any], workspace: Path) -> str:
    """Query the ChromaDB collection; warn on stale source files."""
    import chromadb

    rag_cfg: dict[str, Any] = config.get("rag", {})
    vector_store_dir = Path(
        str(rag_cfg.get("vector_store_dir", "~/.locoder/vectorstore"))
    ).expanduser()
    top_k: int = int(rag_cfg.get("top_k", 5))
    embed_model_name: str = str(rag_cfg.get("embeddings_model", _EMBED_MODEL))

    if not vector_store_dir.exists():
        return "Knowledge base not yet indexed — run /reindex or restart LoCoder."

    try:
        client = chromadb.PersistentClient(path=str(vector_store_dir))
        collection = client.get_collection(_collection_name(workspace))
    except Exception:
        return "Knowledge base not yet indexed — run /reindex or restart LoCoder."

    embedder = _load_embedder(embed_model_name, None)
    query_vec: list[float] = [float(v) for v in next(iter(embedder.embed([query])))]

    _include: list[IncludeEnum] = ["documents", "metadatas"]  # type: ignore[list-item]
    results = collection.query(
        query_embeddings=[query_vec],  # type: ignore[arg-type]
        n_results=top_k,
        include=_include,
    )

    docs: list[str] = (results.get("documents") or [[]])[0]
    metas: list[Any] = (results.get("metadatas") or [[]])[0]

    if not docs:
        return f"No results found for: {query!r}"

    stale: list[str] = []
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        rel = str(meta.get("file", ""))
        indexed_mtime = float(meta.get("mtime", 0))
        p = workspace / rel
        if p.exists() and p.stat().st_mtime > indexed_mtime and rel not in stale:
            stale.append(rel)

    lines: list[str] = []
    for doc, meta in zip(docs, metas, strict=False):
        label = meta.get("file", "?") if isinstance(meta, dict) else "?"
        lines.append(f"[{label}]\n{doc.strip()}")

    output = "\n\n".join(lines)

    if stale:
        truncated = stale[:3]
        suffix = f" (+{len(stale) - 3} more)" if len(stale) > 3 else ""
        output += (
            f"\n\n[rag] Note: {len(stale)} file(s) modified since last index "
            f"({', '.join(truncated)}{suffix}). Run /reindex for fresh results."
        )

    return output
