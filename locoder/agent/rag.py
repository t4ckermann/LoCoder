from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pathspec
from rich.console import Console

from locoder.constants import DEFAULT_EMBED_MODEL

if TYPE_CHECKING:
    from chromadb.api.types import IncludeEnum
    from fastembed import TextEmbedding

_MAX_FILES_WARN = 5_000

# Suppress chromadb telemetry. Must be set before `import chromadb` (which is a lazy import
# inside functions below). posthog v7 also changed capture() to 1-arg; chromadb calls the old
# 3-arg form and logs the resulting TypeError as an error — silence that logger too.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Directories that are almost never worth indexing; merged with user-configured excludes.
_DEFAULT_EXCLUDES: list[str] = [
    "**/.git/**",
    "**/.venv/**",
    "**/venv/**",
    "**/env/**",
    "**/__pycache__/**",
    "**/*.egg-info/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/.pytest_cache/**",
]


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

    user_excludes: list[str] = config.get("rag", {}).get("exclude", [])
    all_excludes = _DEFAULT_EXCLUDES + user_excludes
    exc_spec: pathspec.PathSpec | None = pathspec.PathSpec.from_lines("gitwildmatch", all_excludes)

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


def _load_embedder(model_name: str, console: Console | None, _retry: bool = False) -> TextEmbedding:
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


def index_workspace(
    workspace: Path, config: dict[str, Any], console: Console | None = None
) -> None:
    """Chunk and upsert changed workspace files into ChromaDB. Imports are deferred."""
    # Late imports — ChromaDB adds ~1-2s startup cost; don't pay it at module load.
    import chromadb

    rag_cfg: dict[str, Any] = config.get("rag", {})
    vector_store_dir = Path(
        str(rag_cfg.get("vector_store_dir", "~/.locoder/vectorstore"))
    ).expanduser()
    chunk_size: int = int(rag_cfg.get("chunk_size", 512))
    overlap: int = int(rag_cfg.get("chunk_overlap", 64))
    embed_model_name: str = str(rag_cfg.get("embeddings_model", DEFAULT_EMBED_MODEL))

    files = _collect_files(workspace, config)

    if len(files) > _MAX_FILES_WARN and console is not None:
        console.print(
            f"[yellow][rag] {len(files):,} files found — indexing may take a while. "
            "Add patterns to [rag] exclude in .locoder.toml to skip build artefacts.[/yellow]"
        )

    # Build rel -> mtime map for current files.
    file_mtimes: dict[str, float] = {}
    for fp in files:
        with contextlib.suppress(OSError):
            file_mtimes[str(fp.relative_to(workspace))] = fp.stat().st_mtime

    # Open collection first so we can diff against stored mtimes before loading the model.
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_store_dir))
    collection = client.get_or_create_collection(_collection_name(workspace))

    existing = collection.get(include=["metadatas"])  # type: ignore[list-item]
    existing_ids: list[str] = existing.get("ids") or []
    existing_metas: list[Any] = existing.get("metadatas") or []

    # Derive the highest mtime we already have indexed for each file.
    indexed_mtimes: dict[str, float] = {}
    for meta in existing_metas:
        if not isinstance(meta, dict):
            continue
        rel = str(meta.get("file", ""))
        mtime = float(meta.get("mtime", 0.0))
        if rel and mtime > indexed_mtimes.get(rel, 0.0):
            indexed_mtimes[rel] = mtime

    changed_rels = [rel for rel, mt in file_mtimes.items() if mt > indexed_mtimes.get(rel, 0.0)]
    removed_rels = {rel for rel in indexed_mtimes if rel not in file_mtimes}

    # Purge chunks for deleted or modified files.
    stale_rels = set(changed_rels) | removed_rels
    if stale_rels:
        ids_to_delete = [
            eid
            for eid, emeta in zip(existing_ids, existing_metas, strict=False)
            if isinstance(emeta, dict) and str(emeta.get("file", "")) in stale_rels
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    if not changed_rels:
        if console is not None:
            console.print(f"[dim][rag] Index up-to-date ({len(files)} files).[/dim]")
        return

    model_slug = "models--" + embed_model_name.replace("/", "--")
    if not (_fastembed_cache_dir() / model_slug).exists() and console is not None:
        console.print(
            "[dim][rag] First run: downloading fastembed embedding model (~250 MB). "
            "This happens once — subsequent starts are instant.[/dim]"
        )

    if console is not None:
        console.print(
            f"[dim][rag] Indexing {len(changed_rels)}/{len(files)} files "
            f"(model: {embed_model_name})...[/dim]"
        )

    documents: list[str] = []
    metadatas: list[_Metadata] = []
    ids: list[str] = []

    for rel in changed_rels:
        file_path = workspace / rel
        try:
            text = file_path.read_text(errors="replace")
        except OSError:
            continue
        mtime = file_mtimes[rel]
        for i, chunk in enumerate(_chunk_text(text, chunk_size, overlap)):
            documents.append(chunk)
            metadatas.append({"file": rel, "chunk": i, "mtime": mtime})
            ids.append(f"{rel}::chunk{i}")

    if not documents:
        return

    embedder = _load_embedder(embed_model_name, console)

    total_chunks = len(documents)
    batch: int = int(rag_cfg.get("embed_batch_size", 16))
    for start in range(0, total_chunks, batch):
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
            done = min(start + batch, total_chunks)
            console.print(f"[dim][rag] Embedded {done}/{total_chunks} chunks...[/dim]")

    if console is not None:
        console.print(
            f"[dim][rag] Done — {total_chunks} chunks from {len(changed_rels)} files.[/dim]"
        )


def search(query: str, config: dict[str, Any], workspace: Path) -> str:
    """Query the ChromaDB collection; warn on stale source files."""
    import chromadb

    rag_cfg: dict[str, Any] = config.get("rag", {})
    vector_store_dir = Path(
        str(rag_cfg.get("vector_store_dir", "~/.locoder/vectorstore"))
    ).expanduser()
    top_k: int = int(rag_cfg.get("top_k", 5))
    embed_model_name: str = str(rag_cfg.get("embeddings_model", DEFAULT_EMBED_MODEL))

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
