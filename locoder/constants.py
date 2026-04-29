from __future__ import annotations

# Single source of truth for the default fastembed embedding model.
# Used by both config/manager.py (to write defaults) and agent/rag.py (as fallback).
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
