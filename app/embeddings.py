"""The embedding-model swap point.

IMPORTANT: ingest and query MUST use the same embedding model. If you change
EMBED_MODEL, re-run `python -m app.ingest` to rebuild the index from scratch.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.config import settings


def get_embeddings() -> Embeddings:
    provider = settings.embed_provider

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Unknown EMBED_PROVIDER: {provider!r}. "
        "Only 'ollama' is wired up so far — add a cloud embedder (e.g. Voyage) here."
    )
