"""Chroma vector store factory (embedded / persistent — no server needed).

Chroma runs in-process and persists to data/chroma/ (a chroma.sqlite3 file plus
the HNSW index). Delete that folder to wipe the index.
"""
from __future__ import annotations

import logging

from langchain_chroma import Chroma

from app.config import CHROMA_DIR, settings
from app.embeddings import get_embeddings

# chromadb 0.5.x ships a posthog telemetry call that is incompatible with the
# installed posthog client, so every operation logs "Failed to send telemetry
# event ...". Setting ANONYMIZED_TELEMETRY=False in .env does not gate the call
# in this version, so silence the logger directly. Nothing is being sent — this
# only hides the failure notice.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def get_vectorstore() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
