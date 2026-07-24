"""Ingestion: load co-op documents -> chunk -> embed -> persist to Chroma.

Run whenever documents change:
    python -m app.ingest

This is deliberately a simple, character-based split for v1. The next quality
step is structure-aware chunking (split bylaws by section, minutes by meeting/
agenda item) with richer metadata — see README.
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DOCUMENTS_DIR, settings
from app.store import get_vectorstore

# Which loader handles which file type.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def load_documents(folder: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            print(f"  skip (unsupported type): {path.name}")
            continue
        print(f"  load: {path.name}")
        loaded = loader_cls(str(path)).load()
        # Stamp a clean source name on every page/segment so we can cite it later.
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def main() -> int:
    print(f"Documents dir: {DOCUMENTS_DIR}")
    if not DOCUMENTS_DIR.exists():
        print("Missing data/documents/. Create it and drop files in.")
        return 1

    print("Loading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        print("Nothing loadable found. Add PDF/DOCX/MD/TXT files and re-run.")
        return 1

    print(f"Loaded {len(docs)} segment(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer paragraph -> line -> sentence -> word boundaries.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Produced {len(chunks)} chunk(s).")

    print(
        f"Embedding with {settings.embed_provider}:{settings.embed_model} "
        f"and writing to Chroma..."
    )
    store = get_vectorstore()
    # Rebuild cleanly so re-runs don't pile up duplicates.
    try:
        store.delete_collection()
    except Exception:
        pass
    store = get_vectorstore()
    store.add_documents(chunks)

    print(
        f"Done. Indexed {len(chunks)} chunks into collection "
        f"'{settings.collection_name}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
