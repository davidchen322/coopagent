"""Shared stubs.

Deliberately minimal: nothing here opens Chroma, calls Ollama, or touches the
network, and no test asserts on anything Chroma-specific. The server only ever
calls `retriever.ainvoke()` and `chain.astream()`, so plain objects satisfy it.

That matters beyond speed. Swapping Chroma for pgvector would replace
app/store.py and the retriever it produces — but as long as the replacement
still returns Documents carrying `source` and `page`, every test in this suite
keeps passing unchanged. The seam is the retriever interface, not the store.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document


@pytest.fixture
def docs() -> list[Document]:
    """Two chunks from one file, mimicking real PyPDFLoader metadata.

    `page` is 0-indexed exactly as PyPDFLoader emits it, which is why doc_tag
    adds one.
    """
    return [
        Document(
            page_content="19.0 Pets\n19.1 No dogs, cats or any other animals "
            "may be kept in the building at any time.",
            metadata={"source": "house_rules_2025.pdf", "page": 11},
        ),
        Document(
            page_content="18.1 A supply of fireplace logs is maintained in the "
            "Log Room in the basement.",
            metadata={"source": "house_rules_2025.pdf", "page": 10},
        ),
    ]


class StubRetriever:
    """Returns fixed documents, or raises a queued exception once."""

    def __init__(self, documents, raises=None):
        self.documents = documents
        self.raises = raises
        self.calls = 0
        # Only /reload reaches through to the store.
        self.vectorstore = _StubStore(documents)

    async def ainvoke(self, _question):
        self.calls += 1
        if self.raises is not None:
            exc, self.raises = self.raises, None
            raise exc
        return self.documents


class _StubStore:
    def __init__(self, documents):
        self._documents = documents

    def get(self):
        return {"documents": [d.page_content for d in self._documents]}


class StubChain:
    """Yields a fixed answer as tokens, or raises mid-stream."""

    def __init__(self, answer: str = "", raises: Exception | None = None):
        self.answer = answer
        self.raises = raises

    async def astream(self, _question):
        if self.raises is not None:
            raise self.raises
        # Word-by-word, so tests exercise real multi-token accumulation.
        for i, word in enumerate(self.answer.split(" ")):
            yield word if i == 0 else " " + word


@pytest.fixture
def install(monkeypatch):
    """Point the server at stubs and clear its cached handles.

    Patches the factories rather than the globals so the lazy-build path in
    _handles() and the rebuild path in _refresh_handles() both run for real.
    """

    def _install(retrievers, chain):
        from app import server

        queue = list(retrievers)
        monkeypatch.setattr(server, "get_retriever", lambda: queue.pop(0))
        monkeypatch.setattr(server, "build_chain", lambda: chain)
        monkeypatch.setattr(server, "_retriever", None)
        monkeypatch.setattr(server, "_chain", None)
        return server

    return _install


def sse(response) -> list[dict]:
    """Parse an SSE response body into a list of event payloads."""
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def types(events) -> list[str]:
    return [e["type"] for e in events]
