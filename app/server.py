"""FastAPI server: streaming chat endpoint + a minimal web UI.

    uvicorn app.server:app --reload      # then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from chromadb.errors import InvalidCollectionException
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.rag import build_chain, get_retriever, source_list

app = FastAPI(title="CoopAgent")
WEB_DIR = Path(__file__).parent / "web"

# Retriever/chain are cached, not rebuilt per request — constructing them opens
# the Chroma client and the embedding client, which is far too costly per call.
#
# But `python -m app.ingest` DROPS the collection and creates a new one with a
# fresh UUID, which leaves a cached handle pointing at something that no longer
# exists. Every later request then fails with InvalidCollectionException until
# the process restarts. So: build lazily, and rebuild once on that specific
# error. A re-ingest is picked up automatically on the next question.
_retriever = None
_chain = None
_rebuild_lock = asyncio.Lock()


def _build_handles() -> None:
    global _retriever, _chain
    _retriever = get_retriever()
    _chain = build_chain()


async def _handles():
    """Current (retriever, chain), building them on first use."""
    if _retriever is None or _chain is None:
        async with _rebuild_lock:
            if _retriever is None or _chain is None:  # re-check under the lock
                _build_handles()
    return _retriever, _chain


async def _refresh_handles():
    """Discard stale handles and reopen against the current collection."""
    async with _rebuild_lock:
        _build_handles()
    return _retriever, _chain


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
    }


@app.post("/reload")
async def reload() -> dict:
    """Force a reopen against the current collection.

    Not normally needed — /chat recovers on its own — but useful right after a
    re-ingest to fail fast rather than on a resident's first question.
    """
    retriever, _ = await _refresh_handles()
    count = len(retriever.vectorstore.get()["documents"])
    return {"status": "reloaded", "chunks": count}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()

    async def event_stream():
        retriever, chain = await _handles()

        # 1) Retrieve first so the UI can show sources immediately.
        #    (NOTE: the chain retrieves again internally — fine for v1; later,
        #    fold sources into the chain to avoid the second lookup.)
        #    These tags carry the real page numbers, straight from metadata —
        #    they are the citation to trust, not any page the model writes.
        #
        #    Retrieval happens before any token is streamed, so this is the safe
        #    place to notice a re-ingest and reopen. Retry once; if it fails
        #    again the problem is not a stale handle.
        try:
            docs = await retriever.ainvoke(question)
        except InvalidCollectionException:
            retriever, chain = await _refresh_handles()
            docs = await retriever.ainvoke(question)

        yield _sse({"type": "sources", "sources": source_list(docs)})

        # 2) Stream the answer token by token.
        try:
            async for token in chain.astream(question):
                yield _sse({"type": "token", "text": token})
        except Exception as exc:  # surface errors in the UI instead of a dead stream
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
