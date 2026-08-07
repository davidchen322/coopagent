"""FastAPI server: streaming chat endpoint + a minimal web UI.

    uvicorn app.server:app --reload      # then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from chromadb.errors import InvalidCollectionException
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.logs import get_logger
from app.notices import SAFE_FALLBACK, match_notice, redact_legal_claims
from app.rag import build_chain, doc_tag, get_retriever, source_list

app = FastAPI(title="CoopAgent")
WEB_DIR = Path(__file__).parent / "web"
log = get_logger("chat")

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
    log.info("reloaded — %d chunk(s) in the collection", count)
    return {"status": "reloaded", "chunks": count}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()

    async def event_stream():
        # Short id so interleaved requests from different residents stay
        # readable in the log.
        rid = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        log.info("%s question received (%d chars)", rid, len(question))
        log.debug("%s question: %s", rid, question)

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
        retrieval_started = time.perf_counter()
        try:
            docs = await retriever.ainvoke(question)
        except InvalidCollectionException:
            log.warning("%s collection is gone (re-ingest?) — reopening", rid)
            retriever, chain = await _refresh_handles()
            docs = await retriever.ainvoke(question)

        log.info(
            "%s retrieved %d chunk(s) in %s: %s",
            rid, len(docs), _ms(retrieval_started), ", ".join(source_list(docs)) or "-",
        )
        for d in docs:
            log.debug("%s   %s :: %s", rid, doc_tag(d),
                      d.page_content[:100].replace("\n", " "))

        yield _sse({"type": "sources", "sources": source_list(docs)})

        # 2) Law-sensitive topics get a fixed, human-reviewed notice. Matched in
        #    code and sent verbatim — the model is never asked to produce it, and
        #    never sees it, so it cannot reword or contradict it.
        notice = match_notice(question)
        if notice is not None:
            log.info("%s notice: %s", rid, notice.topic)
            yield _sse({"type": "notice", "topic": notice.topic, "text": notice.text})

        # 3) Ordinary questions stream token by token. Law-sensitive ones are
        #    buffered instead and checked before anything is shown: a fabricated
        #    legal claim cannot be unsent once it has been streamed to the screen.
        #    The cost is losing the streaming effect on those questions only.
        generation_started = time.perf_counter()
        try:
            if notice is None:
                length = 0
                async for token in chain.astream(question):
                    length += len(token)
                    yield _sse({"type": "token", "text": token})
                log.info("%s streamed %d chars in %s", rid, length,
                         _ms(generation_started))
            else:
                yield _sse({"type": "status", "text": "Checking the documents…"})
                parts = [token async for token in chain.astream(question)]
                answer, redacted = redact_legal_claims("".join(parts))
                if redacted:
                    log.warning("%s redacted invented legal claim(s) [%s]",
                                rid, notice.topic)
                if not answer:
                    log.warning("%s answer was entirely legal claim — using fallback",
                                rid)
                log.info("%s buffered %d chars in %s", rid, len(answer),
                         _ms(generation_started))
                yield _sse({"type": "answer", "text": answer or SAFE_FALLBACK})
        except Exception as exc:  # surface errors in the UI instead of a dead stream
            log.error("%s generation failed: %s: %s", rid, type(exc).__name__, exc)
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        log.info("%s done in %s", rid, _ms(started))
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _ms(since: float) -> str:
    return f"{(time.perf_counter() - since) * 1000:.0f}ms"
