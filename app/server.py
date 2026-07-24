"""FastAPI server: streaming chat endpoint + a minimal web UI.

    uvicorn app.server:app --reload      # then open http://localhost:8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.rag import build_chain, get_retriever

app = FastAPI(title="CoopAgent")
WEB_DIR = Path(__file__).parent / "web"

# Built once at import. Re-ingesting rebuilds the index on disk; restart the
# server to pick up the new content.
_retriever = get_retriever()
_chain = build_chain()


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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()

    async def event_stream():
        # 1) Retrieve first so the UI can show sources immediately.
        #    (NOTE: the chain retrieves again internally — fine for v1; later,
        #    fold sources into the chain to avoid the second lookup.)
        docs = await _retriever.ainvoke(question)
        sources: list[str] = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            if src not in sources:
                sources.append(src)
        yield _sse({"type": "sources", "sources": sources})

        # 2) Stream the answer token by token.
        async for token in _chain.astream(question):
            yield _sse({"type": "token", "text": token})

        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
