# CLAUDE.md — coopagent

Guidance for Claude Code working in this repo. This project was scaffolded in a
prior session; this file carries over the architecture and the reasoning behind it.

## What this is

A **RAG** (retrieval-augmented generation) app that answers questions about a
residential co-operative from its own documents (bylaws, house rules, lease,
meeting minutes). A home project for David Chen; runs locally with an optional
cloud model. Repo: `git@github.com:davidchen322/coopagent.git`.

**It is RAG, not fine-tuning.** Documents change (new minutes, amended bylaws),
answers must cite sources, and it must run on modest hardware — all reasons RAG
fits and fine-tuning does not. Don't propose training/fine-tuning a model on the
co-op data.

## Core design decision: local ⇄ cloud must both work

The user wants to run either a **self-hosted local LLM (Ollama)** or a **top-tier
cloud model (Anthropic)**, switchable by config. This is the central constraint.
It's implemented as isolated **swap points** — nothing else in the app knows which
provider is active:

- `app/config.py` — the single source of truth; reads `.env`. Flip `LLM_PROVIDER`
  here (`ollama` | `anthropic`).
- `app/llm.py` — `get_chat_model()` returns a `ChatOllama` or `ChatAnthropic`.
- `app/embeddings.py` — `get_embeddings()`; local Ollama embeddings by default.

When changing behavior, preserve this separation — don't scatter
`if provider == ...` checks through the codebase.

## Architecture

```
Browser UI ──HTTP/SSE──> FastAPI (app/server.py) ──> RAG chain (app/rag.py)
                                                        │            │
                                               retrieve │            │ generate
                                                        ▼            ▼
                                          Chroma (data/chroma)      LLM (ollama|anthropic)
                                          + embeddings
```

| File | Role |
|---|---|
| `app/config.py` | config + local/cloud switch (reads `.env`) |
| `app/llm.py` | chat model swap point |
| `app/embeddings.py` | embedding model (local by default) |
| `app/store.py` | Chroma vector store — **embedded/persistent, no server** |
| `app/ingest.py` | load → chunk → embed → persist (`python -m app.ingest`) |
| `app/rag.py` | retriever + prompt + LCEL chain, cited answers |
| `app/query.py` | CLI retrieval test (`python -m app.query "..."`) |
| `app/server.py` | FastAPI: `/chat` (SSE stream), `/health`, `/` |
| `app/web/index.html` | minimal streaming chat UI |
| `Dockerfile`, `docker-compose.yml` | app + Ollama services |

Stack: **LangChain** (chosen deliberately for RAG + memory + future agent tools in
one ecosystem), **Chroma** (embedded, SQLite + HNSW under the hood), **FastAPI** +
a hand-built web UI (the user wants to build the UI, not use Open WebUI),
**Ollama** via Docker for the local model.

## Build order (status: scaffold complete, not yet run end-to-end)

1. **Ingestion** — get docs into Chroma; verify chunks look right.
2. **Retrieval/query** — `app/query.py`; prove the right chunks come back BEFORE
   trusting answers. This is where RAG quality is won or lost.
3. **Server + UI** — already scaffolded; run and stream.
4. Later: memory, then agentic tools.

The scaffold has never been run with real dependencies or documents yet. First
real task is likely: `poetry install` (or `docker compose up`), add a document,
`python -m app.ingest`, then `python -m app.query`.

## Gotchas (learned / by design)

- **Same embedding model for ingest AND query.** Changing `EMBED_MODEL` requires
  re-running `app/ingest.py` — vectors from different models aren't comparable.
- **Restart the server after re-ingesting** — `app/server.py` builds the retriever
  once at import.
- **macOS + Docker Ollama is CPU-only** (no Metal passthrough), so it's slow. For
  speed, native Ollama (`brew install ollama`) + `OLLAMA_BASE_URL=http://localhost:11434`.
- **`/chat` retrieves twice** (once for sources, once inside the chain) — known v1
  shortcut; fold sources into the chain later.
- Chunking is currently naive character splitting. Biggest pending quality win is
  **structure-aware chunking** (bylaws by section, minutes by meeting/date) with
  rich metadata.
- **Privacy:** co-op docs + index are gitignored, never committed. Local mode keeps
  everything on-machine; Anthropic mode sends retrieved chunks to the API.

## Roadmap

- Structure-aware chunking + metadata (section, date) for citations & filtered retrieval
- Contextual retrieval (LLM-written context line prepended per chunk before embedding)
- Conversation memory (multi-turn follow-ups)
- Agentic tools (date-filtered minutes search, bylaw section lookup)
- Port retry + token-cost tracking from the earlier prototype into a LangChain callback

## Conventions

- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Never commit `.env`, `data/documents/*`, or `data/chroma/` (already gitignored).
- Keep the local/cloud swap confined to `config.py` / `llm.py` / `embeddings.py`.
