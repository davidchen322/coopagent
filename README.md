# coopagent

AI agent for answering questions about your co-op based on ingested data (bylaws, house rules, proprietary lease, building history, etc).

It's a **RAG** (retrieval-augmented generation) app: your documents are chunked,
embedded, and stored in a local vector database; each question retrieves the most
relevant passages and asks an LLM to answer *from them*, with citations. The LLM
can be a **local model (Ollama)** or a **top-tier cloud model (Anthropic)** — swap
with one setting.

## Architecture

```
Browser UI ──HTTP/SSE──> FastAPI (app/server.py)
                              │
                              ▼
                     RAG chain (app/rag.py)
                       │              │
              retrieve │              │ generate
                       ▼              ▼
              Chroma (data/chroma)   LLM  ── ollama (local)  ┐ swap via
              + embeddings                └─ anthropic (cloud) ┘ LLM_PROVIDER
```

| Concern | File | Notes |
|---|---|---|
| Config / local-vs-cloud switch | `app/config.py` | reads `.env` |
| Chat model swap point | `app/llm.py` | Ollama ⇄ Anthropic |
| Embedding model | `app/embeddings.py` | must match between ingest & query |
| Vector store | `app/store.py` | Chroma, embedded (no server) |
| Ingestion | `app/ingest.py` | load → chunk → embed → persist |
| RAG chain | `app/rag.py` | retrieve + prompt + answer |
| CLI test | `app/query.py` | prove retrieval before the web layer |
| Web server + UI | `app/server.py`, `app/web/` | streaming chat |

## Quick start (Docker, local model)

```bash
cp .env.example .env            # defaults to Ollama + llama3.1

docker compose up -d --build    # starts ollama + app

# Pull the models into the Ollama container (one time)
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text

# Add documents, then build the index
#   put PDFs/DOCX/MD/TXT in data/documents/
docker compose exec app python -m app.ingest

# Open the UI
open http://localhost:8000
```

> **macOS note:** Ollama in Docker is CPU-only on a Mac (no Metal passthrough), so
> it's slower. For speed, install Ollama natively (`brew install ollama`), set
> `OLLAMA_BASE_URL=http://localhost:11434`, and run only the `app` service — or
> just run everything locally without Docker (below).

## Quick start (no Docker)

```bash
poetry install
cp .env.example .env
# install & start Ollama, then: ollama pull llama3.1 && ollama pull nomic-embed-text

python -m app.ingest                       # after adding docs to data/documents/
python -m app.query "Can I keep a dog?"    # terminal test
uvicorn app.server:app --reload            # then http://localhost:8000
```

## Using a cloud model instead

In `.env`:

```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5      # or claude-sonnet-4-6 / claude-opus-4-8
ANTHROPIC_API_KEY=sk-ant-...
```

Embeddings still run locally by default, so retrieval stays private; only the
retrieved chunks are sent to Anthropic. Re-ingesting is **not** needed when you
change only the *chat* model — but **is** required if you change `EMBED_MODEL`.

## Roadmap / next quality steps

- **Structure-aware chunking** — split bylaws by section, minutes by meeting/agenda
  item; add rich metadata (section, date) for citations and filtered retrieval.
- **Contextual retrieval** — prepend an LLM-written context line to each chunk
  before embedding (Anthropic's technique) to sharpen matches.
- **Conversation memory** — multi-turn history so follow-ups work.
- **Agentic tools** — e.g. date-filtered minutes search, bylaw section lookup.
- **Cost/retry callback** — port the retry + token-cost logic from the earlier
  prototype into a LangChain callback.

## Privacy

Co-op documents and the built index are gitignored — they never leave your machine
in local mode. Switching the chat model to Anthropic sends *retrieved chunks* to
their API; decide per-document what's acceptable to send to the cloud.
