"""Central configuration — the single place that decides local vs. cloud.

Everything downstream reads `settings`; flip providers here (or in .env) and the
whole pipeline follows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env if present, into os.environ

# ---- Project paths ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"

# Default chat model per provider, used when LLM_MODEL is unset.
_DEFAULT_LLM_MODEL = {
    "ollama": "llama3.1",
    "anthropic": "claude-haiku-4-5",
}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    embed_provider: str
    embed_model: str
    ollama_base_url: str
    anthropic_api_key: str | None
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    collection_name: str = "coop"


def load_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    model = os.getenv("LLM_MODEL") or _DEFAULT_LLM_MODEL.get(provider, "llama3.1")
    return Settings(
        llm_provider=provider,
        llm_model=model,
        embed_provider=os.getenv("EMBED_PROVIDER", "ollama").strip().lower(),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
    )


settings = load_settings()
