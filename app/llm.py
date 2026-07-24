"""The chat-model swap point: one function hands back a local or cloud model.

This is the crux of the local-and-cloud requirement. Nothing else in the app
knows or cares which provider is active.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings


def get_chat_model() -> BaseChatModel:
    provider = settings.llm_provider

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set (check .env)."
            )
        # ChatAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=0,
            max_tokens=1024,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (use 'ollama' or 'anthropic')"
    )
