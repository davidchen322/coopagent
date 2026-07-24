"""Quick terminal test of the retrieval + answering path (no web layer).

    python -m app.query "Can I keep a dog?"

Prints what was retrieved (the part that most affects answer quality) and then
streams the model's answer.
"""
from __future__ import annotations

import sys

from app.config import settings
from app.rag import build_chain, get_retriever


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m app.query "your question"')
        return 1
    question = " ".join(sys.argv[1:])

    print(f"[provider={settings.llm_provider} model={settings.llm_model}]\n")

    docs = get_retriever().invoke(question)
    print("Retrieved chunks:")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        preview = d.page_content[:120].replace("\n", " ")
        print(f"  {i}. [{src}] {preview}...")

    print("\nAnswer:")
    for token in build_chain().stream(question):
        print(token, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
