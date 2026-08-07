"""Quick terminal test of the retrieval + answering path (no web layer).

    python -m app.query "Can I keep a dog?"

Prints what was retrieved (the part that most affects answer quality) and then
streams the model's answer.
"""
from __future__ import annotations

import sys

from app.config import settings
from app.notices import SAFE_FALLBACK, match_notice, redact_legal_claims
from app.rag import build_chain, doc_tag, get_retriever


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m app.query "your question"')
        return 1
    question = " ".join(sys.argv[1:])

    print(f"[provider={settings.llm_provider} model={settings.llm_model}]\n")

    docs = get_retriever().invoke(question)
    print("Retrieved chunks:")
    for i, d in enumerate(docs, 1):
        preview = d.page_content[:120].replace("\n", " ")
        print(f"  {i}. [{doc_tag(d)}] {preview}...")

    # Mirrors the server: law-sensitive questions are buffered and checked
    # before display, and carry the same fixed notice. The two paths must agree
    # — a claim the web UI strips must not survive here.
    notice = match_notice(question)
    chain = build_chain()

    print("\nAnswer:")
    if notice is None:
        for token in chain.stream(question):
            print(token, end="", flush=True)
        print()
    else:
        answer, redacted = redact_legal_claims(chain.invoke(question))
        print(answer or SAFE_FALLBACK)
        if redacted:
            print("\n[redacted invented legal claim(s)]")
        print(f"\n[notice: {notice.topic}]\n{notice.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
