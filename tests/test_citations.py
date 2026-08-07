"""Citation tags derived from metadata.

These are the citations the UI presents as authoritative. They must come from
metadata and never from model output — llama3 fabricated a page number (p.19,
lifted from rule number 19.1) across three separate prompt variants before this
was made deterministic. Nothing here depends on which vector store produced the
Documents.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.rag import doc_tag, format_docs, source_list


def test_page_is_one_indexed():
    """PyPDFLoader pages are 0-indexed; residents count from 1."""
    d = Document(page_content="x", metadata={"source": "a.pdf", "page": 11})
    assert doc_tag(d) == "a.pdf p.12"


def test_missing_page_is_omitted_not_guessed():
    d = Document(page_content="x", metadata={"source": "notes.md"})
    assert doc_tag(d) == "notes.md"


def test_missing_source_does_not_raise():
    assert doc_tag(Document(page_content="x", metadata={})) == "unknown"


def test_non_integer_page_is_ignored():
    d = Document(page_content="x", metadata={"source": "a.pdf", "page": "iv"})
    assert doc_tag(d) == "a.pdf"


def test_source_list_dedupes_and_preserves_order(docs):
    same_page = Document(page_content="more", metadata={"source": "house_rules_2025.pdf", "page": 11})
    assert source_list([docs[0], same_page, docs[1]]) == [
        "house_rules_2025.pdf p.12",
        "house_rules_2025.pdf p.11",
    ]


def test_context_given_to_the_model_has_no_page_numbers(docs):
    """The core of the fabrication fix: withhold the page, so there is no page
    to blur into a rule number. Real pages are attached by source_list()."""
    rendered = format_docs(docs)
    assert 'cite="house_rules_2025.pdf"' in rendered
    assert "p.12" not in rendered
    assert "p.11" not in rendered


def test_context_includes_the_passage_text(docs):
    assert "No dogs, cats or any other animals" in format_docs(docs)
