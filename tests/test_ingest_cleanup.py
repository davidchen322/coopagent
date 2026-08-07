"""Running-header stripping at load time.

The header ("Revised January 23th 2025; Supersedes all previous versions  12")
repeats on every page of the house rules. Left in, it lands in nearly every
chunk — diluting the embedding and parking a bare page number next to real rule
text, which is exactly the material llama3 turned into a fake citation.
"""
from __future__ import annotations

import pytest

from app.ingest import RUNNING_HEADER_RE


def strip(text: str) -> str:
    """The transformation load_documents applies to each page."""
    return RUNNING_HEADER_RE.sub("", text).strip()


def test_header_with_trailing_page_number_is_removed():
    page = (
        "Revised January 23th 2025; Supersedes all previous versions  12\n"
        "19.0 Pets\n19.1 No dogs, cats or any other animals may be kept."
    )
    assert strip(page).startswith("19.0 Pets")
    assert "Supersedes" not in strip(page)


def test_header_without_page_number_is_removed():
    assert strip("Revised 2025; Supersedes all previous versions\nRule text.") == "Rule text."


def test_case_is_ignored():
    assert strip("REVISED 2025; SUPERSEDES ALL PREVIOUS VERSIONS 3\nText.") == "Text."


def test_page_that_is_only_a_header_becomes_empty():
    """load_documents drops these, so they can't return as empty retrieval hits."""
    assert strip("Revised January 23th 2025; Supersedes all previous versions  7") == ""


def test_only_the_leading_header_is_stripped():
    """Anchored to the start — a mid-page mention of the same words is content."""
    body = "19.1 No pets.\nRevised 2025; Supersedes all previous versions 4"
    assert strip(body) == body


@pytest.mark.parametrize(
    "page",
    [
        "19.0 Pets\n19.1 No dogs, cats or any other animals may be kept.",
        "18.1 A supply of fireplace logs is maintained in the Log Room.",
    ],
)
def test_ordinary_pages_are_untouched(page):
    assert strip(page) == page
