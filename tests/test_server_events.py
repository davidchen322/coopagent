"""The /chat event contract, and recovery from a re-ingest.

Every test here runs against stub objects — no Chroma, no Ollama, no network.
What is being asserted is the server's own logic: which events go out, in what
order, and whether a law-sensitive answer is checked before any of it reaches
the screen.

The InvalidCollectionException test is the important one. That failure was
found by accident against a running server after a re-ingest, where it returned
a 500 on every request until the process was restarted. This is its regression
test.
"""
from __future__ import annotations

import pytest
from chromadb.errors import InvalidCollectionException
from fastapi.testclient import TestClient

from conftest import StubChain, StubRetriever, sse, types

ORDINARY = "Where are the fireplace logs kept?"
LAW_SENSITIVE = "Does the building allow for emotional support animals?"

CLEAN_ANSWER = "Logs are kept in the Log Room [house_rules_2025.pdf]."
LEAKY_ANSWER = (
    "Rule 19.1 bans all animals [house_rules_2025.pdf]. "
    "Under the Fair Housing Act you are entitled to an exception."
)


def client(server) -> TestClient:
    return TestClient(server.app)


def ask(server, question: str) -> list[dict]:
    return sse(client(server).post("/chat", json={"question": question}))


# --- ordinary questions stream --------------------------------------------


def test_ordinary_question_streams_tokens(install, docs):
    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    events = ask(server, ORDINARY)

    assert types(events)[0] == "sources"
    assert "notice" not in types(events)
    assert "status" not in types(events)
    assert "answer" not in types(events)
    assert types(events)[-1] == "done"

    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == CLEAN_ANSWER


def test_sources_carry_real_page_numbers(install, docs):
    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    sources = next(e for e in ask(server, ORDINARY) if e["type"] == "sources")
    assert sources["sources"] == [
        "house_rules_2025.pdf p.12",
        "house_rules_2025.pdf p.11",
    ]


# --- law-sensitive questions are buffered and checked ----------------------


def test_law_sensitive_question_is_buffered_not_streamed(install, docs):
    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    events = ask(server, LAW_SENSITIVE)

    assert types(events) == ["sources", "notice", "status", "answer", "done"]
    # Nothing may be streamed: a fabricated legal claim cannot be unsent once
    # it has appeared on screen.
    assert not [e for e in events if e["type"] == "token"]


def test_notice_is_the_fixed_text_not_model_output(install, docs):
    from app.notices import NOTICES

    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    notice = next(e for e in ask(server, LAW_SENSITIVE) if e["type"] == "notice")

    expected = next(n for n in NOTICES if n.topic == "assistance-animals")
    assert notice["topic"] == "assistance-animals"
    assert notice["text"] == expected.text


def test_invented_legal_claim_is_stripped_before_display(install, docs):
    server = install([StubRetriever(docs)], StubChain(LEAKY_ANSWER))
    answer = next(e for e in ask(server, LAW_SENSITIVE) if e["type"] == "answer")

    assert "Fair Housing" not in answer["text"]
    assert "19.1" in answer["text"]                    # useful half kept
    assert "[house_rules_2025.pdf]" in answer["text"]  # citation kept


def test_answer_that_is_entirely_legal_claim_falls_back(install, docs):
    from app.notices import SAFE_FALLBACK

    server = install([StubRetriever(docs)], StubChain("Under federal law you may keep it."))
    answer = next(e for e in ask(server, LAW_SENSITIVE) if e["type"] == "answer")
    assert answer["text"] == SAFE_FALLBACK


# --- surviving a re-ingest -------------------------------------------------


def test_recovers_from_dropped_collection_without_restart(install, docs):
    """`python -m app.ingest` drops the collection and builds a new one with a
    fresh UUID, invalidating any cached handle. The server must reopen and
    answer, not 500."""
    stale = StubRetriever(docs, raises=InvalidCollectionException("gone"))
    fresh = StubRetriever(docs)
    server = install([stale, fresh], StubChain(CLEAN_ANSWER))

    events = ask(server, ORDINARY)

    assert "error" not in types(events)
    assert types(events)[-1] == "done"
    assert stale.calls == 1  # tried once
    assert fresh.calls == 1  # then reopened and succeeded


def test_retry_happens_only_once(install, docs):
    """A second failure is not a stale handle, so it must surface rather than
    loop."""
    always = StubRetriever(docs, raises=InvalidCollectionException("gone"))
    also = StubRetriever(docs, raises=InvalidCollectionException("still gone"))
    server = install([always, also], StubChain(CLEAN_ANSWER))

    with pytest.raises(InvalidCollectionException):
        ask(server, ORDINARY)


def test_reload_reopens_and_reports_chunk_count(install, docs):
    """The /reload endpoint, which had never been exercised."""
    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    body = client(server).post("/reload").json()
    assert body == {"status": "reloaded", "chunks": 2}


# --- failures are visible --------------------------------------------------


def test_model_failure_surfaces_as_an_error_event(install, docs):
    server = install([StubRetriever(docs)], StubChain(raises=RuntimeError("ollama down")))
    events = ask(server, ORDINARY)

    error = next(e for e in events if e["type"] == "error")
    assert "ollama down" in error["message"]
    assert types(events)[-1] == "done"  # stream still closes cleanly


def test_health_reports_the_active_provider(install, docs):
    server = install([StubRetriever(docs)], StubChain(CLEAN_ANSWER))
    body = client(server).get("/health").json()
    assert body["status"] == "ok"
    assert {"llm_provider", "llm_model", "embed_model"} <= body.keys()
