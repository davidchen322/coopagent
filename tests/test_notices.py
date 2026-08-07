"""Notice matching and legal-claim redaction.

Pure functions — no store, no model, no network. These are the guarantees the
app actually rests on, so they are the tests that must never be allowed to rot.
"""
from __future__ import annotations

import pytest

from app.notices import (
    LEGAL_CLAIM_RE,
    NOTICES,
    SAFE_FALLBACK,
    match_notice,
    redact_legal_claims,
)


@pytest.mark.parametrize(
    "question,expected",
    [
        # --- should fire -----------------------------------------------------
        ("Does the building allow for emotional support animals?", "assistance-animals"),
        ("Can I bring my service dog into the elevator?", "assistance-animals"),
        ("Is an ESA allowed?", "assistance-animals"),
        ("what about a guide dog", "assistance-animals"),
        ("Can I install a wheelchair ramp?", "accommodation"),
        ("I need an accommodation for my disability", "accommodation"),
        ("Is the lobby accessible?", "accommodation"),
        ("Can I sublet my apartment?", "tenancy-selection"),
        ("Do you accept Section 8 vouchers?", "tenancy-selection"),
        ("what are the succession rights", "tenancy-selection"),
        # --- should stay silent ----------------------------------------------
        ("Am I allowed to have pets in the building?", None),
        ("Where are the fireplace logs kept?", None),
        ("What are the rules about e-bikes?", None),
        ("Can I store boxes in the basement?", None),
        ("What time can I run the washing machine?", None),
        ("Who do I call about a leak?", None),
    ],
)
def test_match_notice(question, expected):
    got = match_notice(question)
    assert (got.topic if got else None) == expected


def test_pets_question_does_not_fire():
    """The wallpaper guard, stated as its own test because it is the whole
    reason matching ignores retrieved text: 'are pets allowed' retrieves the
    same chunk as the ESA question, and must still come back clean."""
    assert match_notice("Am I allowed to have pets in the building?") is None


def test_only_one_notice_fires():
    """Most-specific wins; notices never stack on a single answer."""
    both = "Can I sublet to a voucher holder and keep a service dog?"
    assert match_notice(both).topic == "assistance-animals"


def test_notice_text_makes_no_legal_assertion():
    """A notice that named a statute would be doing the thing it exists to
    prevent — so the notices are held to their own redaction rule."""
    for notice in NOTICES:
        assert not LEGAL_CLAIM_RE.search(notice.text), notice.topic


# --- redaction ---------------------------------------------------------------

# Verbatim llama3 output. It asserts the opposite of what the ADA says, and it
# recurred on every run of the ESA question, so it is the canonical fixture.
REAL_LEAK = (
    'Based on the provided context from the house rules [house_rules_2025.pdf], '
    'rule 19.1 states that "No dogs, cats or any other animals may be kept in '
    'the building at any time." This categorical rule applies to all types of '
    "animals, including emotional support animals.\n\n"
    "As for New York State laws regarding emotional support animals, I am not "
    "aware of any specific information provided in the context. However, "
    "according to the Americans with Disabilities Act (ADA) and the Fair "
    "Housing Act (FHA), emotional support animals are considered service "
    "animals under certain circumstances. If you are seeking guidance on this "
    "topic, I recommend contacting the co-op board for further clarification."
)


def test_redaction_keeps_the_answer_and_drops_the_invention():
    cleaned, redacted = redact_legal_claims(REAL_LEAK)
    assert redacted
    assert "19.1" in cleaned                      # the useful half survives
    assert "[house_rules_2025.pdf]" in cleaned    # so does the citation
    assert "ADA" not in cleaned
    assert "Fair Housing" not in cleaned
    assert "contacting the co-op board" in cleaned


def test_clean_answer_is_untouched():
    clean = "According to rule 18.1, logs are kept in the Log Room [house_rules_2025.pdf]."
    assert redact_legal_claims(clean) == (clean, False)


@pytest.mark.parametrize(
    "text",
    [
        "Under federal law you are entitled to this.",
        "The Fair Housing Act requires the co-op to allow it.",
        "This is protected by the NY State Human Rights Law.",
        "HUD guidance says otherwise.",
        "You are legally entitled to an exception.",
    ],
)
def test_bare_legal_claims_are_removed_entirely(text):
    cleaned, redacted = redact_legal_claims(text)
    assert redacted
    assert cleaned == ""  # caller substitutes SAFE_FALLBACK


def test_fallback_makes_no_legal_assertion():
    assert not LEGAL_CLAIM_RE.search(SAFE_FALLBACK)


def test_paragraph_structure_survives():
    text = "Rule 19.1 bans animals.\n\nSeparately, the Log Room is in the basement."
    cleaned, redacted = redact_legal_claims(text)
    assert not redacted
    assert cleaned.count("\n\n") == 1
