"""Fixed notices for questions that touch law-sensitive topics.

The model never writes these. They are matched in code and emitted verbatim,
for the same reason page numbers in citations are derived from metadata rather
than generated: anything the model composes, it can also confabulate — and this
text is legal-adjacent, so a plausible-sounding invention is worse here than
anywhere else in the app.

Because the wording is fixed, it can be reviewed once by the board or the
managing agent and then trusted. Model-generated text differs on every run and
could never be signed off in the same way.

What a notice must NOT do: name a statute, state what any law requires, or
predict how a request would turn out. It says only that a separate process
exists and that this app cannot speak to it — both verifiable without legal
expertise. Keep it that way when editing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Notice:
    topic: str
    pattern: re.Pattern[str]
    text: str


def _terms(*terms: str) -> re.Pattern[str]:
    """Match any term at a word boundary. Prefixes are deliberate: 'accessib'
    catches accessible/accessibility, 'discriminat' catches discrimination."""
    return re.compile("|".join(rf"\b{t}" for t in terms), re.IGNORECASE)


# Ordered most-specific first — at most one notice fires (see match_notice).
NOTICES: list[Notice] = [
    Notice(
        topic="assistance-animals",
        pattern=_terms(
            r"service animal", r"service dog", r"support animal",
            r"emotional support", r"esa\b", r"guide dog", r"seeing eye",
            r"assistance animal", r"therapy animal", r"companion animal",
        ),
        text=(
            "The house rules may not be the full picture. Questions about "
            "assistance and support animals in New York housing can involve a "
            "disability-accommodation process that operates separately from a "
            "co-op's own rules. CoopAgent has only the co-op's documents "
            "indexed and cannot tell you how that process applies to you. To "
            "make a request or get a definitive answer, write to the board or "
            "the managing agent."
        ),
    ),
    Notice(
        topic="accommodation",
        pattern=_terms(
            r"accommodation", r"disabilit", r"disabled", r"wheelchair",
            r"accessib", r"handicap", r"mobility impair", r"walker\b",
        ),
        text=(
            "The house rules may not be the full picture. Requests to change a "
            "rule or a physical feature of the building for disability-related "
            "reasons follow a process that operates separately from the co-op's "
            "own rules. CoopAgent has only the co-op's documents indexed and "
            "cannot tell you how that process applies to you. Write to the "
            "board or the managing agent to make a request."
        ),
    ),
    Notice(
        topic="tenancy-selection",
        pattern=_terms(
            r"sublet", r"subletting", r"sublease", r"discriminat",
            r"source of income", r"section 8", r"voucher", r"familial status",
            r"national origin", r"succession",
        ),
        text=(
            "The house rules may not be the full picture. Sublets, tenant "
            "selection and succession in New York are subject to protections "
            "that operate separately from co-op rules. CoopAgent has only the "
            "co-op's documents indexed. Direct questions to the managing agent; "
            "for advice about your own situation, contact a housing attorney or "
            "the relevant civil rights agency."
        ),
    ),
]


# --- Output verification -----------------------------------------------------
#
# Even told to stay inside the documents, llama3 volunteers legal claims: asked
# about emotional support animals it produced "under the ADA and FHA, emotional
# support animals are considered service animals", which is the opposite of what
# the ADA says. Prompt wording could not reliably stop this without breaking the
# citation contract, so offending sentences are removed after generation.

LEGAL_CLAIM_RE = re.compile(
    r"\b(ADA|Americans with Disabilities|Fair Housing|FHA|HUD|"
    r"Human Rights Law|Civil Rights Act|statut|state law|federal law|"
    r"legally (?:required|entitled|obligated|protected)|"
    r"the law (?:requires|says|states|protects)|"
    r"under (?:the )?(?:federal|state|city) law)\b",
    re.IGNORECASE,
)

_SENTENCES = re.compile(r"(?<=[.!?])\s+")

SAFE_FALLBACK = (
    "The co-op documents don't answer this question. Please contact the board "
    "or the managing agent."
)


def redact_legal_claims(answer: str) -> tuple[str, bool]:
    """Drop sentences asserting anything about law. Returns (text, redacted?).

    Sentence-level rather than all-or-nothing so the useful half survives: an
    answer that correctly quotes rule 19.1 and then invents an ADA claim keeps
    the rule and loses the invention. If nothing survives, the caller should
    fall back to SAFE_FALLBACK rather than show an empty answer.
    """
    kept_paragraphs: list[str] = []
    redacted = False
    for paragraph in answer.split("\n\n"):
        sentences = _SENTENCES.split(paragraph)
        keep = [s for s in sentences if not LEGAL_CLAIM_RE.search(s)]
        if len(keep) != len(sentences):
            redacted = True
        if keep:
            kept_paragraphs.append(" ".join(keep).strip())
    return "\n\n".join(p for p in kept_paragraphs if p).strip(), redacted


def match_notice(question: str) -> Notice | None:
    """The single notice for this question, or None.

    Matched against the resident's own wording ONLY, never the retrieved
    chunks. Matching retrieved text would fire the assistance-animal notice on
    every question that happens to pull in the pets section — including "are
    dogs allowed" — and a notice that shows up everywhere stops being read,
    including on the one question where it mattered.

    At most one fires, first match wins, so the list order above is the
    priority order. A question spanning two topics ("can I sublet to a voucher
    holder and keep a service dog") shows only the first; that is accepted in
    exchange for never stacking notices on a single answer.
    """
    for notice in NOTICES:
        if notice.pattern.search(question):
            return notice
    return None
