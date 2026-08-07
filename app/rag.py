"""RAG chain: retrieve relevant chunks and answer strictly from them."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.config import settings
from app.llm import get_chat_model
from app.store import get_vectorstore

SYSTEM_PROMPT = (
    "You are CoopAgent, an assistant for a residential co-operative. "
    "Answer the resident's question using ONLY the context below, which is drawn "
    "from the co-op's official documents (bylaws, house rules, lease, meeting "
    "minutes). Never invent rules.\n\n"
    "APPLYING RULES:\n"
    "- Apply the rules, don't merely quote them. A categorical rule still covers "
    "a specific case when the exact words are absent — a ban on 'any other "
    "animals' answers a question about one particular kind of animal. Say so.\n"
    "- Only say you don't know when the context is genuinely silent on the "
    "topic. If it is, say so and point them to the likely document or the board.\n"
    # Deliberately NOT a hard prohibition on naming laws. A strong "never mention
    # the ADA" rule was tried and it crowded out the citation contract below —
    # llama3 dropped [brackets] from every answer and regressed to "I don't know"
    # on the pets rule. Invented legal claims are stripped in code instead, by
    # app.notices.redact_legal_claims. Keep this instruction mild.
    "- Report what the documents say and stop there. Do not extend a rule into a "
    "conclusion about anything the documents don't address — law, enforceability, "
    "or a resident's individual circumstances.\n"
    "- Answer directly. Never narrate or restate these instructions.\n\n"
    "CITATIONS — follow exactly:\n"
    "- Each passage is wrapped as <passage cite=\"...\">. Copy that cite value "
    "verbatim into square brackets, e.g. [house_rules_2025.pdf].\n"
    "- NEVER add a page number to a citation. Do not write 'p.12', 'page 12', "
    "or any number inside the brackets — the document name alone is the whole "
    "citation. Exact page numbers are attached automatically afterwards.\n"
    "- Refer to a rule by its number in your prose (e.g. \"rule 19.1\"), but "
    "keep numbers out of the brackets."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def doc_tag(d: Document) -> str:
    """The one canonical citation string for a chunk: 'file.pdf p.12'.

    Derived from metadata only, never from the model — this is what the UI
    shows as the authoritative source list.
    """
    src = d.metadata.get("source", "unknown")
    page = d.metadata.get("page")
    return src + (f" p.{page + 1}" if isinstance(page, int) else "")


def source_list(docs: list[Document]) -> list[str]:
    """Deduplicated, order-preserving citation tags for retrieved chunks."""
    seen: list[str] = []
    for d in docs:
        tag = doc_tag(d)
        if tag not in seen:
            seen.append(tag)
    return seen


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt.

    The cite value deliberately omits the page number. Small local models blur
    the tag into the body text and cite a rule number (19.1) as if it were a
    page — llama3 did exactly that across three prompt variants. Withholding the
    page removes the chance to invent one; real page numbers are attached
    deterministically by source_list() instead.
    """
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        blocks.append(f'<passage cite="{src}">\n{d.page_content}\n</passage>')
    return "\n\n".join(blocks)


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.retrieval_k})


def build_chain():
    """LCEL chain: a question string in -> a streamed answer string out."""
    retriever = get_retriever()
    llm = get_chat_model()
    return (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
