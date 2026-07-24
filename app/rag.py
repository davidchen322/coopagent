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
    "minutes). If the answer is not in the context, say you don't know and point "
    "them to the likely document or the board. Never invent rules. "
    "When you state a rule, cite the source document in brackets, e.g. [bylaws.pdf]."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        tag = src + (f" p.{page + 1}" if isinstance(page, int) else "")
        blocks.append(f"[{tag}]\n{d.page_content}")
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
