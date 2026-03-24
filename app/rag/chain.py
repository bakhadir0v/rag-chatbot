import logging
import time
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import Document

from app.core.config import get_settings
from app.core.prompts import PROMPT_VARIANTS
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=settings.openai_temperature,
        openai_api_key=settings.openai_api_key,
    )


def ask(question: str, prompt_variant: str = "default") -> dict:
    """
    Run the full RAG pipeline for a question.

    Returns:
        answer: str
        sources: list of source metadata dicts
        retrieved_chunks: int
        latency_ms: float
    """
    t0 = time.perf_counter()

    prompt = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["default"])
    llm = get_llm()

    docs = retrieve(question)

    if not docs:
        return {
            "answer": "I don't have enough information in the provided documents to answer that.",
            "sources": [],
            "retrieved_chunks": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    context = format_context(docs)

    chain = (
        RunnablePassthrough()
        | (lambda _: {"context": context, "question": question})
        | prompt
        | llm
        | RunnableLambda(lambda msg: msg.content)
    )

    answer = chain.invoke(question)

    sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page")}
        for d in docs
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Answer generated in %.1fms | chunks=%d", latency_ms, len(docs))

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(docs),
        "latency_ms": latency_ms,
    }
