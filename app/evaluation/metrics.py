"""
Evaluation metrics for the RAG pipeline.

- retrieval_precision: fraction of retrieved chunks that are relevant
- answer_relevance: LLM-as-judge score [0.0–1.0]
"""
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def retrieval_precision(retrieved_docs: list[Document], relevant_doc_ids: list[str]) -> float:
    """
    Fraction of retrieved chunks whose source appears in the ground-truth relevant set.

    Args:
        retrieved_docs: docs returned by the retriever
        relevant_doc_ids: list of expected source filenames / IDs

    Returns:
        precision in [0, 1]
    """
    if not retrieved_docs:
        return 0.0
    hits = sum(
        1 for d in retrieved_docs
        if d.metadata.get("source", "") in relevant_doc_ids
    )
    return hits / len(retrieved_docs)


JUDGE_PROMPT = """\
You are an impartial evaluator. Given a question, a reference answer, and a generated answer,
rate the quality of the generated answer on a scale from 0.0 to 1.0.

Criteria:
- 1.0: Fully correct, complete, and faithful to the reference.
- 0.7: Mostly correct with minor omissions or rephrasing.
- 0.4: Partially correct but missing key details or has minor errors.
- 0.0: Incorrect, hallucinated, or irrelevant.

Respond with ONLY a float number, nothing else.

Question: {question}
Reference answer: {reference}
Generated answer: {generated}
"""


def answer_relevance_score(question: str, reference: str, generated: str) -> float:
    """LLM-as-judge: returns a float score in [0, 1]."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, generated=generated
    )
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("Could not parse judge score: %s", response.content)
        return 0.0
