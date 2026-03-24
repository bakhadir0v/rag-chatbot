import logging
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from app.core.config import get_settings
from app.rag.ingestion import load_index

logger = logging.getLogger(__name__)

_index_cache: FAISS | None = None


def get_index() -> FAISS:
    global _index_cache
    if _index_cache is None:
        _index_cache = load_index()
    return _index_cache


def invalidate_cache() -> None:
    global _index_cache
    _index_cache = None


def retrieve(query: str) -> list[Document]:
    """
    Retrieve top-k chunks above the similarity threshold.
    Returns an empty list if no relevant chunks are found.
    """
    settings = get_settings()
    index = get_index()

    # LangChain FAISS cosine strategy normalizes vectors and uses L2 distance.
    # Convert L2 distance → cosine similarity: cos_sim = 1 - L2² / 2
    results_with_scores = index.similarity_search_with_score(query, k=settings.top_k)

    docs = []
    for doc, l2_dist in results_with_scores:
        cos_sim = 1.0 - (l2_dist ** 2) / 2.0
        if cos_sim >= settings.similarity_threshold:
            docs.append(doc)
    logger.info(
        "Retrieved %d/%d chunks above threshold %.2f for query: %.60s...",
        len(docs),
        settings.top_k,
        settings.similarity_threshold,
        query,
    )
    return docs
