import logging
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info("Loaded %d pages from %s", len(documents), file_path)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def build_faiss_index(chunks: list[Document]) -> FAISS:
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(
        chunks, embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    logger.info("Built FAISS index with %d vectors", len(chunks))
    return vector_store


def save_index(vector_store: FAISS) -> None:
    settings = get_settings()
    path = settings.faiss_index_path
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    logger.info("Saved FAISS index to %s", path)


def _index_exists(path: str) -> bool:
    return (Path(path) / "index.faiss").exists()


def load_index() -> FAISS:
    settings = get_settings()
    path = settings.faiss_index_path
    if not _index_exists(path):
        raise FileNotFoundError(f"No FAISS index found at {path}. Run ingestion first.")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info("Loaded FAISS index from %s", path)
    return vector_store


def ingest_pdf(file_path: str) -> int:
    """Full ingestion pipeline: load → chunk → embed → save. Returns chunk count."""
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)

    settings = get_settings()
    index_path = settings.faiss_index_path

    # Merge into existing index if one already exists
    if _index_exists(index_path):
        existing = load_index()
        embeddings = get_embeddings()
        new_store = FAISS.from_documents(
            chunks, embeddings, distance_strategy=DistanceStrategy.COSINE
        )
        existing.merge_from(new_store)
        save_index(existing)
    else:
        vector_store = build_faiss_index(chunks)
        save_index(vector_store)

    return len(chunks)
