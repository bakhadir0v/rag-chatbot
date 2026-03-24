import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from app.rag.ingestion import chunk_documents, ingest_pdf


def make_docs(n: int = 5) -> list[Document]:
    return [
        Document(page_content=f"This is test page {i} content. " * 20, metadata={"source": "test.pdf", "page": i})
        for i in range(n)
    ]


def test_chunk_documents_produces_chunks():
    docs = make_docs(3)
    chunks = chunk_documents(docs)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= 1200  # chunk_size + small buffer


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []


@patch("app.rag.ingestion.PyPDFLoader")
@patch("app.rag.ingestion.FAISS")
@patch("app.rag.ingestion.get_embeddings")
def test_ingest_pdf_returns_chunk_count(mock_embeddings, mock_faiss, mock_loader):
    mock_loader.return_value.load.return_value = make_docs(2)
    mock_embeddings.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        tmp_path = f.name

    try:
        count = ingest_pdf(tmp_path)
        assert count > 0
    finally:
        os.unlink(tmp_path)
