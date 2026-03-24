import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document, AIMessage


def make_doc(content: str = "Test content", source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source, "page": 1})


@patch("app.rag.chain.retrieve")
@patch("app.rag.chain.get_llm")
def test_ask_returns_answer(mock_llm, mock_retrieve):
    mock_retrieve.return_value = [make_doc("The answer is 42.")]
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="The answer is 42.")
    mock_llm.return_value = mock_llm_instance

    from app.rag.chain import ask
    result = ask("What is the answer?")

    assert "answer" in result
    assert "sources" in result
    assert "latency_ms" in result
    assert result["retrieved_chunks"] == 1


@patch("app.rag.chain.retrieve")
def test_ask_no_docs_returns_fallback(mock_retrieve):
    mock_retrieve.return_value = []

    from app.rag.chain import ask
    result = ask("Unanswerable question?")

    assert result["retrieved_chunks"] == 0
    assert "don't have enough information" in result["answer"]
