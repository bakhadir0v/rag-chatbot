import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.api.routes.ask")
def test_chat_success(mock_ask):
    mock_ask.return_value = {
        "answer": "Paris",
        "sources": [{"source": "geo.pdf", "page": 1}],
        "retrieved_chunks": 1,
        "latency_ms": 120.5,
    }
    response = client.post("/api/v1/chat", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris"
    assert data["retrieved_chunks"] == 1


def test_chat_empty_question():
    response = client.post("/api/v1/chat", json={"question": "   "})
    assert response.status_code == 400


@patch("app.api.routes.ingest_pdf")
@patch("app.api.routes.invalidate_cache")
def test_ingest_pdf_success(mock_invalidate, mock_ingest):
    mock_ingest.return_value = 42
    pdf_content = b"%PDF-1.4 fake pdf content"
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_indexed"] == 42
    assert "test.pdf" in data["filename"]


def test_ingest_non_pdf_rejected():
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
