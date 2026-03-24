import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag.chain import ask
from app.rag.ingestion import ingest_pdf
from app.rag.retriever import invalidate_cache

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    question: str
    prompt_variant: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieved_chunks: int
    latency_ms: float


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = ask(request.question, prompt_variant=request.prompt_variant)
    return ChatResponse(**result)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = ingest_pdf(tmp_path)
        invalidate_cache()  # force retriever to reload updated index
        logger.info("Ingested %s → %d chunks", file.filename, chunks)
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        filename=file.filename,
        chunks_indexed=chunks,
        message=f"Successfully indexed {chunks} chunks from {file.filename}",
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
