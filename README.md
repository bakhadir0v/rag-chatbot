# RAG-Powered Document Q&A Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask natural language questions about their contents. Built with Python, FastAPI, LangChain, OpenAI, and Docker — with a full MLOps evaluation loop and Azure CI/CD pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup (without Docker)](#local-setup-without-docker)
  - [Docker Setup](#docker-setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation & MLOps](#evaluation--mlops)
- [Running Tests](#running-tests)
- [Azure Deployment](#azure-deployment)
- [Design Decisions](#design-decisions)

---

## Overview

This chatbot answers questions about your documents using RAG — rather than relying on an LLM's pre-trained knowledge, it retrieves the most relevant passages from your own PDFs and grounds every answer in that context. This eliminates hallucinations for domain-specific questions and keeps responses traceable back to a source page.

**Key capabilities:**
- Upload any PDF and start asking questions immediately
- Sub-second vector retrieval from a FAISS index
- Source citations with page numbers on every answer
- Graceful "I don't know" fallback when no relevant content is found
- A/B prompt variant testing and LLM-as-judge quality evaluation
- Azure-integrated CI/CD with automatic redeployment on each commit

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────────────────────────────┐
│  FastAPI    │────▶│            RAG Pipeline                  │
│  REST API   │     │                                          │
└─────────────┘     │  1. Embed question (text-embedding-3-small)│
                    │  2. Cosine similarity search (FAISS)      │
PDF Upload          │  3. Filter chunks (threshold = 0.4)       │
     │              │  4. Format context + prompt template      │
     ▼              │  5. Generate answer (GPT-4o)              │
┌─────────────┐     │  6. Return answer + sources + latency    │
│  Ingestion  │     └──────────────────────────────────────────┘
│  Pipeline   │
│             │     ┌──────────────────────────────────────────┐
│ Load PDF    │     │           Vector Store                   │
│ Chunk text  │────▶│  FAISS (local dev)                       │
│ Embed chunks│     │  Azure AI Search (production)            │
│ Store index │     └──────────────────────────────────────────┘
└─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **LLM Orchestration** | LangChain (LCEL) |
| **Chat Model** | OpenAI GPT-4o |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | FAISS (local) / Azure AI Search (production) |
| **PDF Parsing** | pypdf |
| **Evaluation** | RAGAS + custom LLM-as-judge |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | Azure Pipelines |
| **Config** | Pydantic Settings |
| **Testing** | pytest + pytest-asyncio |

---

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py                  # FastAPI app factory, CORS, routing
│   ├── core/
│   │   ├── config.py            # Pydantic settings (env vars, LRU cached)
│   │   └── prompts.py           # Prompt templates (default & verbose variants)
│   ├── rag/
│   │   ├── ingestion.py         # PDF load → chunk → embed → FAISS index
│   │   ├── retriever.py         # Cosine similarity search with threshold filter
│   │   └── chain.py             # LangChain LCEL chain: retrieve → prompt → LLM
│   ├── api/
│   │   └── routes.py            # POST /chat, POST /ingest, GET /health
│   └── evaluation/
│       ├── metrics.py           # Retrieval precision + LLM-as-judge scoring
│       └── eval_runner.py       # CLI: runs prompt variants against eval dataset
├── tests/
│   ├── test_api.py              # Endpoint integration tests
│   ├── test_ingestion.py        # Chunking and indexing unit tests
│   └── test_chain.py            # RAG chain unit tests with mocks
├── data/
│   ├── documents/               # Drop PDFs here for ingestion
│   ├── faiss_index/             # Persisted FAISS index (auto-created)
│   └── eval_dataset.json        # Ground-truth Q&A pairs for evaluation
├── Dockerfile                   # Multi-stage build, non-root user
├── docker-compose.yml           # App + persistent FAISS volume
├── azure-pipelines.yml          # 4-stage CI/CD: test → build → evaluate → deploy
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for containerized setup)
- An [OpenAI API key](https://platform.openai.com/api-keys) with available credits

### Local Setup (without Docker)

**1. Clone the repo and create a virtual environment:**

```bash
git clone <repo-url>
cd rag-chatbot
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment variables:**

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI key:

```env
OPENAI_API_KEY=sk-...
```

**3. Start the API server:**

```bash
uvicorn app.main:app --reload --port 8000
```

**4. Upload a PDF:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/your/document.pdf"
```

**5. Ask a question:**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

---

### Docker Setup

**1. Build and start the container:**

```bash
cp .env.example .env          # Add your OPENAI_API_KEY
docker compose up --build -d
```

**2. Verify it's running:**

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok"}
```

**3. Upload a PDF and chat:**

```bash
# Ingest
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/documents/my_doc.pdf"

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key points."}'
```

The FAISS index is stored in a named Docker volume (`faiss_data`) and persists across container restarts.

---

## API Reference

### `POST /api/v1/ingest`

Upload a PDF to be chunked, embedded, and added to the vector store.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | PDF file (`.pdf` extension required) |

**Response:**

```json
{
  "filename": "document.pdf",
  "chunks_indexed": 24,
  "message": "Successfully indexed 24 chunks from document.pdf"
}
```

---

### `POST /api/v1/chat`

Ask a question against all ingested documents.

**Request body:**

```json
{
  "question": "What are the main findings?",
  "prompt_variant": "default"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Your question |
| `prompt_variant` | string | `"default"` | `"default"` or `"verbose"` (adds source citation requirement) |

**Response:**

```json
{
  "answer": "The main findings indicate that... (Source: report.pdf, Page: 3)",
  "sources": [
    {"source": "report.pdf", "page": 3}
  ],
  "retrieved_chunks": 2,
  "latency_ms": 2341.5
}
```

> If no relevant chunks are found above the similarity threshold, the API returns `"I don't have enough information in the provided documents to answer that."` rather than hallucinating.

---

### `GET /api/v1/health`

Liveness probe for Docker and Kubernetes.

```json
{"status": "ok"}
```

---

## Configuration

All settings are controlled via environment variables (`.env` file):

```env
# ── OpenAI ───────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_TEMPERATURE=0               # 0 = deterministic, factual answers

# ── RAG Pipeline ─────────────────────────────────────
CHUNK_SIZE=1000                    # Characters per chunk
CHUNK_OVERLAP=200                  # Overlap between adjacent chunks
TOP_K=4                            # Max chunks to retrieve per query
SIMILARITY_THRESHOLD=0.4           # Min cosine similarity to include a chunk

# ── Vector Store ─────────────────────────────────────
VECTOR_STORE_TYPE=faiss            # "faiss" (local) or "azure" (production)
FAISS_INDEX_PATH=data/faiss_index

# ── Azure AI Search (production) ─────────────────────
AZURE_SEARCH_ENDPOINT=https://<your-service>.search.windows.net
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX_NAME=rag-documents

# ── App ──────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO
```

### Similarity Threshold

The threshold controls how strict retrieval is. Scores are cosine similarities derived from L2 distances on unit-normalized OpenAI embeddings:

| Score Range | Meaning |
|---|---|
| `0.7 – 1.0` | Highly relevant |
| `0.4 – 0.7` | Relevant (default cutoff at 0.4) |
| `0.0 – 0.4` | Loosely related — filtered out |
| `< 0.0` | Not relevant at all |

---

## Evaluation & MLOps

The `evaluation/` module implements an MLOps loop that tracks quality across prompt variants and gates deployment on metric thresholds.

### Evaluation Dataset

Create `data/eval_dataset.json` with ground-truth Q&A pairs:

```json
[
  {
    "question": "What is RAG?",
    "reference_answer": "Retrieval-Augmented Generation combines retrieval and generation.",
    "relevant_sources": ["ai_overview.pdf"]
  }
]
```

### Run an Evaluation

```bash
python -m app.evaluation.eval_runner \
  --dataset data/eval_dataset.json \
  --variants default verbose \
  --output data/eval_results.json
```

**Metrics tracked per variant:**

| Metric | Description | CI Gate |
|---|---|---|
| `retrieval_precision` | Fraction of retrieved chunks from ground-truth sources | ≥ 0.6 |
| `answer_relevance` | LLM-as-judge score vs. reference answer (0–1) | ≥ 0.7 |
| `latency_p95` | 95th percentile response time (ms) | — |

The evaluation stage runs automatically in the Azure Pipeline on every commit and blocks deployment if either gate fails.

---

## Running Tests

```bash
# All tests
pytest

# With verbose output
pytest -v

# Specific module
pytest tests/test_api.py -v
pytest tests/test_chain.py -v
```

Tests use mocks for all OpenAI API calls — no API key required and no costs incurred during testing.

---

## Azure Deployment

The `azure-pipelines.yml` defines a 4-stage pipeline:

```
Commit → [1] Test → [2] Build & Push to ACR → [3] Evaluate → [4] Deploy
```

| Stage | What it does |
|---|---|
| **Test** | Runs `pytest` suite on Ubuntu runner |
| **Build** | Builds Docker image, pushes to Azure Container Registry with `buildId` tag |
| **Evaluate** | Runs eval runner against `eval_dataset.json`, publishes results as artifact, enforces quality gates |
| **Deploy** | Updates Azure Container App image (main branch only) |

### Setup Requirements

1. Create an Azure Container Registry (ACR)
2. Create an Azure Container App
3. Set these pipeline variables in Azure DevOps:

```
ACR_NAME              = your-registry
CONTAINER_APP_NAME    = rag-chatbot-app
RESOURCE_GROUP        = your-resource-group
OPENAI_API_KEY        = sk-...  (mark as secret)
```

### Switch to Azure AI Search (Production)

Update `.env` (or Container App environment variables):

```env
VECTOR_STORE_TYPE=azure
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_KEY=your-key
AZURE_SEARCH_INDEX_NAME=rag-documents
```

---

## Design Decisions

**Why FAISS locally, Azure AI Search in production?**
FAISS runs entirely in-process with zero infrastructure. For production, Azure AI Search adds hybrid keyword+vector search, role-based access control, and managed scaling.

**Why cosine similarity and not L2?**
LangChain's FAISS wrapper uses L2 distance internally on normalized vectors. Cosine similarity is derived as `cos_sim = 1 - L2² / 2`, giving a human-interpretable [−1, 1] range. The 0.4 threshold was calibrated against actual embedding scores for `text-embedding-3-small`.

**Why temperature = 0?**
This is a factual Q&A system — deterministic outputs are preferred. Creativity would introduce variance and potential hallucination.

**Why chunk overlap of 200?**
Overlap ensures that sentences crossing chunk boundaries don't lose context. 200 characters (~2–3 sentences) is sufficient without causing excessive index bloat.

**Why LLM-as-judge for evaluation?**
Ground-truth string matching is brittle for natural language. GPT-4o scoring against a reference answer is more robust and aligns with how humans assess answer quality.

---

## Cost Estimate

Running the full project end-to-end is inexpensive:

| Operation | Model | Approx. Cost |
|---|---|---|
| Embed a 10-page PDF | `text-embedding-3-small` | ~$0.001 |
| Single Q&A exchange | `gpt-4o` | ~$0.005 |
| Full eval run (50 questions) | `gpt-4o` | ~$0.25 |

$5 in OpenAI credits is sufficient for extensive development and testing.

---

## License

MIT
