# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

This project uses **uv** as the primary package manager. Always use `uv run` to execute Python commands:

```bash
uv sync              # Install dependencies
uv run pytest tests -v   # Run tests
uv run ruff format src scripts tests   # Format code
uv run ruff check src scripts tests --fix   # Lint
```

## Core Architecture

ProfiRAG is an advanced RAG framework (~11,500 lines) built on LlamaIndex with three query modes:

1. **Pipeline** (default): Fixed flow - PreRetrieval → HybridRetriever → Reranker → ResponseSynthesizer
2. **ReAct Agent**: Think → Act → Observe loop with dynamic tool selection
3. **PlanAgent**: Plan → Approve → Execute → Replan workflow

### Key Components

- **RAGPipeline** (`src/profirag/pipeline/rag_pipeline.py`): Central orchestration class. All ingestion, querying, and agent creation flows through here.
- **StorageRegistry** (`src/profirag/storage/registry.py`): Factory for vector stores. Supports Qdrant, PostgreSQL/pgvector, and local file storage.
- **RAGTools** (`src/profirag/agent/tools.py`): 10 agent tools including vector_search, keyword_search, multi_query_search, hyde_search, rewrite_query, rerank_results, filter_results, generate_answer, retrieve_and_answer, table_lookup.
- **HybridRetriever** (`src/profirag/retrieval/hybrid.py`): Combines dense vector search with BM25 sparse retrieval.
- **RAGConfig** (`src/profirag/config/settings.py`): Pydantic configuration loaded from `.env` file.

### Embedding Providers

Three embedding providers are supported:
- **OpenAI**: API-based embedding (default)
- **FastEmbed**: Local embedding via `FastEmbedEmbedding` class
- **Ollama**: Local embedding via Ollama's OpenAI-compatible API endpoint

Provider selection via `PROFIRAG_EMBEDDING_PROVIDER` env var (openai, fastembed, ollama).

For Ollama:
- Run `ollama serve` to start Ollama server
- Run `ollama pull nomic-embed-text` to download embedding model
- Set `PROFIRAG_EMBEDDING_PROVIDER=ollama` in `.env`

### Agent Tool Context

Agent tools share state via `_last_retrieved_nodes` - tools that need prior retrieval (rerank, filter) check this attribute. `rewrite_query` has LLM fallback when QueryRewriter is unavailable.

## Configuration

All configuration via `.env` file. Key settings:

```bash
PROFIRAG_STORAGE_TYPE=qdrant      # qdrant, local, postgres
PROFIRAG_INDEX_MODE=hybrid        # hybrid (dense+BM25), vector
PROFIRAG_RETRIEVE_INDEX_MODE=hybrid  # hybrid, sparse, vector
PROFIRAG_EMBEDDING_PROVIDER=openai   # openai, fastembed
PROFIRAG_AGENT_ENABLED=false      # Enable agent mode
PROFIRAG_AGENT_MODE=react         # react, plan, pipeline
PROFIRAG_RERANK_ENABLED=true
PROFIRAG_RERANK_PROVIDER=local    # local, cohere, dashscope
```

See `.env.example` for full configuration options.

## Testing

Tests are organized by module in `tests/` directory:

```bash
uv run pytest tests -v                    # Run all tests
uv run pytest tests/config -v             # Run config tests
uv run pytest tests/embedding/test_fastembed.py -v  # Single test file
uv run pytest tests/integration -v        # Integration tests
```

Integration tests may require Qdrant running. Tests use pytest-asyncio with `asyncio_mode = "auto"`.

## Web Service

Backend (FastAPI) and Frontend (Vue 3):

```bash
# Backend
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd web/frontend && npm install && npm run dev
```

API routes in `web/api/routes/`: pdf.py, split.py, doc_import.py, chat.py, search.py

## Document Ingestion

```bash
uv run python scripts/ingest_documents.py --documents ./documents
uv run python scripts/ingest_documents.py --file ./documents/example.pdf
```

## Chinese Text Support

- `ChineseTextSplitter` for Chinese chunking
- jieba tokenization for BM25
- Chinese prompt templates in generation module
- Set `PROFIRAG_LANGUAGE=zh` for Chinese mode

## Entry Points

- `main.py`: Interactive CLI with `--mode` flag (pipeline/agent/plan)
- `scripts/ingest_documents.py`: Document ingestion CLI
- `web/api/main.py`: FastAPI web service entry

## Code Style

- Line length: 100 (ruff)
- Python 3.10 target
- Strict mypy enabled