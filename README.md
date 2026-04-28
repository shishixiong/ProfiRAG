# ProfiRAG

Advanced RAG (Retrieval-Augmented Generation) framework built with LlamaIndex, supporting three query modes: Pipeline (fixed flow), ReAct Agent (dynamic reasoning), and PlanAgent (plan-then-execute).

~11,500 lines of Python, MIT licensed.

## Features

### Core RAG Pipeline

| Stage | Components | Description |
|-------|-----------|-------------|
| Pre-Retrieval | HyDE, Query Rewriting, Multi-Query | Transform/normalize queries before retrieval |
| Retrieval | Hybrid (Vector + BM25), RRF fusion | Multi-strategy dense + sparse retrieval |
| Post-Retrieval | Reranker (CrossEncoder/Cohere/DashScope) | Precision reranking of retrieved results |
| Generation | Multiple response modes, streaming | Customizable answer synthesis |

### Agent Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Pipeline** | Fixed flow: transform → retrieve → rerank → synthesize | Simple, predictable queries |
| **ReAct Agent** | Think → Act → Observe loop, dynamic tool selection | Complex multi-step reasoning |
| **PlanAgent** | Plan → Approve → Execute → Answer, with auto-replanning | Complex queries, production workflows |

### Agent Tools (10 tools)

**Retrieval Tools:**
- `vector_search` — Vector similarity search
- `keyword_search` — BM25 keyword search
- `multi_query_search` — Multi-variant query expansion
- `hyde_search` — Hypothetical document retrieval

**Optimization Tools:**
- `rewrite_query` — Query rewriting for vague inputs
- `rerank_results` — Post-retrieval reranking (requires Reranker config)
- `filter_results` — Filter by source file, score range

**Generation Tools:**
- `generate_answer` — Answer from collected context
- `retrieve_and_answer` — One-step retrieval + answer
- `table_lookup` — Table content lookup (requires markdown_base_path)

### Storage Backends

- Qdrant (vector + BM25 native support)
- PostgreSQL/pgvector
- Local file storage

### Chinese Support

- jieba tokenization for BM25
- Chinese text splitter
- Chinese prompt templates (4 modes: simple/default/professional/technical)

### Web Service

- **Backend**: FastAPI with PDF conversion, document splitting, import, and chat endpoints
- **Frontend**: Vue 3 SPA with tabbed interface for all operations
- **Chat Modes**: Pipeline/Agent/Plan mode selector in web UI

## Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=                    # Optional: custom API endpoint
OPENAI_EMBEDDING_API_KEY=sk-xxx     # Optional: separate embedding key
OPENAI_EMBEDDING_BASE_URL=          # Optional: custom embedding endpoint
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4-turbo

# Storage Configuration
PROFIRAG_STORAGE_TYPE=qdrant        # qdrant, local, or postgres

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=profirag

# Index Mode (controls storage mode)
PROFIRAG_INDEX_MODE=hybrid          # hybrid (dense + BM25) or vector (dense only)

# Retrieve Mode (controls query mode)
PROFIRAG_RETRIEVE_INDEX_MODE=hybrid # hybrid, sparse (BM25 only), or vector (dense only)

# Agent Configuration
PROFIRAG_AGENT_ENABLED=false        # Enable Agent mode
PROFIRAG_AGENT_MODE=react           # react, plan, or pipeline
PROFIRAG_AGENT_MAX_ITERATIONS=10
PROFIRAG_AGENT_MARKDOWN_BASE_PATH=  # Optional: for table_lookup tool

# Reranking Configuration
PROFIRAG_RERANK_ENABLED=true
PROFIRAG_RERANK_PROVIDER=local      # local, cohere, or dashscope
PROFIRAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Quick Start

### Interactive Q&A

```bash
# Pipeline mode (default)
python main.py

# ReAct Agent mode
python main.py --mode agent

# Plan Agent mode
python main.py --mode plan

# Single query
python main.py --query "What is GaussDB?" --mode agent
```

### Programmatic Usage

```python
from profirag.config import load_config
from profirag.pipeline import RAGPipeline

config = load_config()
pipeline = RAGPipeline(config)

# Pipeline mode
result = pipeline.query("Your question here", top_k=10)

# Agent mode
result = pipeline.query_with_agent("Complex multi-step question", mode="agent")

# Plan mode
result = pipeline.query_with_agent("Complex question", mode="plan", auto_approve=True)
```

### Document Ingestion

```bash
# Ingest from directory
uv run python scripts/ingest_documents.py --documents ./documents

# Ingest single file
uv run python scripts/ingest_documents.py --file ./documents/example.pdf
```

### Web Service

Start the web interface for interactive document processing and chat:

```bash
# Start backend (from project root)
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend dev server
cd web/frontend && npm install && npm run dev
```

Access at http://localhost:5173. Features:
- **PDF Convert**: Upload PDF → Markdown conversion with table/image extraction
- **Doc Splitter**: Preview document chunking with multiple splitter types
- **Doc Import**: Import documents to vector store with progress tracking
- **Chat**: Knowledge Q&A with three modes:
  - **直接问答 (Pipeline)**: Fast retrieval-based answers
  - **Agent**: Intelligent tool selection for complex queries
  - **Plan**: Structured execution with auto-approved plans

## Project Structure

```
ProfiRAG/
├── main.py                     # Interactive CLI entry point
├── src/profirag/
│   ├── config/settings.py      # Pydantic config (env vars, model configs)
│   ├── agent/                   # Agent system
│   │   ├── tools.py             # RAGTools — 10 agent tools
│   │   ├── react_agent.py       # RAGReActAgent + AgentFactory
│   │   └── plan_agent.py        # RAGPlanAgent (Plan → Execute → Replan)
│   ├── pipeline/
│   │   └── rag_pipeline.py      # RAGPipeline — orchestration layer
│   ├── ingestion/               # Document loaders, splitters (AST, markdown, Chinese)
│   ├── retrieval/               # HybridRetriever, QueryTransform, Reranker
│   ├── generation/              # ResponseSynthesizer, PromptTemplates
│   ├── embedding/               # Custom OpenAI embedding wrapper
│   ├── storage/                 # Storage abstraction (Qdrant, PG, Local)
│   └── evaluation/              # Retrieval, response, chunking, dataset eval
├── web/                         # Web service
│   ├── api/                     # FastAPI backend
│   │   ├── main.py              # API entry point
│   │   ├── routes/              # PDF, split, import, chat endpoints
│   │   ├── services.py          # Business logic wrappers
│   │   └── schemas.py           # Pydantic request/response models
│   └── frontend/                # Vue 3 frontend
│   │   ├── src/views/           # PdfConvert, DocSplitter, DocImport, Chat
│   │   ├── src/components/      # ModeSelector, shared components
│   │   └── src/api/             # Axios API client
│   │   └── src/App.vue          # Main layout with tab navigation
└── scripts/                     # Utility scripts (PDF conversion, ingestion)
```

## Key Architecture Decisions

### Pipeline Flow (Standard Mode)
```
User Query → PreRetrievalPipeline → HybridRetriever → Reranker → ResponseSynthesizer → Answer
```

### ReAct Agent Flow
```
User Query → Think → Tool Selection → Execute → Observe → Think ... → Final Answer
```

### PlanAgent Flow
```
User Query → PlanGenerator (LLM) → User Approval → PlanExecutor → Replan(on failure) → Finalize
```

### Tools Design
- All tools are `FunctionTool` instances from LlamaIndex
- `_last_retrieved_nodes` acts as shared context between tools
- Optimization tools (rerank, filter) require prior retrieval
- `rewrite_query` has LLM fallback when QueryRewriter not configured

## Development

```bash
# Run tests
uv run pytest tests -v

# Format code
uv run ruff format src scripts tests

# Lint code
uv run ruff check src scripts tests --fix
```

## License

MIT
