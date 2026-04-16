# ProfiRAG

Advanced RAG (Retrieval-Augmented Generation) framework built with LlamaIndex.

## Features

- **Pluggable Vector Storage**: Abstract storage layer supporting multiple backends
  - Qdrant
  - PostgreSQL/pgvector
  - Local file storage

- **Advanced RAG Pipeline**:
  - Pre-Retrieval: HyDE, query rewriting, multi-query generation
  - Hybrid Retrieval: Vector + BM25 with RRF fusion
  - Re-ranking: Cross-encoder models
  - Generation: Multiple response synthesis modes

- **Chinese Support**:
  - jieba tokenization for BM25
  - Chinese text splitter
  - Chinese prompt templates

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
```

## Quick Start

```python
from profirag import RAGPipeline
from profirag.config import load_config

# Load configuration from .env
config = load_config()
pipeline = RAGPipeline(config)

# Ingest documents
from profirag.ingestion import DocumentLoader
loader = DocumentLoader()
documents = loader.load_directory("./documents")
pipeline.ingest_documents(documents)

# Query
result = pipeline.query("What is GaussDB?")
print(result["response"])
```

## Usage

## convert pdf to markdown
```bash
python scripts/pdf_to_markdown.py --write-images --pages "2914-3393" --exclude-header-footer --extract-tables
```

### Ingest Documents

```bash
# Ingest from directory
uv run python scripts/ingest_documents.py --documents ./documents

# Ingest single file
uv run python scripts/ingest_documents.py --file ./documents/example.pdf
```

### Run Queries

```python
from profirag import RAGPipeline

pipeline = RAGPipeline.from_env()

# Standard query
result = pipeline.query("Your question here", top_k=10)

# Streaming query
for chunk in pipeline.query_stream("Your question here"):
    print(chunk, end="", flush=True)
```

## Project Structure

```
ProfiRAG/
├── src/profirag/
│   ├── config/          # Configuration management
│   ├── storage/         # Vector store abstraction
│   ├── ingestion/       # Document loaders and splitters
│   ├── retrieval/       # Query transform, hybrid retrieval, reranking
│   ├── generation/      # Response synthesis
│   └── pipeline/        # Main RAG pipeline
├── scripts/             # Utility scripts
└── tests/               # Test suite
```

## Development

```bash
# Run tests
uv run pytest tests -v

# Format code
uv run ruff format src scripts tests

# Lint code
uv run ruff check src scripts tests --fix

# Type check
uv run mypy src --ignore-missing-imports
```

## License

MIT