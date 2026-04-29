# FastEmbed Dense Embedding Support Design

**Date:** 2026-04-29
**Status:** Approved
**Scope:** Add local embedding support via FastEmbed as alternative to OpenAI API

## Summary

Add FastEmbed support for local dense vectorization, enabling zero-cost, offline embedding generation. Users can switch between OpenAI and FastEmbed via a simple configuration flag.

## Requirements

- Support FastEmbed dense embeddings as alternative to OpenAI API
- Configurable model selection (BGE, MTEB, multilingual models)
- Provider enum: `embedding.provider: "openai" | "fastembed"`
- Fail explicitly on model loading errors (no silent fallback)
- Work with all vector store backends (Qdrant, Local, Postgres)

## Architecture

### New Class: FastEmbedEmbedding

Location: `src/profirag/embedding/fastembed_embedding.py`

```
FastEmbedEmbedding(BaseEmbedding):
  - model: str                 # Model name (e.g., "BAAI/bge-small-en-v1.5")
  - cache_dir: Optional[str]   # Local model cache directory
  - dimension: int             # Embedding dimension (model-specific)
  - _model: TextEmbedding      # Lazy-loaded FastEmbed instance
```

Inherits from `llama_index.core.base.embeddings.base.BaseEmbedding`, wrapping `fastembed.TextEmbedding`.

### Provider Enum Expansion

Update `EmbeddingConfig.provider` type:
```python
provider: Literal["openai", "fastembed"] = "openai"
```

New environment variables:
- `PROFIRAG_EMBEDDING_PROVIDER`: "openai" or "fastembed"
- `PROFIRAG_EMBEDDING_MODEL`: FastEmbed model name (default: "BAAI/bge-small-en-v1.5")
- `PROFIRAG_EMBEDDING_CACHE_DIR`: Optional cache directory
- `PROFIRAG_EMBEDDING_DIMENSION`: Optional dimension override (auto-detected for FastEmbed if not set)

### Factory Pattern

`RAGPipeline._create_embed_model()` selects embedding class based on provider:
- `"fastembed"` → `FastEmbedEmbedding`
- `"openai"` → `CustomOpenAIEmbedding`

### Async Handling

FastEmbed is synchronous only. Async methods wrap sync calls via `asyncio.to_thread()`.

## Components

### File: src/profirag/embedding/fastembed_embedding.py

```
FastEmbedEmbedding(BaseEmbedding):
  Properties:
    model: str
    cache_dir: Optional[str]
    dimension: int
    _model: Optional[TextEmbedding]

  Methods:
    _load_model() -> TextEmbedding        # Lazy initialization
    _get_query_embedding(query) -> List[float]
    _aget_query_embedding(query) -> List[float]   # asyncio.to_thread wrapper
    _get_text_embedding(text) -> List[float]
    _aget_text_embedding(text) -> List[float]
    _get_text_embeddings(texts) -> List[List[float]]
    _aget_text_embeddings(texts) -> List[List[float]]
```

### File: src/profirag/embedding/__init__.py

Update exports:
```python
from .custom_embedding import CustomOpenAIEmbedding
from .fastembed_embedding import FastEmbedEmbedding

__all__ = ["CustomOpenAIEmbedding", "FastEmbedEmbedding"]
```

### File: src/profirag/config/settings.py

EnvSettings additions:
```python
profirag_embedding_provider: Literal["openai", "fastembed"] = "openai"
profirag_embedding_model: str = "BAAI/bge-small-en-v1.5"
profirag_embedding_dimension: Optional[int] = None  # Auto-detected for FastEmbed
profirag_embedding_cache_dir: Optional[str] = None
```

EmbeddingConfig update:
```python
class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "fastembed"] = "openai"
    model: str = "text-embedding-3-small"  # Default for OpenAI
    dimension: int = 1536
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cache_dir: Optional[str] = None  # For FastEmbed

    def __post_init__(self):
        # Auto-detect dimension for FastEmbed if not set
        if self.provider == "fastembed" and self.dimension is None:
            self.dimension = FASTEMBED_MODEL_DIMENSIONS.get(self.model, 768)
```

FASTEMBED_MODEL_DIMENSIONS mapping:
```python
FASTEMBED_MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
```

RAGConfig.from_env() update:
```python
# Determine dimension based on provider
if env_settings.profirag_embedding_provider == "fastembed":
    model = env_settings.profirag_embedding_model
    dimension = env_settings.profirag_embedding_dimension or FASTEMBED_MODEL_DIMENSIONS.get(model, 768)
else:
    model = env_settings.openai_embedding_model
    dimension = env_settings.openai_embedding_dimension

embedding=EmbeddingConfig(
    provider=env_settings.profirag_embedding_provider,
    model=model,
    dimension=dimension,
    api_key=env_settings.openai_embedding_api_key if env_settings.profirag_embedding_provider == "openai" else None,
    base_url=env_settings.openai_embedding_base_url if env_settings.profirag_embedding_provider == "openai" else None,
    cache_dir=env_settings.profirag_embedding_cache_dir,
)
```

### File: src/profirag/pipeline/rag_pipeline.py

_update _create_embed_model():
```python
def _create_embed_model(self) -> BaseEmbedding:
    if self.config.embedding.provider == "fastembed":
        return FastEmbedEmbedding(
            model=self.config.embedding.model,
            dimension=self.config.embedding.dimension,
            cache_dir=self.config.embedding.cache_dir,
        )
    else:  # openai
        return CustomOpenAIEmbedding(...)
```

### File: pyproject.toml

Add dependency:
```toml
dependencies = [
    ...
    "fastembed>=0.2.0",
]
```

### File: .env.example

Add FastEmbed configuration examples:
```bash
# Embedding Provider (openai or fastembed)
PROFIRAG_EMBEDDING_PROVIDER=openai

# FastEmbed Configuration (only used when provider=fastembed)
# Model name - dimension auto-detected for known models
PROFIRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# Optional: override dimension (defaults to auto-detected based on model)
PROFIRAG_EMBEDDING_DIMENSION=
# Optional: custom cache directory for model files
PROFIRAG_EMBEDDING_CACHE_DIR=
```

## Data Flow

### Configuration Flow

```
.env → EnvSettings → EmbeddingConfig → RAGConfig → RAGPipeline
```

1. User sets `PROFIRAG_EMBEDDING_PROVIDER=fastembed`
2. `EnvSettings` reads provider value
3. `RAGConfig.from_env()` creates `EmbeddingConfig(provider="fastembed")`
4. `RAGPipeline.__init__()` calls `_create_embed_model()`
5. Factory returns `FastEmbedEmbedding` instance

### Embedding Flow (Ingestion)

```
Documents → TextSplitter → TextNodes
                          ↓
         VectorStoreIndex.insert_nodes()
                          ↓
         FastEmbedEmbedding._get_text_embeddings()
                          ↓
         TextEmbedding.embed() → vectors
                          ↓
         VectorStore stores vectors + payloads
```

### Embedding Flow (Query)

```
Query string → QueryBundle
              ↓
   FastEmbedEmbedding._get_query_embedding()
              ↓
   TextEmbedding.embed() → query_vector
              ↓
   VectorStore.query(query_vector) → NodeWithScore[]
```

### Lazy Model Loading

```
First request → _load_model()
                ↓
        Download model (if not cached)
                ↓
        Initialize TextEmbedding
                ↓
        Store in _model instance variable
                ↓
        Return instance

Subsequent requests → return cached _model
```

## Model Dimension Mapping

Built-in mapping for common FastEmbed models:

| Model | Dimension |
|-------|-----------|
| BAAI/bge-small-en-v1.5 | 384 |
| BAAI/bge-base-en-v1.5 | 768 |
| BAAI/bge-large-en-v1.5 | 1024 |
| intfloat/multilingual-e5-large | 1024 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 |

For unknown models, default to 768 and log warning.

## Error Handling

### Model Loading Errors

```python
try:
    from fastembed import TextEmbedding
    self._model = TextEmbedding(model_name=self.model, cache_dir=self.cache_dir)
except ImportError:
    raise ImportError("fastembed package not installed. Run: uv add fastembed")
except ValueError as e:
    raise ValueError(f"Invalid FastEmbed model '{self.model}'. Available models: {TextEmbedding.list_supported_models()}")
except Exception as e:
    raise RuntimeError(f"Failed to load FastEmbed model '{self.model}': {e}")
```

### Embedding Errors

```python
try:
    embeddings = list(self._model.embed([text]))
    return embeddings[0]
except Exception as e:
    raise RuntimeError(f"FastEmbed embedding failed: {e}")
```

### Empty Text Handling

```python
if not text or not text.strip():
    return [0.0] * self.dimension
```

### Configuration Validation

- Warn if user-provided dimension doesn't match model's expected dimension
- Validate model name at initialization (fail early)

## Testing

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| test_fastembed_init | tests/embedding/test_fastembed.py | Valid model initialization |
| test_fastembed_invalid_model | tests/embedding/test_fastembed.py | Error on invalid model |
| test_fastembed_embedding_single | tests/embedding/test_fastembed.py | Single text embedding |
| test_fastembed_embedding_batch | tests/embedding/test_fastembed.py | Batch embedding |
| test_fastembed_async | tests/embedding/test_fastembed.py | Async method wrappers |
| test_fastembed_empty_text | tests/embedding/test_fastembed.py | Empty text handling |
| test_config_provider_switch | tests/config/test_settings.py | Provider enum validation |

### Integration Tests

| Test | File | Description |
|------|------|-------------|
| test_pipeline_fastembed_qdrant | tests/integration/test_fastembed.py | Full pipeline with Qdrant |
| test_pipeline_fastembed_ingest_query | tests/integration/test_fastembed.py | Ingest + query cycle |
| test_pipeline_provider_selection | tests/integration/test_fastembed.py | Factory selection |

### Test Fixtures

```python
@pytest.fixture
def fastembed_config():
    return EmbeddingConfig(
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        dimension=384,
    )

@pytest.fixture
def fastembed_embedding():
    return FastEmbedEmbedding(
        model="BAAI/bge-small-en-v1.5",
        dimension=384,
    )
```

## Implementation Notes

1. FastEmbed caches models locally by default in `~/.cache/fastembed`
2. First run downloads model (~100MB-500MB depending on model)
3. Subsequent runs use cached model (fast startup)
4. Batch embedding is more efficient than single calls
5. FastEmbed supports CUDA acceleration if GPU available

## Out of Scope

- FastEmbed sparse embeddings (already via Qdrant)
- Hybrid mode (FastEmbed + OpenAI fallback)
- Custom model fine-tuning
- GPU configuration options

## Dependencies

- `fastembed>=0.2.0` - Core embedding library
- Existing: `llama-index-core` (BaseEmbedding)

## Migration Path

Existing users: No change required. Default remains OpenAI.

New users wanting local embeddings:
1. Set `PROFIRAG_EMBEDDING_PROVIDER=fastembed`
2. Set `PROFIRAG_EMBEDDING_MODEL=<desired_model>` (dimension auto-detected for known models)
3. Optionally set `PROFIRAG_EMBEDDING_DIMENSION` to override auto-detected value

No breaking changes.