# Reranker API Support Design

## Overview

Add support for OpenAPI-compatible reranker services alongside existing local CrossEncoder model. Support both Cohere-compatible API format and Alibaba Cloud DashScope format.

## Configuration

### New Environment Variables

```bash
PROFIRAG_RERANK_PROVIDER=local|cohere|dashscope  # Default: local
PROFIRAG_RERANK_API_KEY=                          # Required for API modes
PROFIRAG_RERANK_BASE_URL=                         # Required for API modes
PROFIRAG_RERANK_MODEL=                            # Model name (meaning varies by provider)
PROFIRAG_RERANK_TOP_N=5                           # Number of results to return
PROFIRAG_RERANK_ENABLED=true                      # Enable/disable reranking
PROFIRAG_RERANK_TIMEOUT=30                        # API timeout in seconds (optional)
```

### Configuration Model Updates

`RerankingConfig` in `settings.py`:

```python
class RerankingConfig(BaseModel):
    enabled: bool = True
    provider: Literal["local", "cohere", "dashscope"] = "local"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
```

### .env.example Updates

```bash
# ==================== Reranking Configuration ====================
PROFIRAG_RERANK_ENABLED=true
PROFIRAG_RERANK_PROVIDER=local  # local (CrossEncoder), cohere, dashscope
PROFIRAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# For API-based reranking (required when provider != local)
PROFIRAG_RERANK_API_KEY=
PROFIRAG_RERANK_BASE_URL=
PROFIRAG_RERANK_TOP_N=5
PROFIRAG_RERANK_TIMEOUT=30
```

## Architecture

### Code Structure

```
src/profirag/retrieval/reranker.py
├── BaseReranker (ABC)          # Abstract base class
├── CrossEncoderReranker        # Local mode (existing, minimal changes)
├── CohereReranker              # Cohere API format
├── DashScopeReranker           # DashScope API format
└── Reranker                    # Unified entry point (factory pattern)
```

### Class Design

#### BaseReranker

```python
class BaseReranker(ABC):
    top_n: int

    @abstractmethod
    def rerank(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        pass
```

#### CohereReranker

- Endpoint: `{base_url}/rerank`
- Request body:
  ```json
  {
    "model": "rerank-v1",
    "query": "...",
    "documents": ["doc1", "doc2"],
    "top_n": 5
  }
  ```
- Response:
  ```json
  {
    "results": [{"index": 0, "relevance_score": 0.9}, ...]
  }
  ```

#### DashScopeReranker

- Endpoint: `{base_url}/api/v1/services/rerank`
- Request body:
  ```json
  {
    "model": "rerank-v1",
    "input": {
      "query": "...",
      "documents": ["doc1", "doc2"],
      "top_n": 5
    }
  }
  ```
- Response:
  ```json
  {
    "output": {"results": [{"index": 0, "relevance_score": 0.9}, ...]},
    "request_id": "..."
  }
  ```

#### Reranker (Factory)

```python
class Reranker:
    def __init__(self, config: RerankingConfig):
        if config.provider == "local":
            self._impl = CrossEncoderReranker(...)
        elif config.provider == "cohere":
            self._impl = CohereReranker(...)
        elif config.provider == "dashscope":
            self._impl = DashScopeReranker(...)
```

## API Format Details

### Cohere-Compatible Format

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model identifier |
| query | string | Search query |
| documents | string[] | Documents to rerank |
| top_n | int | Number of results |

### DashScope Format

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model identifier (e.g., "rerank-v1") |
| input.query | string | Search query |
| input.documents | string[] | Documents to rerank |
| input.top_n | int | Number of results |

## Error Handling

1. **Missing API key for API modes**: Raise `ValueError` at initialization
2. **API call failure**: Raise `RuntimeError` with error details, log error
3. **Timeout**: Use configurable timeout (default 30s), raise on timeout
4. **Empty response**: Return empty list
5. **Invalid response format**: Raise `RuntimeError`

## Backward Compatibility

- Default `provider=local` preserves existing behavior
- Existing `model` config still works for local mode
- API modes require new config fields, no impact on existing users

## Testing

- Unit tests for each reranker class
- Mock API responses for API rerankers
- Integration test for Reranker factory
- Configuration loading tests