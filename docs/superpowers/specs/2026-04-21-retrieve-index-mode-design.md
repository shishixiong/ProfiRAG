# Retrieve Index Mode Design

**Date:** 2026-04-21
**Status:** Draft

## Summary

Add a new environment variable `profirag_retrieve_index_mode` to control retrieval/querying behavior, supporting three modes: `hybrid`, `sparse` (BM25), and `vector`.

## Motivation

The existing `profirag_index_mode` controls **indexing/ingestion** (how data is stored). Users need the ability to control **retrieval/querying** independently, allowing:
- A/B testing different retrieval strategies without re-indexing
- Choosing the best retrieval method for specific query types
- Supporting pure keyword search for exact term matching scenarios

## Design Principle

**Independence:** Retrieval mode is independent of indexing mode.
- Data indexed with `hybrid` (dense + sparse vectors) can be retrieved with any mode
- Retrieval mode determines which vectors/embeddings are used at query time
- This enables flexible experimentation without re-indexing costs

## Configuration

### Environment Variable

```bash
PROFIRAG_RETRIEVE_INDEX_MODE=hybrid  # Options: hybrid, sparse, vector
```

### Settings.py Changes

**EnvSettings class:**
```python
profirag_retrieve_index_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"
```

**RetrievalConfig class:**
```python
retrieve_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"
```

**RAGConfig.from_env:**
Pass retrieve_mode to RetrievalConfig.

### .env.example Update

```bash
# ==================== Retrieval Configuration ====================
PROFIRAG_TOP_K=10
PROFIRAG_ALPHA=0.5
PROFIRAG_RETRIEVE_INDEX_MODE=hybrid  # hybrid (dense+BM25), sparse (BM25 only), vector (dense only)
```

## Retrieval Modes

| Mode | Description | Dense Embedding | Sparse Embedding | Best For |
|------|-------------|-----------------|------------------|----------|
| `hybrid` | RRF fusion of dense + sparse | Yes | Yes | General use, best recall |
| `sparse` | Pure BM25/keyword search | No | Yes | Exact term matching, keyword-heavy queries |
| `vector` | Pure semantic/dense search | Yes | No | Semantic similarity, conceptual queries |

## Key Discovery: LlamaIndex Native Support

LlamaIndex **already provides native support** for all three retrieval modes through `VectorStoreQueryMode`:

```python
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Pure vector search (DEFAULT)
VectorStoreQueryMode.DEFAULT

# Pure sparse/BM25 search
VectorStoreQueryMode.SPARSE

# Hybrid search (dense + sparse fusion)
VectorStoreQueryMode.HYBRID
```

**QdrantVectorStore.query()** already handles all three modes correctly:
- `HYBRID`: Queries both dense and sparse vectors, applies fusion
- `SPARSE`: Queries only sparse vectors via `using=sparse_vector_name`
- `DEFAULT`: Queries only dense vectors

This means **no custom implementation needed** in QdrantStore - we just need to pass the correct `VectorStoreQueryMode` when creating retrievers.

## Implementation

### 1. HybridRetriever Changes (Simplified)

**Add vector_store_query_mode parameter and use native LlamaIndex retriever:**

```python
from llama_index.core.vector_stores.types import VectorStoreQueryMode

def __init__(
    self,
    vector_index: VectorStoreIndex,
    alpha: float = 0.5,
    rrf_k: int = 60,
    vector_store: Optional[Any] = None,
    retrieve_mode: str = "hybrid",  # New parameter
    **kwargs
):
    self.vector_index = vector_index
    self.alpha = alpha
    self.rrf_k = rrf_k
    self.vector_store = vector_store
    self.retrieve_mode = retrieve_mode

    # Map retrieve_mode to VectorStoreQueryMode
    self._query_mode = self._map_retrieve_mode(retrieve_mode)

    # Create retriever with native LlamaIndex support
    self._vector_retriever = vector_index.as_retriever(
        vector_store_query_mode=self._query_mode,
        alpha=alpha,
        **kwargs
    )

def _map_retrieve_mode(self, mode: str) -> VectorStoreQueryMode:
    """Map retrieve_mode string to VectorStoreQueryMode enum."""
    mode_map = {
        "hybrid": VectorStoreQueryMode.HYBRID,
        "sparse": VectorStoreQueryMode.SPARSE,
        "vector": VectorStoreQueryMode.DEFAULT,
    }
    return mode_map.get(mode, VectorStoreQueryMode.HYBRID)

def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[NodeWithScore]:
    """Perform retrieval using configured mode."""
    # Native LlamaIndex retriever handles everything
    return self._vector_retriever.retrieve(query)
```

### 2. RAGPipeline Changes

**Pass retrieve_mode to HybridRetriever:**

```python
self._hybrid_retriever = HybridRetriever(
    vector_index=self._index,
    alpha=config.retrieval.alpha,
    vector_store=self._vector_store,
    retrieve_mode=config.retrieval.retrieve_mode,  # New parameter
)
```

### 3. QdrantStore Changes (Minimal)

**No changes needed to query method** - QdrantVectorStore already handles all modes correctly via `VectorStoreQuery.mode`.

The only potential change is ensuring `enable_hybrid=True` is set during initialization when `index_mode="hybrid"` (already done in current code).

### 4. BaseVectorStore Interface (Optional)

No interface changes required - the mode is handled internally by the LlamaIndex retriever.

## Edge Cases & Error Handling

1. **Sparse retrieval on vector-only indexed data:**
   - QdrantVectorStore will raise: "Hybrid search is not enabled. Please build the query with `enable_hybrid=True`"
   - This is correct behavior - sparse vectors don't exist

2. **Invalid retrieve_mode:**
   - Error: "Invalid retrieve_mode: {mode}. Must be hybrid, sparse, or vector"
   - Validation in `_map_retrieve_mode` method

3. **Dense embedding for sparse-only mode:**
   - LlamaIndex handles this internally - no dense embedding generated for SPARSE mode

## Testing

### Unit Tests

1. Test configuration loading of retrieve_mode
2. Test `_map_retrieve_mode` mapping function
3. Test HybridRetriever initialization with each mode
4. Test that correct VectorStoreQueryMode is passed to retriever

### Integration Tests

1. Index with hybrid mode, retrieve with each mode (hybrid, sparse, vector)
2. Verify retrieval results differ by mode
3. Verify sparse mode works for keyword-heavy queries (exact term matching)
4. Verify vector mode works for semantic queries
5. Verify error when sparse retrieval on vector-only indexed data

## Files to Modify

1. `src/profirag/config/settings.py` - Add retrieve_mode config
2. `src/profirag/retrieval/hybrid.py` - Add retrieve_mode and use native LlamaIndex retriever
3. `src/profirag/pipeline/rag_pipeline.py` - Pass retrieve_mode to retriever
4. `.env.example` - Add documentation for new env var

## Files NOT Modified (Simplified Approach)

- `src/profirag/storage/base.py` - No interface changes needed
- `src/profirag/storage/qdrant_store.py` - No changes needed (native support)

## Out of Scope

- Changes to `profirag_index_mode` (indexing remains unchanged)
- Local/Postgres store implementations (only Qdrant supports native sparse)