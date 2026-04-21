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

## Implementation

### 1. QdrantStore Changes

**Modify query method to accept retrieve_mode:**

```python
def query(
    self,
    query: QueryBundle,
    similarity_top_k: int = 10,
    retrieve_mode: str = "hybrid",
    **kwargs
) -> List[NodeWithScore]:
```

**Implementation per mode:**

- **`hybrid`**: Use existing hybrid search (delegates to QdrantVectorStore with enable_hybrid=True)
- **`vector`**: Query using only dense vectors (QueryBundle with embedding, no sparse)
- **`sparse`**: Query using only sparse vectors via Qdrant's sparse query API

**Sparse-only query implementation:**

For sparse-only retrieval, we need to generate sparse embeddings manually since QdrantVectorStore handles this internally for hybrid mode. We'll use fastembed directly:

```python
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

# Initialize sparse embedding model (same as used by QdrantVectorStore)
self._sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

def query_sparse(self, query: QueryBundle, similarity_top_k: int = 10) -> List[NodeWithScore]:
    # Generate sparse embedding for query
    sparse_embedding = list(self._sparse_model.embed(query.query_str))[0]

    # Query using sparse vector
    results = self._client.query_points(
        collection_name=self.collection_name,
        query=sparse_embedding.as_object(),
        using=self.SPARSE_VECTOR_NAME,
        limit=similarity_top_k,
        with_payload=True,
    )

    # Convert to NodeWithScore
    nodes = []
    for point in results.points:
        node = TextNode(
            id_=str(point.id),
            text=point.payload.get("text", ""),
            metadata=point.payload.get("metadata", {}) or {},
        )
        nodes.append(NodeWithScore(node=node, score=point.score))
    return nodes
```

### 2. HybridRetriever Changes

**Add retrieve_mode parameter:**

```python
def __init__(
    self,
    vector_index: VectorStoreIndex,
    alpha: float = 0.5,
    rrf_k: int = 60,
    vector_store: Optional[Any] = None,
    retrieve_mode: str = "hybrid",
    **kwargs
):
    self.retrieve_mode = retrieve_mode
```

**Modify retrieve method:**
```python
def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[NodeWithScore]:
    # Pass retrieve_mode to vector_store.query()
    query_bundle = QueryBundle(query_str=query)

    if self.retrieve_mode in ("hybrid", "vector"):
        # Generate dense embedding
        query_bundle.embedding = self.vector_index._embed_model.get_text_embedding(query)

    return self.vector_store.query(
        query_bundle,
        similarity_top_k=top_k,
        retrieve_mode=self.retrieve_mode,
        **kwargs
    )
```

### 3. RAGPipeline Changes

**Pass retrieve_mode to HybridRetriever:**

```python
self._hybrid_retriever = HybridRetriever(
    vector_index=self._index,
    alpha=config.retrieval.alpha,
    vector_store=self._vector_store,
    retrieve_mode=config.retrieval.retrieve_mode,
)
```

### 4. BaseVectorStore Interface

**Add retrieve_mode to query signature:**

```python
def query(
    self,
    query: QueryBundle,
    similarity_top_k: int = 10,
    retrieve_mode: str = "hybrid",
    **kwargs
) -> List[NodeWithScore]:
```

## Edge Cases & Error Handling

1. **Sparse retrieval on vector-only indexed data:**
   - Error: "Sparse retrieval requires data indexed with hybrid mode"
   - Check: Verify sparse vectors exist before sparse query

2. **Invalid retrieve_mode:**
   - Error: "Invalid retrieve_mode: {mode}. Must be hybrid, sparse, or vector"
   - Validation in settings with Literal type

3. **Missing embedding for vector/sparse modes:**
   - Dense embedding generated by embed_model for vector/hybrid
   - Sparse embedding generated by fastembed model for sparse/hybrid

## Testing

### Unit Tests

1. Test configuration loading of retrieve_mode
2. Test HybridRetriever with each mode
3. Test QdrantStore.query with each mode
4. Test error handling for invalid modes

### Integration Tests

1. Index with hybrid mode, retrieve with each mode (hybrid, sparse, vector)
2. Verify retrieval results differ by mode
3. Verify sparse mode works for keyword-heavy queries
4. Verify vector mode works for semantic queries

## Files to Modify

1. `src/profirag/config/settings.py` - Add retrieve_mode config
2. `src/profirag/retrieval/hybrid.py` - Add retrieve_mode to HybridRetriever
3. `src/profirag/storage/base.py` - Update query interface
4. `src/profirag/storage/qdrant_store.py` - Implement mode-specific queries
5. `src/profirag/pipeline/rag_pipeline.py` - Pass retrieve_mode to retriever
6. `.env.example` - Add documentation for new env var

## Out of Scope

- Changes to `profirag_index_mode` (indexing remains unchanged)
- BM25Index class modifications (using native Qdrant BM25 instead)
- Local/Postgres store implementations (only Qdrant supports native sparse)