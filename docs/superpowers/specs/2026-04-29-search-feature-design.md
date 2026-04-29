# Search Feature Design Spec

**Date:** 2026-04-29
**Status:** Draft - Pending User Review

## Overview

Add a pure retrieval "Search" feature to the ProfiRAG web UI. Users input natural language queries and receive raw retrieved chunks without LLM-generated answers. The UI follows a split-pane layout similar to Qdrant's code-search demo: file tree on the left, chunk cards on the right.

## Requirements

### Functional Requirements
1. Sidebar navigation item for Search page
2. Natural language query input
3. Retrieval-only results (no LLM synthesis)
4. File tree showing documents with retrieved chunks
5. Chunk cards with heading, score, and text preview
6. Click-to-expand for full chunk content
7. Configurable result count and reranking

### Non-functional Requirements
- Response time: < 2 seconds for typical queries
- Support up to 100 retrieved chunks per query
- Mobile-friendly responsive layout (collapse file tree on small screens)

## Architecture

### Frontend Components

| Component | Path | Purpose |
|-----------|------|---------|
| `SearchView.vue` | `web/frontend/src/views/SearchView.vue` | Main search page with query input and results display |
| `searchApi` | `web/frontend/src/api/index.js` (add) | Frontend API client for search endpoint |
| Route `/search` | `web/frontend/src/main.js` (add) | Vue router configuration |

### Backend Components

| Component | Path | Purpose |
|-----------|------|---------|
| `SearchService` | `web/api/services.py` (add) | Retrieval-only service wrapping HybridRetriever |
| `search` router | `web/api/routes/search.py` (new) | FastAPI route for search endpoint |
| `SearchRequest` | `web/api/schemas.py` (add) | Pydantic request model |
| `SearchResponse` | `web/api/schemas.py` (add) | Pydantic response model |

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  SearchView │────▶│  searchApi  │────▶│ /search/query│────▶│SearchService│
│   (Vue)     │     │  (axios)    │     │  (FastAPI)   │     │  (Python) │
└─────────────┘     └─────────────┘     └──────────────┘     └───────────┘
                                                                │
                                                                ▼
                        ┌────────────────────────────────────────────┐
                        │                                            │
                        ▼                                            ▼
              ┌──────────────┐                           ┌──────────────┐
              │PreRetrieval  │                           │HybridRetriever│
              │(optional)    │                           │              │
              └──────────────┘                           └──────────────┘
                        │                                            │
                        └────────────────────────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │   Reranker   │
                                    │  (optional)  │
                                    └──────────────┘
                                              │
                                              ▼
                                    ┌──────────────┐
                                    │   Response   │
                                    └──────────────┘
```

## API Specification

### Endpoint

```
POST /api/search/query
```

### Request Schema

```python
class SearchRequest(BaseModel):
    """Request for pure retrieval search."""
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")
    rerank: bool = Field(True, description="Enable reranking")
    use_pre_retrieval: bool = Field(False, description="Enable query transformation")
    env_file: str = Field(".env", description="Config file path")
```

### Response Schema

```python
class SearchResultFile(BaseModel):
    """File summary in search results."""
    filename: str
    chunk_count: int

class SearchResultChunk(BaseModel):
    """Single retrieved chunk."""
    chunk_id: str
    heading: Optional[str] = Field(None, description="Section heading")
    score: float
    text_preview: str = Field(..., description="First 200 characters")
    full_text: str
    source_file: str
    header_path: Optional[str] = Field(None, description="Full header hierarchy")

class SearchResponse(BaseModel):
    """Response for search query."""
    query: str
    total_results: int
    files: List[SearchResultFile]
    chunks: List[SearchResultChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Example Request

```json
{
  "query": "how to configure the embedding model",
  "top_k": 20,
  "rerank": true
}
```

### Example Response

```json
{
  "query": "how to configure the embedding model",
  "total_results": 5,
  "files": [
    {"filename": "config.md", "chunk_count": 3},
    {"filename": "setup.md", "chunk_count": 2}
  ],
  "chunks": [
    {
      "chunk_id": "abc123",
      "heading": "# Configuration",
      "score": 0.89,
      "text_preview": "The embedding model can be configured through environment variables...",
      "full_text": "The embedding model can be configured through environment variables or a YAML config file...",
      "source_file": "config.md",
      "header_path": "# Configuration > ## Embedding"
    }
  ],
  "metadata": {
    "reranked": true,
    "retrieval_mode": "hybrid"
  }
}
```

## Frontend Implementation

### SearchView.vue Structure

```vue
<template>
  <div class="search-container">
    <!-- Query Input -->
    <div class="search-input-area">
      <input v-model="query" placeholder="Enter natural language query..." />
      <button @click="executeSearch">Search</button>
      <div class="search-config">
        <label>Results: <input type="number" v-model="topK" /></label>
        <label><input type="checkbox" v-model="rerank" /> Rerank</label>
      </div>
    </div>

    <!-- Results Split View -->
    <div class="search-results" v-if="results">
      <!-- Left: File Tree -->
      <aside class="file-tree">
        <div class="file-item" v-for="file in results.files"
             :class="{ active: selectedFile === file.filename }"
             @click="selectFile(file.filename)">
          📄 {{ file.filename }} ({{ file.chunk_count }})
        </div>
      </aside>

      <!-- Right: Chunk Cards -->
      <div class="chunk-list">
        <div class="chunk-card" v-for="chunk in filteredChunks"
             :class="{ expanded: expandedChunk === chunk.chunk_id }"
             @click="toggleExpand(chunk.chunk_id)">
          <div class="chunk-header">
            <span class="heading">{{ chunk.heading || 'No heading' }}</span>
            <span class="score">{{ chunk.score.toFixed(2) }}</span>
          </div>
          <div class="chunk-preview" v-if="expandedChunk !== chunk.chunk_id">
            {{ chunk.text_preview }}
          </div>
          <div class="chunk-full" v-else>
            {{ chunk.full_text }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
```

### UI Behavior

1. **Query Input**: Text input with Enter key support, configurable top_k and rerank options
2. **File Tree**: Click file name to filter chunks to only that file; "All files" option at top
3. **Chunk Cards**:
   - Default: collapsed, showing heading + score + preview (200 chars)
   - Expanded: shows full text with collapse button
   - Click anywhere on card to toggle expand/collapse
4. **Empty State**: Show message when no results or no query yet

### Styling

- Follow existing App.vue CSS variables for consistency
- File tree: 200px width, dark background (#f8fafc)
- Chunk cards: white background, border-radius, hover state
- Score badge: green background (#dcfce7), right-aligned
- Expanded state: blue border highlight (#3b82f6)

## Backend Implementation

### SearchService

```python
class SearchService:
    """Pure retrieval service without LLM synthesis."""

    @staticmethod
    def query(
        query_str: str,
        top_k: int = 20,
        rerank: bool = True,
        use_pre_retrieval: bool = False,
        env_file: str = ".env",
    ) -> Dict[str, Any]:
        """Execute retrieval-only query."""
        # Load config
        config_path = PROJECT_ROOT / env_file
        load_dotenv(str(config_path), override=True)
        config = load_config()

        # Initialize pipeline components
        pipeline = RAGPipeline(config)

        # Pre-retrieval transformation (optional)
        if use_pre_retrieval:
            query_bundles = pipeline._pre_retrieval.transform(query_str)
        else:
            query_bundles = [QueryBundle(query_str=query_str)]

        # Retrieve from all query variants
        all_nodes = []
        for qb in query_bundles:
            nodes = pipeline._hybrid_retriever.retrieve(qb.query_str, top_k=top_k * 2)
            all_nodes.extend(nodes)

        # Deduplicate
        unique_nodes = pipeline._deduplicate_nodes(all_nodes)

        # Rerank (optional)
        if rerank:
            unique_nodes = pipeline._reranker.rerank(query_str, unique_nodes)

        # Format results
        return SearchService._format_results(query_str, unique_nodes[:top_k])

    @staticmethod
    def _format_results(query_str: str, nodes: List[NodeWithScore]) -> Dict[str, Any]:
        """Format nodes into SearchResponse structure."""
        # Group by file
        file_counts = {}
        chunks = []

        for node in nodes:
            source_file = node.node.metadata.get("source_file", "unknown")
            file_counts[source_file] = file_counts.get(source_file, 0) + 1

            text = node.node.text
            chunks.append({
                "chunk_id": node.node.node_id,
                "heading": node.node.metadata.get("current_heading"),
                "score": node.score,
                "text_preview": text[:200] if len(text) > 200 else text,
                "full_text": text,
                "source_file": source_file,
                "header_path": node.node.metadata.get("header_path"),
            })

        files = [{"filename": f, "chunk_count": c} for f, c in file_counts.items()]

        return {
            "query": query_str,
            "total_results": len(chunks),
            "files": files,
            "chunks": chunks,
            "metadata": {"reranked": True},
        }
```

### Router

```python
# web/api/routes/search.py

from fastapi import APIRouter, HTTPException
import schemas
import services

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("/query", response_model=schemas.SearchResponse)
async def query(request: schemas.SearchRequest):
    """Execute pure retrieval search."""
    try:
        result = services.SearchService.query(
            query_str=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            use_pre_retrieval=request.use_pre_retrieval,
            env_file=request.env_file,
        )
        return schemas.SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Integration Points

### Files to Modify

1. **web/frontend/src/App.vue** - Add Search nav item to sidebar
2. **web/frontend/src/main.js** - Add `/search` route
3. **web/frontend/src/api/index.js** - Add `searchApi` object
4. **web/api/routes/__init__.py** - Register search router
5. **web/api/services.py** - Add `SearchService` class
6. **web/api/schemas.py** - Add search-related schemas

### New Files

1. **web/frontend/src/views/SearchView.vue** - Search page component
2. **web/api/routes/search.py** - Search endpoint router

## Testing

### Manual Testing Checklist
- [ ] Search nav item appears in sidebar
- [ ] Search page loads correctly
- [ ] Query input accepts natural language
- [ ] Results appear after search
- [ ] File tree shows all source files
- [ ] Clicking file filters chunks
- [ ] Chunk cards show heading + score + preview
- [ ] Click expands chunk to show full text
- [ ] Configurable top_k works
- [ ] Rerank toggle works

### Edge Cases
- Empty query: Show validation error
- No results: Show "No results found" message
- Very long chunks: Preview truncates at 200 chars
- Multiple query variants (pre-retrieval): Deduplicate correctly

## Out of Scope

The following are explicitly NOT included in this feature:
- LLM-generated summaries or answers (use Chat for that)
- Conversation/session history
- Saving search queries
- Filtering by metadata (e.g., has_table, has_code flags)
- Pagination (showing all results in one page)
- Export/download results

## Success Criteria

1. Users can search with natural language and get relevant chunks
2. Response time under 2 seconds for typical queries
3. File tree navigation works correctly
4. Chunk expand/collapse is intuitive
5. UI consistent with existing ProfiRAG design