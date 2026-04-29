# Search Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure retrieval search feature to ProfiRAG web UI with split-pane layout (file tree left, chunk cards right).

**Architecture:** New backend endpoint `/api/search/query` returns raw chunks without LLM synthesis. Frontend SearchView.vue displays results in expandable cards with file filtering.

**Tech Stack:** FastAPI (backend), Vue 3 + Vue Router (frontend), existing RAGPipeline components (HybridRetriever, Reranker)

---

## File Structure

### Backend (web/api/)
- **schemas.py** (modify) - Add SearchRequest, SearchResponse, SearchResultFile, SearchResultChunk
- **services.py** (modify) - Add SearchService class
- **routes/search.py** (create) - New FastAPI router for search endpoint
- **routes/__init__.py** (modify) - Register search router

### Frontend (web/frontend/src/)
- **api/index.js** (modify) - Add searchApi object
- **views/SearchView.vue** (create) - New search page component
- **main.js** (modify) - Add /search route
- **App.vue** (modify) - Add Search nav item to sidebar

---

## Task 1: Add Search Schemas

**Files:**
- Modify: `web/api/schemas.py`

- [ ] **Step 1: Add SearchRequest schema at end of file**

```python
# Search Models
class SearchRequest(BaseModel):
    """Request for pure retrieval search."""
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")
    rerank: bool = Field(True, description="Enable reranking")
    use_pre_retrieval: bool = Field(False, description="Enable query transformation")
    env_file: str = Field(".env", description="Config file path")
```

- [ ] **Step 2: Add SearchResultFile schema**

```python
class SearchResultFile(BaseModel):
    """File summary in search results."""
    filename: str
    chunk_count: int
```

- [ ] **Step 3: Add SearchResultChunk schema**

```python
class SearchResultChunk(BaseModel):
    """Single retrieved chunk."""
    chunk_id: str
    heading: Optional[str] = Field(None, description="Section heading")
    score: float
    text_preview: str = Field(..., description="First 200 characters")
    full_text: str
    source_file: str
    header_path: Optional[str] = Field(None, description="Full header hierarchy")
```

- [ ] **Step 4: Add SearchResponse schema**

```python
class SearchResponse(BaseModel):
    """Response for search query."""
    query: str
    total_results: int
    files: List[SearchResultFile]
    chunks: List[SearchResultChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Commit schemas**

```bash
git add web/api/schemas.py
git commit -m "feat(api): add search request/response schemas"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 2: Add SearchService

**Files:**
- Modify: `web/api/services.py`

- [ ] **Step 1: Add QueryBundle import at top of file**

Add to existing imports (around line 5-10):
```python
from llama_index.core import QueryBundle
```

- [ ] **Step 2: Add SearchService class at end of services.py**

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
        """Execute retrieval-only query.

        Args:
            query_str: Natural language query
            top_k: Number of results to return
            rerank: Enable reranking
            use_pre_retrieval: Enable query transformation (HyDE, rewrite)
            env_file: Config file path

        Returns:
            Dictionary with query, total_results, files, chunks, metadata
        """
        # Load config
        config_path = PROJECT_ROOT / env_file
        load_dotenv(str(config_path), override=True)
        config = load_config()

        # Initialize pipeline (get retrieval components)
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
        return SearchService._format_results(query_str, unique_nodes[:top_k], rerank)

    @staticmethod
    def _format_results(
        query_str: str,
        nodes: List[NodeWithScore],
        reranked: bool
    ) -> Dict[str, Any]:
        """Format nodes into SearchResponse structure.

        Args:
            query_str: Original query string
            nodes: List of NodeWithScore objects
            reranked: Whether reranking was applied

        Returns:
            Dictionary matching SearchResponse schema
        """
        # Group by file
        file_counts: Dict[str, int] = {}
        chunks: List[Dict[str, Any]] = []

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
            "metadata": {"reranked": reranked},
        }
```

- [ ] **Step 3: Commit SearchService**

```bash
git add web/api/services.py
git commit -m "feat(api): add SearchService for pure retrieval"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 3: Create Search Router

**Files:**
- Create: `web/api/routes/search.py`

- [ ] **Step 1: Create search.py router file**

```python
"""Search endpoints for pure retrieval queries."""

from fastapi import APIRouter, HTTPException

import schemas
import services

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/query", response_model=schemas.SearchResponse, summary="Execute search query")
async def query(request: schemas.SearchRequest):
    """Execute pure retrieval search and return raw chunks.

    Unlike Chat, this endpoint returns only retrieved chunks without
    LLM-generated answers. Useful for browsing and exploring documents.

    Args:
        request: SearchRequest with query, top_k, rerank options

    Returns:
        SearchResponse with files summary and chunk details
    """
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

- [ ] **Step 2: Commit search router**

```bash
git add web/api/routes/search.py
git commit -m "feat(api): add search router with /query endpoint"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 4: Register Search Router

**Files:**
- Modify: `web/api/routes/__init__.py`

- [ ] **Step 1: Read current routes/__init__.py**

Run: Check the current content to see how routers are registered.

- [ ] **Step 2: Add search router import and registration**

Add search router to imports and router list:
```python
from .search import router as search_router

# In router registration section, add:
api_router.include_router(search_router)
```

- [ ] **Step 3: Commit router registration**

```bash
git add web/api/routes/__init__.py
git commit -m "feat(api): register search router in API"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 5: Add searchApi to Frontend

**Files:**
- Modify: `web/frontend/src/api/index.js`

- [ ] **Step 1: Add searchApi object after chatApi**

Add at end of file (after chatApi definition):
```javascript
// Search endpoints
export const searchApi = {
  query: async (query, topK = 20, rerank = true, usePreRetrieval = false) => {
    return api.post('/search/query', {
      query,
      top_k: topK,
      rerank,
      use_pre_retrieval: usePreRetrieval,
    })
  },
}
```

- [ ] **Step 2: Commit frontend API**

```bash
git add web/frontend/src/api/index.js
git commit -m "feat(frontend): add searchApi client"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 6: Create SearchView Component

**Files:**
- Create: `web/frontend/src/views/SearchView.vue`

- [ ] **Step 1: Create SearchView.vue with template**

```vue
<template>
  <div class="search-container">
    <!-- Header -->
    <h2 class="search-title">文档检索</h2>

    <!-- Query Input Area -->
    <div class="search-input-area">
      <input
        v-model="query"
        type="text"
        placeholder="输入自然语言查询..."
        @keyup.enter="executeSearch"
        :disabled="loading"
      />
      <button class="btn btn-primary" @click="executeSearch" :disabled="loading || !query">
        {{ loading ? '检索中...' : '搜索' }}
      </button>
      <div class="search-config">
        <label style="font-size: 12px;">
          结果数:
          <input type="number" v-model.number="topK" min="1" max="100" style="width: 60px;" />
        </label>
        <label style="font-size: 12px;">
          <input type="checkbox" v-model="rerank" />
          重排序
        </label>
      </div>
    </div>

    <!-- Results Split View -->
    <div class="search-results" v-if="results">
      <!-- Left: File Tree -->
      <aside class="file-tree">
        <div class="file-header">文件列表</div>
        <div
          class="file-item"
          :class="{ active: selectedFile === null }"
          @click="selectedFile = null"
        >
          📁 全部文件 ({{ results.total_results })
        </div>
        <div
          class="file-item"
          v-for="file in results.files"
          :key="file.filename"
          :class="{ active: selectedFile === file.filename }"
          @click="selectedFile = file.filename"
        >
          📄 {{ file.filename }} ({{ file.chunk_count }})
        </div>
      </aside>

      <!-- Right: Chunk Cards -->
      <div class="chunk-list">
        <div v-if="filteredChunks.length === 0" class="empty-state">
          <p>未找到相关内容</p>
        </div>
        <div
          class="chunk-card"
          v-for="chunk in filteredChunks"
          :key="chunk.chunk_id"
          :class="{ expanded: expandedChunk === chunk.chunk_id }"
          @click="toggleExpand(chunk.chunk_id)"
        >
          <div class="chunk-header">
            <span class="heading">{{ chunk.heading || '无标题' }}</span>
            <span class="score-badge">{{ chunk.score.toFixed(2) }}</span>
          </div>
          <div class="chunk-preview" v-if="expandedChunk !== chunk.chunk_id">
            {{ chunk.text_preview }}
          </div>
          <div class="chunk-full" v-else>
            <div class="chunk-meta">
              <span>文件: {{ chunk.source_file }}</span>
              <span v-if="chunk.header_path">路径: {{ chunk.header_path }}</span>
            </div>
            <div class="chunk-text">{{ chunk.full_text }}</div>
            <button class="btn btn-secondary collapse-btn" @click.stop="expandedChunk = null">
              收起
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty State (before search) -->
    <div class="empty-state" v-if="!results && !loading">
      <p>输入查询开始文档检索</p>
      <p style="color: var(--text-secondary); font-size: 13px;">
        仅返回检索结果，不生成回答
      </p>
    </div>

    <!-- Loading State -->
    <div class="loading-state" v-if="loading">
      <p>正在检索相关文档...</p>
    </div>
  </div>
</template>
```

- [ ] **Step 2: Add script section**

```vue
<script setup>
import { ref, computed } from 'vue'
import { searchApi } from '../api'

const query = ref('')
const topK = ref(20)
const rerank = ref(true)
const loading = ref(false)
const results = ref(null)
const selectedFile = ref(null)
const expandedChunk = ref(null)

const filteredChunks = computed(() => {
  if (!results.value) return []
  if (selectedFile.value === null) return results.value.chunks
  return results.value.chunks.filter(c => c.source_file === selectedFile.value)
})

async function executeSearch() {
  if (!query.value.trim() || loading.value) return

  const searchQuery = query.value.trim()
  loading.value = true
  expandedChunk.value = null
  selectedFile.value = null

  try {
    const res = await searchApi.query(searchQuery, topK.value, rerank.value)
    results.value = res.data
  } catch (err) {
    results.value = {
      query: searchQuery,
      total_results: 0,
      files: [],
      chunks: [],
      metadata: { error: err.message }
    }
  } finally {
    loading.value = false
  }
}

function toggleExpand(chunkId) {
  expandedChunk.value = expandedChunk.value === chunkId ? null : chunkId
}
</script>
```

- [ ] **Step 3: Add scoped styles**

```vue
<style scoped>
.search-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 48px);
}

.search-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 16px;
}

.search-input-area {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 16px;
  background: var(--bg-primary);
  border-radius: var(--border-radius);
  border: 1px solid var(--border);
  margin-bottom: 16px;
}

.search-input-area input[type="text"] {
  flex: 1;
  min-width: 300px;
}

.search-config {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-left: auto;
}

.search-results {
  display: flex;
  gap: 16px;
  flex: 1;
  overflow: hidden;
}

.file-tree {
  width: 220px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius);
  padding: 12px;
  overflow-y: auto;
}

.file-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
}

.file-item {
  padding: 10px 12px;
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 13px;
  transition: background 0.2s;
  margin-bottom: 4px;
}

.file-item:hover {
  background: var(--border);
}

.file-item.active {
  background: var(--primary);
  color: white;
}

.chunk-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 4px;
}

.chunk-card {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--border-radius);
  padding: 16px;
  cursor: pointer;
  transition: border-color 0.2s;
}

.chunk-card:hover {
  border-color: var(--primary);
}

.chunk-card.expanded {
  border: 2px solid var(--primary);
  background: #eff6ff;
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.heading {
  font-weight: 600;
  color: var(--text-primary);
}

.score-badge {
  background: #dcfce7;
  color: #166534;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}

.chunk-preview {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.5;
}

.chunk-full {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

.chunk-text {
  background: var(--bg-primary);
  padding: 12px;
  border-radius: var(--border-radius);
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.collapse-btn {
  align-self: flex-end;
  font-size: 12px;
  padding: 6px 12px;
}

.empty-state, .loading-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: var(--text-secondary);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .search-results {
    flex-direction: column;
  }

  .file-tree {
    width: 100%;
    max-height: 200px;
  }

  .search-input-area input[type="text"] {
    min-width: 100%;
  }

  .search-config {
    width: 100%;
    margin-left: 0;
  }
}
</style>
```

- [ ] **Step 4: Commit SearchView component**

```bash
git add web/frontend/src/views/SearchView.vue
git commit -m "feat(frontend): add SearchView component with split layout"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 7: Add Search Route

**Files:**
- Modify: `web/frontend/src/main.js`

- [ ] **Step 1: Import SearchView**

Add to imports (around line 6-9):
```javascript
import SearchView from './views/SearchView.vue'
```

- [ ] **Step 2: Add route definition**

Add to routes array (after ChatView route):
```javascript
{ path: '/search', name: 'SearchView', component: SearchView },
```

- [ ] **Step 3: Commit route addition**

```bash
git add web/frontend/src/main.js
git commit -m "feat(frontend): add /search route"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 8: Add Search Nav Item

**Files:**
- Modify: `web/frontend/src/App.vue`

- [ ] **Step 1: Add Search nav item to sidebar**

Add after the Chat nav-item (around line 24):
```vue
        <router-link to="/search" class="nav-item" active-class="active">
          <span class="icon">🔍</span>
          <span class="label">Search</span>
        </router-link>
```

- [ ] **Step 2: Commit nav item**

```bash
git add web/frontend/src/App.vue
git commit -m "feat(frontend): add Search nav item to sidebar"

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Task 9: Manual Testing

- [ ] **Step 1: Start backend server**

```bash
cd web/api && python main.py
```

Expected: Server starts on port 8000

- [ ] **Step 2: Start frontend dev server**

```bash
cd web/frontend && npm run dev
```

Expected: Vite starts on port 5173

- [ ] **Step 3: Test search functionality**

1. Open browser to http://localhost:5173
2. Click Search nav item - should navigate to /search
3. Enter a query and click Search
4. Verify results appear with file tree and chunk cards
5. Click a file in file tree - chunks should filter
6. Click a chunk card - should expand to show full text
7. Click again - should collapse

- [ ] **Step 4: Test edge cases**

1. Empty query - button should be disabled
2. No results - should show "未找到相关内容"
3. Change top_k value - should affect result count
4. Toggle rerank - should affect result order

---

## Self-Review Checklist

After plan completion, verify:

1. **Spec coverage:** All functional requirements covered (sidebar nav, query input, file tree, chunk cards, expand/collapse, configurable options)
2. **Placeholder scan:** No TBD/TODO placeholders - all code complete
3. **Type consistency:** SearchRequest/SearchResponse schemas match service output and frontend expectations