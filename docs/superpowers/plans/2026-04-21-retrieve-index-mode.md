# Retrieve Index Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `profirag_retrieve_index_mode` environment variable to control retrieval mode (hybrid, sparse, vector), leveraging LlamaIndex's native VectorStoreQueryMode support.

**Architecture:** Config layer adds retrieve_mode setting → HybridRetriever maps mode to VectorStoreQueryMode → Native LlamaIndex retriever handles all three modes internally. No custom sparse query implementation needed.

**Tech Stack:** Python, Pydantic, LlamaIndex, Qdrant, pytest

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `src/profirag/config/settings.py` | Add retrieve_mode to EnvSettings and RetrievalConfig, pass through RAGConfig.from_env | Modify |
| `src/profirag/retrieval/hybrid.py` | Add retrieve_mode parameter, map to VectorStoreQueryMode, use native retriever | Modify |
| `src/profirag/pipeline/rag_pipeline.py` | Pass retrieve_mode from config to HybridRetriever | Modify |
| `.env.example` | Document PROFIRAG_RETRIEVE_INDEX_MODE env var | Modify |
| `tests/retrieval/test_retrieve_mode.py` | Unit tests for retrieve_mode configuration and mapping | Create |

---

### Task 1: Configuration - Add retrieve_mode to settings.py

**Files:**
- Modify: `src/profirag/config/settings.py`
- Create: `tests/config/test_retrieve_mode_config.py`

- [ ] **Step 1: Write the failing test for EnvSettings retrieve_mode**

Create `tests/config/test_retrieve_mode_config.py`:

```python
"""Tests for retrieve_mode configuration"""

from profirag.config.settings import EnvSettings, RetrievalConfig, RAGConfig


def test_env_settings_retrieve_mode_default():
    """Test that retrieve_mode defaults to 'hybrid'."""
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "hybrid"


def test_env_settings_retrieve_mode_values():
    """Test that retrieve_mode accepts valid values."""
    # Test via environment variable simulation
    import os
    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "sparse"
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "sparse"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]

    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "vector"
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "vector"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]


def test_retrieval_config_retrieve_mode():
    """Test RetrievalConfig has retrieve_mode field."""
    config = RetrievalConfig(retrieve_mode="sparse")
    assert config.retrieve_mode == "sparse"


def test_rag_config_from_env_includes_retrieve_mode():
    """Test that RAGConfig.from_env passes retrieve_mode."""
    import os
    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "vector"
    config = RAGConfig.from_env()
    assert config.retrieval.retrieve_mode == "vector"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_retrieve_mode_config.py -v`
Expected: FAIL - `profirag_retrieve_index_mode` attribute not found

- [ ] **Step 3: Add retrieve_mode to EnvSettings class**

Modify `src/profirag/config/settings.py` line 75 (after `profirag_index_mode`):

```python
# In EnvSettings class, add after profirag_index_mode:
profirag_retrieve_index_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"
```

- [ ] **Step 4: Add retrieve_mode to RetrievalConfig class**

Modify `src/profirag/config/settings.py` line 143 (in RetrievalConfig):

```python
class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = 10
    alpha: float = 0.5  # Vector search weight (1-alpha for BM25)
    use_hybrid: bool = True
    retrieve_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"
```

- [ ] **Step 5: Pass retrieve_mode in RAGConfig.from_env**

Modify `src/profirag/config/settings.py` around line 264-267 (in `from_env` method):

```python
retrieval=RetrievalConfig(
    top_k=env_settings.profirag_top_k,
    alpha=env_settings.profirag_alpha,
    retrieve_mode=env_settings.profirag_retrieve_index_mode,
),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/config/test_retrieve_mode_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit config changes**

```bash
git add src/profirag/config/settings.py tests/config/test_retrieve_mode_config.py
git commit -m "feat(config): add profirag_retrieve_index_mode configuration"
```

---

### Task 2: HybridRetriever - Add retrieve_mode support

**Files:**
- Modify: `src/profirag/retrieval/hybrid.py`
- Create: `tests/retrieval/test_retrieve_mode.py`

- [ ] **Step 1: Write the failing test for _map_retrieve_mode**

Create `tests/retrieval/test_retrieve_mode.py`:

```python
"""Tests for retrieve_mode in HybridRetriever"""

import pytest
from unittest.mock import MagicMock
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from profirag.retrieval.hybrid import HybridRetriever


class TestRetrieveModeMapping:
    """Test retrieve_mode to VectorStoreQueryMode mapping."""

    def test_map_retrieve_mode_hybrid(self):
        """Test hybrid mode maps to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode("hybrid")
        assert mode == VectorStoreQueryMode.HYBRID

    def test_map_retrieve_mode_sparse(self):
        """Test sparse mode maps to SPARSE."""
        mode = HybridRetriever._map_retrieve_mode("sparse")
        assert mode == VectorStoreQueryMode.SPARSE

    def test_map_retrieve_mode_vector(self):
        """Test vector mode maps to DEFAULT."""
        mode = HybridRetriever._map_retrieve_mode("vector")
        assert mode == VectorStoreQueryMode.DEFAULT

    def test_map_retrieve_mode_invalid_defaults_to_hybrid(self):
        """Test invalid mode defaults to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode("invalid")
        assert mode == VectorStoreQueryMode.HYBRID

    def test_map_retrieve_mode_none_defaults_to_hybrid(self):
        """Test None defaults to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode(None)
        assert mode == VectorStoreQueryMode.HYBRID


class TestHybridRetrieverRetrieveMode:
    """Test HybridRetriever initialization with retrieve_mode."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        index = MagicMock()
        index.as_retriever = MagicMock(return_value=MagicMock())
        return index

    def test_init_with_retrieve_mode_hybrid(self, mock_index):
        """Test initialization with hybrid mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="hybrid",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "hybrid"
        assert retriever._query_mode == VectorStoreQueryMode.HYBRID

    def test_init_with_retrieve_mode_sparse(self, mock_index):
        """Test initialization with sparse mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="sparse",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "sparse"
        assert retriever._query_mode == VectorStoreQueryMode.SPARSE

    def test_init_with_retrieve_mode_vector(self, mock_index):
        """Test initialization with vector mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="vector",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "vector"
        assert retriever._query_mode == VectorStoreQueryMode.DEFAULT

    def test_init_passes_query_mode_to_as_retriever(self, mock_index):
        """Test that vector_store_query_mode is passed to as_retriever."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="sparse",
            alpha=0.7,
        )
        mock_index.as_retriever.assert_called_once()
        call_kwargs = mock_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE
        assert call_kwargs["alpha"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retrieve_mode.py -v`
Expected: FAIL - `_map_retrieve_mode` method not found

- [ ] **Step 3: Add import for VectorStoreQueryMode**

Modify `src/profirag/retrieval/hybrid.py` line 3-5:

```python
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
```

- [ ] **Step 4: Add retrieve_mode parameter to __init__**

Modify `src/profirag/retrieval/hybrid.py` __init__ method (lines 17-44):

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
    """Initialize hybrid retriever.

    Args:
        vector_index: LlamaIndex VectorStoreIndex for vector search
        alpha: Weight for vector search (1-alpha for BM25)
               Default 0.5 means equal weight
        rrf_k: RRF constant for smoothing (default 60)
        vector_store: Optional BaseVectorStore reference.
                     If it has native BM25 (use_bm25=True), retrieval
                     is delegated to vector_store.query().
        retrieve_mode: Retrieval mode - "hybrid" (dense+BM25), "sparse" (BM25 only),
                      or "vector" (dense only). Default is "hybrid".
        **kwargs: Additional arguments passed to as_retriever
    """
    self.vector_index = vector_index
    self.alpha = alpha
    self.rrf_k = rrf_k
    self.vector_store = vector_store
    self.retrieve_mode = retrieve_mode
    self.kwargs = kwargs

    # Map retrieve_mode to VectorStoreQueryMode
    self._query_mode = self._map_retrieve_mode(retrieve_mode)

    # Create retriever with native LlamaIndex support
    retriever_kwargs = kwargs.copy()
    retriever_kwargs["vector_store_query_mode"] = self._query_mode
    retriever_kwargs["alpha"] = alpha

    self._vector_retriever = vector_index.as_retriever(**retriever_kwargs) if vector_index is not None else None
```

- [ ] **Step 5: Add _map_retrieve_mode static method**

Add after the `__init__` method (around line 52):

```python
@staticmethod
def _map_retrieve_mode(mode: Optional[str]) -> VectorStoreQueryMode:
    """Map retrieve_mode string to VectorStoreQueryMode enum.

    Args:
        mode: Retrieve mode string ("hybrid", "sparse", "vector")

    Returns:
        VectorStoreQueryMode enum value
    """
    mode_map = {
        "hybrid": VectorStoreQueryMode.HYBRID,
        "sparse": VectorStoreQueryMode.SPARSE,
        "vector": VectorStoreQueryMode.DEFAULT,
    }
    return mode_map.get(mode, VectorStoreQueryMode.HYBRID)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/retrieval/test_retrieve_mode.py -v`
Expected: PASS

- [ ] **Step 7: Commit HybridRetriever changes**

```bash
git add src/profirag/retrieval/hybrid.py tests/retrieval/test_retrieve_mode.py
git commit -m "feat(retrieval): add retrieve_mode support to HybridRetriever"
```

---

### Task 3: RAGPipeline - Pass retrieve_mode from config

**Files:**
- Modify: `src/profirag/pipeline/rag_pipeline.py`

- [ ] **Step 1: Update HybridRetriever initialization**

Modify `src/profirag/pipeline/rag_pipeline.py` lines 94-98:

Find the existing code:
```python
self._hybrid_retriever = HybridRetriever(
    vector_index=self._index,
    alpha=config.retrieval.alpha,
    vector_store=self._vector_store,
)
```

Replace with:
```python
self._hybrid_retriever = HybridRetriever(
    vector_index=self._index,
    alpha=config.retrieval.alpha,
    vector_store=self._vector_store,
    retrieve_mode=config.retrieval.retrieve_mode,
)
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `pytest tests/pipeline/ -v`
Expected: PASS (existing tests should still work)

- [ ] **Step 3: Commit pipeline changes**

```bash
git add src/profirag/pipeline/rag_pipeline.py
git commit -m "feat(pipeline): pass retrieve_mode to HybridRetriever"
```

---

### Task 4: Documentation - Update .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add PROFIRAG_RETRIEVE_INDEX_MODE to .env.example**

Modify `.env.example` in the Retrieval Configuration section (around line 38-42):

Find existing:
```bash
# ==================== Retrieval Configuration ====================
PROFIRAG_TOP_K=10
PROFIRAG_ALPHA=0.5
PROFIRAG_USE_HYBRID=true
PROFIRAG_USE_BM25=true
```

Replace with:
```bash
# ==================== Retrieval Configuration ====================
PROFIRAG_TOP_K=10
PROFIRAG_ALPHA=0.5
PROFIRAG_RETRIEVE_INDEX_MODE=hybrid  # hybrid (dense+BM25), sparse (BM25 only), vector (dense only)
PROFIRAG_USE_HYBRID=true
PROFIRAG_USE_BM25=true
```

- [ ] **Step 2: Commit documentation changes**

```bash
git add .env.example
git commit -m "docs: add PROFIRAG_RETRIEVE_INDEX_MODE to .env.example"
```

---

### Task 5: Integration Test - Verify retrieval modes work

**Files:**
- Create: `tests/integration/test_retrieve_modes.py`

- [ ] **Step 1: Write integration test for all three modes**

Create `tests/integration/test_retrieve_modes.py`:

```python
"""Integration tests for retrieve_mode functionality"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import NodeWithScore, TextNode

from profirag.config.settings import RAGConfig, RetrievalConfig
from profirag.retrieval.hybrid import HybridRetriever


class TestRetrieveModesIntegration:
    """Test that all three retrieve modes work correctly."""

    @pytest.fixture
    def mock_vector_index(self):
        """Create a mock VectorStoreIndex with retriever."""
        index = MagicMock()

        # Mock retriever that returns different results based on mode
        def mock_retrieve(query):
            return [
                NodeWithScore(node=TextNode(id_="node-1", text=f"Result for {query}"), score=0.9)
            ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = mock_retrieve
        index.as_retriever = MagicMock(return_value=mock_retriever)
        index._embed_model = MagicMock()
        index._embed_model.get_text_embedding = MagicMock(return_value=[0.1] * 1536)

        return index

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.has_native_bm25 = MagicMock(return_value=True)
        store.query = MagicMock(return_value=[
            NodeWithScore(node=TextNode(id_="node-2", text="BM25 result"), score=0.85)
        ])
        return store

    def test_hybrid_mode_creates_hybrid_retriever(self, mock_vector_index, mock_vector_store):
        """Test that hybrid mode creates retriever with HYBRID query mode."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="hybrid",
        )

        # Verify as_retriever was called with HYBRID mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.HYBRID

    def test_sparse_mode_creates_sparse_retriever(self, mock_vector_index, mock_vector_store):
        """Test that sparse mode creates retriever with SPARSE query mode."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="sparse",
        )

        # Verify as_retriever was called with SPARSE mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE

    def test_vector_mode_creates_default_retriever(self, mock_vector_index, mock_vector_store):
        """Test that vector mode creates retriever with DEFAULT query mode."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="vector",
        )

        # Verify as_retriever was called with DEFAULT mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.DEFAULT

    def test_retrieve_delegates_to_vector_retriever(self, mock_vector_index, mock_vector_store):
        """Test that retrieve() delegates to the internal vector retriever."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="hybrid",
        )

        results = retriever.retrieve("test query", top_k=5)

        # Should have called the retriever
        assert len(results) == 1
        assert results[0].node.text == "Result for test query"

    def test_config_integration(self):
        """Test that RAGConfig properly passes retrieve_mode."""
        config = RAGConfig(
            storage={"type": "qdrant", "config": {"collection_name": "test"}},
            retrieval=RetrievalConfig(retrieve_mode="sparse"),
        )

        assert config.retrieval.retrieve_mode == "sparse"
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/integration/test_retrieve_modes.py -v`
Expected: PASS

- [ ] **Step 3: Create tests/integration/__init__.py if needed**

If the directory doesn't exist:
```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

- [ ] **Step 4: Run all tests to verify everything works**

Run: `pytest tests/ -v --tb=short`
Expected: PASS (all tests pass)

- [ ] **Step 5: Commit integration tests**

```bash
git add tests/integration/
git commit -m "test(integration): add retrieve_mode integration tests"
```

---

### Task 6: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify lint/format**

Run: `python -m py_compile src/profirag/config/settings.py src/profirag/retrieval/hybrid.py src/profirag/pipeline/rag_pipeline.py`
Expected: No syntax errors

- [ ] **Step 3: Final commit if any fixes needed**

If there were any fixes:
```bash
git add -A
git commit -m "fix: final cleanup for retrieve_mode implementation"
```

- [ ] **Step 4: Push changes**

```bash
git push origin bm25_enhanmement
```

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] Config: EnvSettings.retrieve_mode ✓ Task 1
- [x] Config: RetrievalConfig.retrieve_mode ✓ Task 1
- [x] Config: RAGConfig.from_env passes retrieve_mode ✓ Task 1
- [x] HybridRetriever._map_retrieve_mode ✓ Task 2
- [x] HybridRetriever.__init__ retrieve_mode param ✓ Task 2
- [x] HybridRetriever passes VectorStoreQueryMode to as_retriever ✓ Task 2
- [x] RAGPipeline passes retrieve_mode ✓ Task 3
- [x] .env.example documentation ✓ Task 4
- [x] Unit tests ✓ Task 1, 2
- [x] Integration tests ✓ Task 5

**2. Placeholder scan:**
- [x] No TBD/TODO found
- [x] All code blocks contain actual code
- [x] All test assertions are specific

**3. Type consistency:**
- [x] `retrieve_mode` uses `Literal["hybrid", "sparse", "vector"]` consistently
- [x] `_map_retrieve_mode` returns `VectorStoreQueryMode` consistently
- [x] Method signatures match across tasks