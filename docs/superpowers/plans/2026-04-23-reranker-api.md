# Reranker API Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for Cohere-compatible and DashScope reranker APIs alongside existing local CrossEncoder model.

**Architecture:** Factory pattern with abstract base class. Three implementations: CrossEncoderReranker (local), CohereReranker (API), DashScopeReranker (API). Unified Reranker entry point selects implementation based on provider config.

**Tech Stack:** Python, Pydantic, httpx (for API calls), sentence_transformers (local mode)

---

## File Structure

```
src/profirag/config/settings.py       # Modify: EnvSettings, RerankingConfig
src/profirag/retrieval/reranker.py    # Refactor: Add base class, API rerankers
src/profirag/retrieval/__init__.py    # Modify: Export new classes
.env.example                          # Modify: Add new env vars
tests/retrieval/test_reranker.py      # Create: Unit tests
```

---

### Task 1: Update EnvSettings with New Rerank Variables

**Files:**
- Modify: `src/profirag/config/settings.py:78-82`

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_rerank_config.py
"""Tests for rerank configuration"""

import os
from profirag.config.settings import EnvSettings, RerankingConfig


def test_env_settings_rerank_provider_default():
    """Test that rerank_provider defaults to 'local'."""
    original_provider = os.environ.pop("PROFIRAG_RERANK_PROVIDER", None)
    original_key = os.environ.pop("PROFIRAG_RERANK_API_KEY", None)
    original_url = os.environ.pop("PROFIRAG_RERANK_BASE_URL", None)
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_provider == "local"
    if original_provider:
        os.environ["PROFIRAG_RERANK_PROVIDER"] = original_provider
    if original_key:
        os.environ["PROFIRAG_RERANK_API_KEY"] = original_key
    if original_url:
        os.environ["PROFIRAG_RERANK_BASE_URL"] = original_url


def test_env_settings_rerank_provider_values():
    """Test that rerank_provider accepts valid values."""
    for provider in ["local", "cohere", "dashscope"]:
        os.environ["PROFIRAG_RERANK_PROVIDER"] = provider
        settings = EnvSettings(_env_file=None)
        assert settings.profirag_rerank_provider == provider
        del os.environ["PROFIRAG_RERANK_PROVIDER"]


def test_env_settings_rerank_api_key():
    """Test that rerank_api_key can be set."""
    os.environ["PROFIRAG_RERANK_API_KEY"] = "test-api-key"
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_api_key == "test-api-key"
    del os.environ["PROFIRAG_RERANK_API_KEY"]


def test_env_settings_rerank_base_url():
    """Test that rerank_base_url can be set."""
    os.environ["PROFIRAG_RERANK_BASE_URL"] = "https://api.example.com"
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_base_url == "https://api.example.com"
    del os.environ["PROFIRAG_RERANK_BASE_URL"]


def test_env_settings_rerank_timeout():
    """Test that rerank_timeout defaults to 30."""
    original = os.environ.pop("PROFIRAG_RERANK_TIMEOUT", None)
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_timeout == 30
    if original:
        os.environ["PROFIRAG_RERANK_TIMEOUT"] = original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_rerank_config.py -v`
Expected: FAIL with "profirag_rerank_provider field not found"

- [ ] **Step 3: Add new fields to EnvSettings**

Modify `src/profirag/config/settings.py`, add after line 82 (after `profirag_rerank_top_n`):

```python
    # Reranking Configuration
    profirag_rerank_enabled: bool = True
    profirag_rerank_provider: Literal["local", "cohere", "dashscope"] = "local"
    profirag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    profirag_rerank_top_n: int = 5
    profirag_rerank_api_key: Optional[str] = None
    profirag_rerank_base_url: Optional[str] = None
    profirag_rerank_timeout: int = 30
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_rerank_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/config/settings.py tests/config/test_rerank_config.py
git commit -m "feat(config): add rerank provider config fields"
```

---

### Task 2: Update RerankingConfig Model

**Files:**
- Modify: `src/profirag/config/settings.py:147-152`

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_rerank_config.py (append to existing file)

def test_reranking_config_provider_field():
    """Test RerankingConfig has provider field."""
    config = RerankingConfig(provider="cohere")
    assert config.provider == "cohere"


def test_reranking_config_api_key_field():
    """Test RerankingConfig has api_key field."""
    config = RerankingConfig(api_key="test-key")
    assert config.api_key == "test-key"


def test_reranking_config_base_url_field():
    """Test RerankingConfig has base_url field."""
    config = RerankingConfig(base_url="https://api.example.com")
    assert config.base_url == "https://api.example.com"


def test_reranking_config_timeout_field():
    """Test RerankingConfig has timeout field."""
    config = RerankingConfig(timeout=60)
    assert config.timeout == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_rerank_config.py::test_reranking_config_provider_field -v`
Expected: FAIL

- [ ] **Step 3: Update RerankingConfig class**

Replace `src/profirag/config/settings.py` lines 147-152:

```python
class RerankingConfig(BaseModel):
    """Reranking configuration"""
    enabled: bool = True
    provider: Literal["local", "cohere", "dashscope"] = "local"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_rerank_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/config/settings.py tests/config/test_rerank_config.py
git commit -m "feat(config): update RerankingConfig with API fields"
```

---

### Task 3: Update RAGConfig.from_env to Pass Rerank Config

**Files:**
- Modify: `src/profirag/config/settings.py:270-274`

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_rerank_config.py (append)

def test_rag_config_from_env_rerank_provider():
    """Test RAGConfig.from_env passes rerank_provider."""
    os.environ["PROFIRAG_RERANK_PROVIDER"] = "cohere"
    os.environ["PROFIRAG_RERANK_API_KEY"] = "test-key"
    os.environ["PROFIRAG_RERANK_BASE_URL"] = "https://api.cohere.ai"
    config = RAGConfig.from_env()
    assert config.reranking.provider == "cohere"
    assert config.reranking.api_key == "test-key"
    assert config.reranking.base_url == "https://api.cohere.ai"
    del os.environ["PROFIRAG_RERANK_PROVIDER"]
    del os.environ["PROFIRAG_RERANK_API_KEY"]
    del os.environ["PROFIRAG_RERANK_BASE_URL"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_rerank_config.py::test_rag_config_from_env_rerank_provider -v`
Expected: FAIL

- [ ] **Step 3: Update RAGConfig.from_env reranking section**

Replace `src/profirag/config/settings.py` lines 270-274:

```python
            reranking=RerankingConfig(
                enabled=env_settings.profirag_rerank_enabled,
                provider=env_settings.profirag_rerank_provider,
                model=env_settings.profirag_rerank_model,
                top_n=env_settings.profirag_rerank_top_n,
                api_key=env_settings.profirag_rerank_api_key,
                base_url=env_settings.profirag_rerank_base_url,
                timeout=env_settings.profirag_rerank_timeout,
            ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/config/test_rerank_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/config/settings.py tests/config/test_rerank_config.py
git commit -m "feat(config): pass rerank API config to RAGConfig"
```

---

### Task 4: Add BaseReranker Abstract Class

**Files:**
- Modify: `src/profirag/retrieval/reranker.py:1-10`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py
"""Tests for reranker implementations"""

from profirag.retrieval.reranker import BaseReranker


def test_base_reranker_is_abstract():
    """Test BaseReranker cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        BaseReranker(top_n=5)


def test_base_reranker_has_rerank_method():
    """Test BaseReranker defines rerank abstract method."""
    from abc import ABC
    assert hasattr(BaseReranker, 'rerank')
    # Check it's abstract
    assert BaseReranker.__bases__[0] == ABC
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: FAIL with "cannot import name 'BaseReranker'"

- [ ] **Step 3: Add BaseReranker class**

Add to `src/profirag/retrieval/reranker.py` after imports (around line 7):

```python
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Abstract base class for reranker implementations."""

    top_n: int = 5

    @abstractmethod
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes by relevance to query.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects
        """
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat(reranker): add BaseReranker abstract class"
```

---

### Task 5: Add CohereReranker Implementation

**Files:**
- Modify: `src/profirag/retrieval/reranker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py (append)

import pytest
from unittest.mock import patch, MagicMock
from profirag.retrieval.reranker import CohereReranker
from llama_index.core.schema import NodeWithScore, TextNode


def test_cohere_reranker_init():
    """Test CohereReranker initialization."""
    reranker = CohereReranker(
        api_key="test-key",
        base_url="https://api.cohere.ai",
        model="rerank-v1",
        top_n=5
    )
    assert reranker.api_key == "test-key"
    assert reranker.base_url == "https://api.cohere.ai"
    assert reranker.model == "rerank-v1"
    assert reranker.top_n == 5


def test_cohere_reranker_requires_api_key():
    """Test CohereReranker raises error without api_key."""
    with pytest.raises(ValueError, match="api_key is required"):
        CohereReranker(api_key=None, base_url="https://api.cohere.ai")


def test_cohere_reranker_requires_base_url():
    """Test CohereReranker raises error without base_url."""
    with pytest.raises(ValueError, match="base_url is required"):
        CohereReranker(api_key="test-key", base_url=None)


@patch("httpx.post")
def test_cohere_reranker_rerank(mock_post):
    """Test CohereReranker rerank method."""
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.75}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = CohereReranker(
        api_key="test-key",
        base_url="https://api.cohere.ai",
        model="rerank-v1",
        top_n=2
    )

    nodes = [
        NodeWithScore(node=TextNode(text="Document 0"), score=0.5),
        NodeWithScore(node=TextNode(text="Document 1"), score=0.6)
    ]

    result = reranker.rerank("test query", nodes)

    assert len(result) == 2
    assert result[0].node.text == "Document 1"
    assert result[0].score == 0.95
    assert result[1].node.text == "Document 0"
    assert result[1].score == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_reranker.py::test_cohere_reranker_init -v`
Expected: FAIL with "cannot import name 'CohereReranker'"

- [ ] **Step 3: Add CohereReranker class**

Add to `src/profirag/retrieval/reranker.py` after BaseReranker:

```python
import httpx


class CohereReranker(BaseReranker):
    """Cohere-compatible API reranker.

    Supports Cohere rerank API format and compatible services.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model: str = "rerank-v1",
        top_n: int = 5,
        timeout: int = 30,
        **kwargs
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: API key (required)
            base_url: API base URL (required)
            model: Model name
            top_n: Number of results to return
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Raises:
            ValueError: If api_key or base_url is not provided
        """
        if not api_key:
            raise ValueError("api_key is required for Cohere reranker")
        if not base_url:
            raise ValueError("base_url is required for Cohere reranker")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.kwargs = kwargs

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes using Cohere API.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects

        Raises:
            RuntimeError: If API call fails
        """
        if not nodes:
            return nodes

        documents = [node.node.text for node in nodes]

        # Build request
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": self.top_n,
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Cohere API error: {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Cohere API request failed: {str(e)}")

        # Parse results
        results = data.get("results", [])
        reranked = []
        for r in results:
            idx = r["index"]
            score = r["relevance_score"]
            reranked.append(NodeWithScore(node=nodes[idx].node, score=score))

        return reranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat(reranker): add CohereReranker implementation"
```

---

### Task 6: Add DashScopeReranker Implementation

**Files:**
- Modify: `src/profirag/retrieval/reranker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py (append)

from profirag.retrieval.reranker import DashScopeReranker


def test_dashscope_reranker_init():
    """Test DashScopeReranker initialization."""
    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        model="rerank-v1",
        top_n=5
    )
    assert reranker.api_key == "test-key"
    assert reranker.base_url == "https://dashscope.aliyuncs.com"
    assert reranker.model == "rerank-v1"
    assert reranker.top_n == 5


def test_dashscope_reranker_requires_api_key():
    """Test DashScopeReranker raises error without api_key."""
    with pytest.raises(ValueError, match="api_key is required"):
        DashScopeReranker(api_key=None, base_url="https://dashscope.aliyuncs.com")


def test_dashscope_reranker_requires_base_url():
    """Test DashScopeReranker raises error without base_url."""
    with pytest.raises(ValueError, match="base_url is required"):
        DashScopeReranker(api_key="test-key", base_url=None)


@patch("httpx.post")
def test_dashscope_reranker_rerank(mock_post):
    """Test DashScopeReranker rerank method with wrapped response."""
    # Mock DashScope response format
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.68}
            ]
        },
        "request_id": "test-req-id"
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        model="rerank-v1",
        top_n=2
    )

    nodes = [
        NodeWithScore(node=TextNode(text="Document 0"), score=0.5),
        NodeWithScore(node=TextNode(text="Document 1"), score=0.6)
    ]

    result = reranker.rerank("test query", nodes)

    assert len(result) == 2
    assert result[0].node.text == "Document 1"
    assert result[0].score == 0.92


@patch("httpx.post")
def test_dashscope_reranker_request_format(mock_post):
    """Test DashScopeReranker sends correct request format."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "output": {"results": []},
        "request_id": "test"
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    reranker = DashScopeReranker(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        model="rerank-v1",
        top_n=5
    )

    nodes = [NodeWithScore(node=TextNode(text="doc"), score=0.5)]
    reranker.rerank("query", nodes)

    # Verify request format
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    # DashScope uses nested input structure
    assert "input" in payload
    assert payload["input"]["query"] == "query"
    assert payload["input"]["documents"] == ["doc"]
    assert payload["input"]["top_n"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_reranker.py::test_dashscope_reranker_init -v`
Expected: FAIL

- [ ] **Step 3: Add DashScopeReranker class**

Add to `src/profirag/retrieval/reranker.py` after CohereReranker:

```python


class DashScopeReranker(BaseReranker):
    """Alibaba Cloud DashScope reranker.

    Uses DashScope's specific API format with nested input structure.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model: str = "rerank-v1",
        top_n: int = 5,
        timeout: int = 30,
        **kwargs
    ):
        """Initialize DashScope reranker.

        Args:
            api_key: DashScope API key (required)
            base_url: DashScope API base URL (required)
            model: Model name (e.g., "rerank-v1")
            top_n: Number of results to return
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Raises:
            ValueError: If api_key or base_url is not provided
        """
        if not api_key:
            raise ValueError("api_key is required for DashScope reranker")
        if not base_url:
            raise ValueError("base_url is required for DashScope reranker")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.kwargs = kwargs

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes using DashScope API.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects

        Raises:
            RuntimeError: If API call fails
        """
        if not nodes:
            return nodes

        documents = [node.node.text for node in nodes]

        # Build request - DashScope uses nested input structure
        url = f"{self.base_url}/api/v1/services/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
                "top_n": self.top_n,
            }
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"DashScope API error: {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"DashScope API request failed: {str(e)}")

        # Parse results - DashScope wraps in "output"
        output = data.get("output", {})
        results = output.get("results", [])
        reranked = []
        for r in results:
            idx = r["index"]
            score = r["relevance_score"]
            reranked.append(NodeWithScore(node=nodes[idx].node, score=score))

        return reranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat(reranker): add DashScopeReranker implementation"
```

---

### Task 7: Update CrossEncoderReranker to Extend BaseReranker

**Files:**
- Modify: `src/profirag/retrieval/reranker.py:9-95`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py (append)

from profirag.retrieval.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_extends_base():
    """Test CrossEncoderReranker extends BaseReranker."""
    assert issubclass(CrossEncoderReranker, BaseReranker)


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_reranker_rerank_method(mock_ce):
    """Test CrossEncoderReranker has rerank method from BaseReranker."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.5]
    mock_ce.return_value = mock_model

    reranker = CrossEncoderReranker(model="test-model", top_n=2)
    nodes = [
        NodeWithScore(node=TextNode(text="doc1"), score=0.5),
        NodeWithScore(node=TextNode(text="doc2"), score=0.6)
    ]

    result = reranker.rerank("query", nodes)

    assert hasattr(reranker, 'rerank')
    assert len(result) == 2
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/retrieval/test_reranker.py::test_cross_encoder_reranker_extends_base -v`
Expected: May FAIL or PASS depending on current state

- [ ] **Step 3: Update CrossEncoderReranker class**

Replace the CrossEncoderReranker class definition (lines 9-95) with:

```python
class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers.

    Uses a cross-encoder model to compute relevance scores for
    query-document pairs and reorder results.
    """

    model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model name or path"
    )
    top_n: int = Field(default=5, description="Number of results to return")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    device: Optional[str] = Field(default=None, description="Device to use")

    _model: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize cross-encoder reranker.

        Args:
            model: Cross-encoder model name or path
            top_n: Number of top results to return after reranking
            batch_size: Batch size for encoding
            device: Device to use ("cuda", "cpu", None for auto)
            **kwargs: Additional arguments
        """
        super().__init__(
            model=model,
            top_n=top_n,
            batch_size=batch_size,
            device=device,
            **kwargs
        )
        self._model = None

    def _load_model(self) -> None:
        """Load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model, device=self.device)

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes based on cross-encoder scores.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not nodes:
            return nodes

        self._load_model()

        # Prepare query-document pairs
        pairs = [(query, node.node.text) for node in nodes]

        # Compute relevance scores
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        # Create reranked results
        reranked = [
            NodeWithScore(node=nodes[i].node, score=float(scores[i]))
            for i in range(len(nodes))
        ]

        # Sort by score and limit to top_n
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked[:self.top_n]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Legacy method for llama_index compatibility.

        Args:
            nodes: List of NodeWithScore objects to rerank
            query_bundle: QueryBundle containing the query string

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not nodes or not query_bundle:
            return nodes
        return self.rerank(query_bundle.query_str, nodes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/profirag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "refactor(reranker): CrossEncoderReranker extends BaseReranker"
```

---

### Task 8: Update Reranker Factory Class

**Files:**
- Modify: `src/profirag/retrieval/reranker.py:98-179`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_reranker.py (append)

from profirag.config.settings import RerankingConfig
from profirag.retrieval.reranker import Reranker


def test_reranker_factory_local():
    """Test Reranker factory creates local reranker."""
    config = RerankingConfig(provider="local", model="test-model", top_n=5)
    reranker = Reranker(config)
    assert isinstance(reranker._impl, CrossEncoderReranker)


def test_reranker_factory_cohere():
    """Test Reranker factory creates Cohere reranker."""
    config = RerankingConfig(
        provider="cohere",
        api_key="test-key",
        base_url="https://api.cohere.ai",
        top_n=5
    )
    reranker = Reranker(config)
    assert isinstance(reranker._impl, CohereReranker)


def test_reranker_factory_dashscope():
    """Test Reranker factory creates DashScope reranker."""
    config = RerankingConfig(
        provider="dashscope",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com",
        top_n=5
    )
    reranker = Reranker(config)
    assert isinstance(reranker._impl, DashScopeReranker)


def test_reranker_factory_disabled():
    """Test Reranker with enabled=False."""
    config = RerankingConfig(enabled=False, provider="local")
    reranker = Reranker(config)
    assert reranker.enabled == False


def test_reranker_factory_cohere_missing_key():
    """Test Reranker raises error for Cohere without api_key."""
    config = RerankingConfig(provider="cohere", base_url="https://api.cohere.ai")
    with pytest.raises(ValueError):
        Reranker(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_reranker.py::test_reranker_factory_local -v`
Expected: FAIL (factory not updated yet)

- [ ] **Step 3: Update Reranker factory class**

Replace the Reranker class (lines 98-179) with:

```python
class Reranker:
    """Flexible reranker supporting multiple reranking strategies.

    Factory class that selects the appropriate reranker implementation
    based on configuration.
    """

    def __init__(self, config: RerankingConfig):
        """Initialize reranker from configuration.

        Args:
            config: RerankingConfig instance

        Raises:
            ValueError: If API mode is selected but api_key/base_url missing
        """
        self.config = config
        self.enabled = config.enabled
        self.top_n = config.top_n
        self._impl: Optional[BaseReranker] = None

        if config.enabled:
            self._impl = self._create_reranker(config)

    def _create_reranker(self, config: RerankingConfig) -> BaseReranker:
        """Create appropriate reranker based on provider.

        Args:
            config: RerankingConfig instance

        Returns:
            Reranker implementation instance

        Raises:
            ValueError: If API mode requires missing config
        """
        if config.provider == "local":
            return CrossEncoderReranker(
                model=config.model,
                top_n=config.top_n,
            )
        elif config.provider == "cohere":
            if not config.api_key or not config.base_url:
                raise ValueError(
                    f"api_key and base_url are required for {config.provider} reranker"
                )
            return CohereReranker(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                top_n=config.top_n,
                timeout=config.timeout,
            )
        elif config.provider == "dashscope":
            if not config.api_key or not config.base_url:
                raise ValueError(
                    f"api_key and base_url are required for {config.provider} reranker"
                )
            return DashScopeReranker(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                top_n=config.top_n,
                timeout=config.timeout,
            )
        else:
            raise ValueError(f"Unknown reranker provider: {config.provider}")

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes by relevance to query.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not self.enabled or not self._impl:
            return nodes[:self.top_n]

        if not nodes:
            return nodes

        return self._impl.rerank(query, nodes, **kwargs)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable reranking.

        Args:
            enabled: Whether to enable reranking
        """
        self.enabled = enabled
        if enabled and self._impl is None:
            self._impl = self._create_reranker(self.config)

    def set_top_n(self, top_n: int) -> None:
        """Update number of results to return.

        Args:
            top_n: New top_n value
        """
        self.top_n = top_n
        if self._impl:
            self._impl.top_n = top_n
```

- [ ] **Step 4: Add import for RerankingConfig**

Add to imports in `src/profirag/retrieval/reranker.py`:

```python
from profirag.config.settings import RerankingConfig
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/retrieval/test_reranker.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/profirag/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "refactor(reranker): update Reranker factory for API providers"
```

---

### Task 9: Update .env.example

**Files:**
- Modify: `.env.example:53-56`

- [ ] **Step 1: Update .env.example rerank section**

Replace lines 53-56 in `.env.example`:

```bash
# ==================== Reranking Configuration ====================
PROFIRAG_RERANK_ENABLED=true
PROFIRAG_RERANK_PROVIDER=local  # local (CrossEncoder), cohere, dashscope
PROFIRAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
PROFIRAG_RERANK_TOP_N=5
PROFIRAG_RERANK_TIMEOUT=30

# For API-based reranking (required when provider != local)
# PROFIRAG_RERANK_API_KEY=your-api-key
# PROFIRAG_RERANK_BASE_URL=https://api.cohere.ai
# DashScope example: PROFIRAG_RERANK_BASE_URL=https://dashscope.aliyuncs.com
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with rerank API config"
```

---

### Task 10: Update Exports

**Files:**
- Modify: `src/profirag/retrieval/__init__.py`

- [ ] **Step 1: Update __init__.py exports**

Replace `src/profirag/retrieval/__init__.py`:

```python
"""Retrieval components"""

from .query_transform import PreRetrievalPipeline
from .hybrid import HybridRetriever
from .reranker import Reranker, BaseReranker, CrossEncoderReranker, CohereReranker, DashScopeReranker

__all__ = [
    "PreRetrievalPipeline",
    "HybridRetriever",
    "Reranker",
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "DashScopeReranker",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/profirag/retrieval/__init__.py
git commit -m "feat: export new reranker classes"
```

---

### Task 11: Run Full Test Suite

**Files:**
- None

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Fix any failures**

If tests fail, fix the issues.

---

### Task 12: Final Commit and Cleanup

**Files:**
- None

- [ ] **Step 1: Remove LLMReranker if unused**

Check if `LLMReranker` class is used anywhere:

```bash
grep -r "LLMRereranker" src/ --include="*.py"
```

If not used, remove it from `reranker.py`.

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete reranker API support"
```