# FastEmbed Dense Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FastEmbed support for local dense vectorization as alternative to OpenAI API.

**Architecture:** Create FastEmbedEmbedding class inheriting from BaseEmbedding, wrapping fastembed.TextEmbedding. Provider enum switches between OpenAI and FastEmbed at config level. Factory pattern in RAGPipeline selects embedding class.

**Tech Stack:** fastembed>=0.2.0, llama-index-core (BaseEmbedding), asyncio for async wrappers

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add fastembed dependency |
| `.env.example` | Modify | Add FastEmbed env var documentation |
| `src/profirag/config/settings.py` | Modify | Add EnvSettings fields, update EmbeddingConfig, add dimension mapping |
| `src/profirag/embedding/fastembed_embedding.py` | Create | FastEmbedEmbedding class implementation |
| `src/profirag/embedding/__init__.py` | Modify | Export FastEmbedEmbedding |
| `src/profirag/pipeline/rag_pipeline.py` | Modify | Factory method for embedding selection |
| `tests/embedding/__init__.py` | Create | Test package init |
| `tests/embedding/test_fastembed.py` | Create | Unit tests for FastEmbedEmbedding |
| `tests/integration/test_fastembed.py` | Create | Integration tests with pipeline |

---

### Task 1: Add fastembed dependency

**Files:**
- Modify: `pyproject.toml:13-35`

- [ ] **Step 1: Add fastembed to dependencies**

Edit `pyproject.toml` to add `fastembed>=0.2.0` to the dependencies list:

```toml
dependencies = [
    "llama-index-core>=0.10.0",
    "llama-index-embeddings-openai>=0.1.0",
    "llama-index-llms-openai>=0.1.0",
    "llama-index-vector-stores-qdrant>=0.1.0",
    "llama-index-vector-stores-postgres>=0.1.0",
    "qdrant-client>=1.7.0",
    "psycopg2-binary>=2.9.0",
    "pgvector>=0.2.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "rank-bm25>=0.2.2",
    "jieba>=0.42.1",
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0,<2.10.0",
    "pymupdf4llm>=0.0.5",
    "pandas>=2.0.0",
    "tree-sitter>=0.20.0",
    "tree-sitter-python>=0.20.0",
    "tree-sitter-java>=0.20.0",
    "tree-sitter-cpp>=0.20.0",
    "tree-sitter-go>=0.20.0",
    "fastembed>=0.2.0",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies resolved and fastembed installed

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add fastembed dependency for local embedding support"
```

---

### Task 2: Update .env.example with FastEmbed configuration

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add FastEmbed configuration section**

Edit `.env.example` to add FastEmbed configuration after the OpenAI Configuration section (around line 15):

```bash
# ==================== Embedding Provider Configuration ====================
# Provider: openai (API-based) or fastembed (local)
PROFIRAG_EMBEDDING_PROVIDER=openai

# FastEmbed Configuration (only used when PROFIRAG_EMBEDDING_PROVIDER=fastembed)
# Model name - dimension auto-detected for known models
PROFIRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# Optional: override dimension (defaults to auto-detected based on model)
PROFIRAG_EMBEDDING_DIMENSION=
# Optional: custom cache directory for model files (defaults to ~/.cache/fastembed)
PROFIRAG_EMBEDDING_CACHE_DIR=
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add FastEmbed configuration to .env.example"
```

---

### Task 3: Update EnvSettings and EmbeddingConfig in settings.py

**Files:**
- Modify: `src/profirag/config/settings.py`

- [ ] **Step 1: Add FASTEMBED_MODEL_DIMENSIONS mapping**

Add the dimension mapping constant at the top of `src/profirag/config/settings.py` after the imports (around line 11):

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata


# FastEmbed model dimension mapping for auto-detection
FASTEMBED_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
```

- [ ] **Step 2: Add new EnvSettings fields**

Edit `EnvSettings` class to add new FastEmbed-related fields after `openai_llm_max_tokens` (around line 31):

```python
    openai_llm_max_tokens: Optional[int] = None

    # Embedding Provider Configuration
    profirag_embedding_provider: Literal["openai", "fastembed"] = "openai"
    profirag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    profirag_embedding_dimension: Optional[int] = None  # Auto-detected for FastEmbed
    profirag_embedding_cache_dir: Optional[str] = None
```

- [ ] **Step 3: Update EmbeddingConfig class**

Edit `EmbeddingConfig` class to support both providers (around line 141):

```python
class EmbeddingConfig(BaseModel):
    """Embedding configuration supporting OpenAI and FastEmbed providers"""
    provider: Literal["openai", "fastembed"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cache_dir: Optional[str] = None  # For FastEmbed model cache
```

- [ ] **Step 4: Update RAGConfig.from_env() embedding creation**

Edit `RAGConfig.from_env()` method to handle provider selection (around line 303-311):

```python
        # Build embedding config based on provider
        if env_settings.profirag_embedding_provider == "fastembed":
            model = env_settings.profirag_embedding_model
            dimension = env_settings.profirag_embedding_dimension or FASTEMBED_MODEL_DIMENSIONS.get(model, 768)
        else:
            model = env_settings.openai_embedding_model
            dimension = env_settings.openai_embedding_dimension

        return cls(
            storage=StorageConfig(type=storage_type, config=storage_config),
            embedding=EmbeddingConfig(
                provider=env_settings.profirag_embedding_provider,
                model=model,
                dimension=dimension,
                api_key=env_settings.openai_embedding_api_key or env_settings.openai_api_key if env_settings.profirag_embedding_provider == "openai" else None,
                base_url=env_settings.openai_embedding_base_url or env_settings.openai_base_url if env_settings.profirag_embedding_provider == "openai" else None,
                cache_dir=env_settings.profirag_embedding_cache_dir,
            ),
```

- [ ] **Step 5: Run existing tests to verify no breaking changes**

Run: `pytest tests/config/ -v`
Expected: All existing config tests pass

- [ ] **Step 6: Commit**

```bash
git add src/profirag/config/settings.py
git commit -m "feat: add FastEmbed provider configuration to settings"
```

---

### Task 4: Create FastEmbedEmbedding class

**Files:**
- Create: `src/profirag/embedding/fastembed_embedding.py`

- [ ] **Step 1: Write the FastEmbedEmbedding class**

Create `src/profirag/embedding/fastembed_embedding.py`:

```python
"""FastEmbed embedding model for local vectorization"""

import asyncio
import logging
import warnings
from typing import List, Any, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class FastEmbedEmbedding(BaseEmbedding):
    """Embedding model using FastEmbed for local vectorization.

    FastEmbed provides fast, local embedding generation without API calls.
    Models are cached locally after first download.

    Args:
        model: Model name (e.g., "BAAI/bge-small-en-v1.5")
        dimension: Embedding dimension (model-specific, auto-detected for known models)
        cache_dir: Optional directory for caching downloaded models
    """

    model: str
    dimension: int
    cache_dir: Optional[str] = None
    _model: Optional[Any] = None  # TextEmbedding instance

    def __init__(
        self,
        model: str,
        dimension: int,
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize FastEmbed embedding model.

        Args:
            model: FastEmbed model name
            dimension: Expected embedding dimension
            cache_dir: Optional cache directory for model files
            **kwargs: Additional arguments passed to BaseEmbedding
        """
        super().__init__(
            model=model,
            dimension=dimension,
            cache_dir=cache_dir,
            **kwargs
        )
        self._model = None

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def _load_model(self) -> Any:
        """Lazy load the FastEmbed model.

        Returns:
            TextEmbedding instance

        Raises:
            ImportError: If fastembed package not installed
            ValueError: If model name is invalid
            RuntimeError: If model fails to load
        """
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed package not installed. Run: uv add fastembed"
            )

        try:
            self._model = TextEmbedding(
                model_name=self.model,
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded FastEmbed model: {self.model}")
            return self._model
        except ValueError as e:
            # Get list of supported models for error message
            try:
                supported = TextEmbedding.list_supported_models()
                model_names = [m.get("model", "") for m in supported]
            except Exception:
                model_names = []
            raise ValueError(
                f"Invalid FastEmbed model '{self.model}'. "
                f"Available models: {model_names}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FastEmbed model '{self.model}': {e}"
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Handle empty text
        if not text or not text.strip():
            return [0.0] * self.dimension

        model = self._load_model()
        text = text.replace("\n", " ")

        try:
            embeddings = list(model.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            raise RuntimeError(f"FastEmbed embedding failed: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Handle empty texts
        processed_texts = []
        zero_vector_indices = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                zero_vector_indices.append(i)
                processed_texts.append(" ")  # Placeholder for empty
            else:
                processed_texts.append(text.replace("\n", " "))

        model = self._load_model()

        try:
            embeddings_iter = model.embed(processed_texts)
            embeddings = [e.tolist() for e in embeddings_iter]

            # Replace empty text embeddings with zero vectors
            for idx in zero_vector_indices:
                embeddings[idx] = [0.0] * self.dimension

            return embeddings
        except Exception as e:
            raise RuntimeError(f"FastEmbed batch embedding failed: {e}")

    # Required BaseEmbedding method implementations
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return await asyncio.to_thread(self._get_embedding, query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return await asyncio.to_thread(self._get_embedding, text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings in batch."""
        return self._get_embeddings(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings asynchronously in batch."""
        return await asyncio.to_thread(self._get_embeddings, texts)
```

- [ ] **Step 2: Commit**

```bash
git add src/profirag/embedding/fastembed_embedding.py
git commit -m "feat: add FastEmbedEmbedding class for local vectorization"
```

---

### Task 5: Update embedding __init__.py exports

**Files:**
- Modify: `src/profirag/embedding/__init__.py`

- [ ] **Step 1: Add FastEmbedEmbedding export**

Edit `src/profirag/embedding/__init__.py`:

```python
"""Embedding models for RAG pipeline"""

from .custom_embedding import CustomOpenAIEmbedding
from .fastembed_embedding import FastEmbedEmbedding

__all__ = ["CustomOpenAIEmbedding", "FastEmbedEmbedding"]
```

- [ ] **Step 2: Commit**

```bash
git add src/profirag/embedding/__init__.py
git commit -m "feat: export FastEmbedEmbedding from embedding module"
```

---

### Task 6: Update RAGPipeline._create_embed_model() factory

**Files:**
- Modify: `src/profirag/pipeline/rag_pipeline.py:134-144`

- [ ] **Step 1: Add import for FastEmbedEmbedding**

Edit imports at the top of `src/profirag/pipeline/rag_pipeline.py` (around line 11):

```python
from ..config.settings import RAGConfig, CustomOpenAILLM
from ..embedding import CustomOpenAIEmbedding, FastEmbedEmbedding
```

- [ ] **Step 2: Update _create_embed_model method**

Edit `_create_embed_model` method (around line 134):

```python
    def _create_embed_model(self) -> BaseEmbedding:
        """Create embedding model based on provider configuration."""
        if self.config.embedding.provider == "fastembed":
            return FastEmbedEmbedding(
                model=self.config.embedding.model,
                dimension=self.config.embedding.dimension,
                cache_dir=self.config.embedding.cache_dir,
            )
        else:  # openai
            embed_kwargs = {
                "model": self.config.embedding.model,
                "api_key": self.config.embedding.api_key,
            }
            if self.config.embedding.dimension:
                embed_kwargs["dimensions"] = self.config.embedding.dimension
            if self.config.embedding.base_url:
                embed_kwargs["api_base"] = self.config.embedding.base_url
            return CustomOpenAIEmbedding(**embed_kwargs)
```

- [ ] **Step 3: Run existing pipeline tests**

Run: `pytest tests/pipeline/ -v`
Expected: Existing tests pass (OpenAI provider is default)

- [ ] **Step 4: Commit**

```bash
git add src/profirag/pipeline/rag_pipeline.py
git commit -m "feat: add FastEmbed provider selection in RAGPipeline factory"
```

---

### Task 7: Create unit tests for FastEmbedEmbedding

**Files:**
- Create: `tests/embedding/__init__.py`
- Create: `tests/embedding/test_fastembed.py`

- [ ] **Step 1: Create tests/embedding/__init__.py**

```python
"""Tests for embedding models"""
```

- [ ] **Step 2: Write unit tests**

Create `tests/embedding/test_fastembed.py`:

```python
"""Unit tests for FastEmbedEmbedding class"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from profirag.embedding import FastEmbedEmbedding
from profirag.config.settings import FASTEMBED_MODEL_DIMENSIONS


class TestFastEmbedEmbeddingInit:
    """Tests for FastEmbedEmbedding initialization"""

    def test_init_with_valid_model(self):
        """Test initialization with valid model name"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        assert embedding.model == "BAAI/bge-small-en-v1.5"
        assert embedding.dimension == 384
        assert embedding._model is None  # Lazy loaded

    def test_init_with_cache_dir(self):
        """Test initialization with custom cache directory"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
            cache_dir="/tmp/fastembed_cache",
        )
        assert embedding.cache_dir == "/tmp/fastembed_cache"

    def test_class_name(self):
        """Test class_name method returns correct name"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        assert embedding.class_name() == "FastEmbedEmbedding"


class TestFastEmbedEmbeddingLoadModel:
    """Tests for model loading"""

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_load_model_success(self, mock_text_embedding):
        """Test successful model loading"""
        mock_instance = Mock()
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        model = embedding._load_model()

        mock_text_embedding.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=None
        )
        assert model == mock_instance

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_load_model_cached(self, mock_text_embedding):
        """Test that model is cached after first load"""
        mock_instance = Mock()
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        # First load
        embedding._load_model()
        # Second load (should use cached)
        embedding._load_model()

        # Should only create once
        mock_text_embedding.assert_called_once()

    def test_load_model_import_error(self):
        """Test error when fastembed not installed"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with patch.dict('sys.modules', {'fastembed': None}):
            with pytest.raises(ImportError) as exc_info:
                embedding._load_model()

            assert "fastembed package not installed" in str(exc_info.value)

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_load_model_invalid_model(self, mock_text_embedding):
        """Test error for invalid model name"""
        mock_text_embedding.side_effect = ValueError("Invalid model")
        mock_text_embedding.list_supported_models = Mock(return_value=[
            {"model": "BAAI/bge-small-en-v1.5"},
            {"model": "BAAI/bge-base-en-v1.5"},
        ])

        embedding = FastEmbedEmbedding(
            model="invalid-model-name",
            dimension=384,
        )

        with pytest.raises(ValueError) as exc_info:
            embedding._load_model()

        assert "Invalid FastEmbed model" in str(exc_info.value)
        assert "invalid-model-name" in str(exc_info.value)

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_load_model_runtime_error(self, mock_text_embedding):
        """Test error for unexpected model loading failure"""
        mock_text_embedding.side_effect = RuntimeError("Network error")

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with pytest.raises(RuntimeError) as exc_info:
            embedding._load_model()

        assert "Failed to load FastEmbed model" in str(exc_info.value)


class TestFastEmbedEmbeddingMethods:
    """Tests for embedding methods"""

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_embedding_single(self, mock_text_embedding):
        """Test single text embedding"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embedding("test text")

        assert len(result) == 384
        assert result == [0.1] * 384

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_embedding_empty_text(self, mock_text_embedding):
        """Test embedding for empty text returns zero vector"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embedding("")

        assert len(result) == 384
        assert result == [0.0] * 384
        # Should not call model.embed for empty text
        mock_text_embedding.assert_not_called()

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_embeddings_batch(self, mock_text_embedding):
        """Test batch embedding for multiple texts"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embeddings(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384
        assert result[0] == [0.1] * 384
        assert result[1] == [0.2] * 384

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_embeddings_with_empty_texts(self, mock_text_embedding):
        """Test batch embedding handles empty texts with zero vectors"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384  # Placeholder
        mock_embeddings[2].tolist.return_value = [0.3] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embeddings(["text one", "", "text three"])

        assert len(result) == 3
        # Empty text should have zero vector
        assert result[1] == [0.0] * 384

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_query_embedding(self, mock_text_embedding):
        """Test _get_query_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_query_embedding("query text")

        assert len(result) == 384

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_text_embedding(self, mock_text_embedding):
        """Test _get_text_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_text_embedding("document text")

        assert len(result) == 384

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_get_text_embeddings(self, mock_text_embedding):
        """Test _get_text_embeddings batch method"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_text_embeddings(["text 1", "text 2"])

        assert len(result) == 2
        assert len(result[0]) == 384


class TestFastEmbedEmbeddingAsync:
    """Tests for async embedding methods"""

    @pytest.mark.asyncio
    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    async def test_aget_query_embedding(self, mock_text_embedding):
        """Test async _aget_query_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_query_embedding("query text")

        assert len(result) == 384

    @pytest.mark.asyncio
    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    async def test_aget_text_embedding(self, mock_text_embedding):
        """Test async _aget_text_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_text_embedding("document text")

        assert len(result) == 384

    @pytest.mark.asyncio
    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    async def test_aget_text_embeddings(self, mock_text_embedding):
        """Test async _aget_text_embeddings batch method"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_text_embeddings(["text 1", "text 2"])

        assert len(result) == 2
        assert len(result[0]) == 384


class TestFastEmbedEmbeddingErrorHandling:
    """Tests for error handling"""

    @patch('profirag.embedding.fastembed_embedding.TextEmbedding')
    def test_embedding_runtime_error(self, mock_text_embedding):
        """Test error handling when embedding fails"""
        mock_instance = Mock()
        mock_instance.embed.side_effect = RuntimeError("GPU memory error")
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with pytest.raises(RuntimeError) as exc_info:
            embedding._get_embedding("test text")

        assert "FastEmbed embedding failed" in str(exc_info.value)
```

- [ ] **Step 3: Run tests to verify they work with mocks**

Run: `pytest tests/embedding/test_fastembed.py -v`
Expected: All tests pass (using mocks, no actual model download)

- [ ] **Step 4: Commit**

```bash
git add tests/embedding/__init__.py tests/embedding/test_fastembed.py
git commit -m "test: add unit tests for FastEmbedEmbedding"
```

---

### Task 8: Create integration tests

**Files:**
- Create: `tests/integration/test_fastembed.py`

- [ ] **Step 1: Write integration tests**

Create `tests/integration/test_fastembed.py`:

```python
"""Integration tests for FastEmbed with RAGPipeline"""

import pytest
import os
from unittest.mock import patch, Mock

from profirag.config.settings import RAGConfig, EmbeddingConfig, StorageConfig
from profirag.embedding import FastEmbedEmbedding
from profirag.pipeline.rag_pipeline import RAGPipeline


# Skip tests if fastembed not installed or if network unavailable
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_FASTEMBED_TESTS", "true").lower() == "true",
    reason="FastEmbed integration tests skipped (set SKIP_FASTEMBED_TESTS=false to run)"
)


class TestFastEmbedPipelineIntegration:
    """Integration tests for FastEmbed with full pipeline"""

    @pytest.fixture
    def fastembed_config(self):
        """Create RAGConfig with FastEmbed provider"""
        return RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_fastembed_storage",
                    "collection_name": "test_fastembed",
                    "dimension": 384,
                }
            ),
            embedding=EmbeddingConfig(
                provider="fastembed",
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            ),
        )

    def test_pipeline_creates_fastembed_embedding(self, fastembed_config):
        """Test that pipeline creates FastEmbedEmbedding when provider is fastembed"""
        pipeline = RAGPipeline(fastembed_config)
        assert isinstance(pipeline._embed_model, FastEmbedEmbedding)
        assert pipeline._embed_model.model == "BAAI/bge-small-en-v1.5"
        assert pipeline._embed_model.dimension == 384

    def test_pipeline_embed_model_dimension(self, fastembed_config):
        """Test that embedding dimension is correct"""
        pipeline = RAGPipeline(fastembed_config)
        # Dimension should match config
        assert pipeline._embed_model.dimension == 384

    def test_pipeline_get_stats_shows_fastembed(self, fastembed_config):
        """Test that pipeline stats show FastEmbed as embedding provider"""
        pipeline = RAGPipeline(fastembed_config)
        stats = pipeline.get_stats()

        assert stats["embedding"]["model"] == "BAAI/bge-small-en-v1.5"
        assert stats["embedding"]["dimension"] == 384


class TestFastEmbedProviderSelection:
    """Tests for provider selection logic"""

    def test_openai_provider_creates_custom_embedding(self):
        """Test that openai provider creates CustomOpenAIEmbedding"""
        from profirag.embedding import CustomOpenAIEmbedding

        config = RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_openai_storage",
                    "collection_name": "test_openai",
                    "dimension": 1536,
                }
            ),
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                api_key="test-key",
            ),
        )
        pipeline = RAGPipeline(config)
        assert isinstance(pipeline._embed_model, CustomOpenAIEmbedding)

    def test_fastembed_provider_creates_fastembed_embedding(self):
        """Test that fastembed provider creates FastEmbedEmbedding"""
        config = RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_fastembed_storage2",
                    "collection_name": "test_fastembed2",
                    "dimension": 384,
                }
            ),
            embedding=EmbeddingConfig(
                provider="fastembed",
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            ),
        )
        pipeline = RAGPipeline(config)
        assert isinstance(pipeline._embed_model, FastEmbedEmbedding)


class TestFastEmbedConfigFromEnv:
    """Tests for loading FastEmbed config from environment"""

    @patch.dict(os.environ, {
        "PROFIRAG_STORAGE_TYPE": "local",
        "LOCAL_STORAGE_PATH": "/tmp/test_env_storage",
        "LOCAL_COLLECTION_NAME": "test_env",
        "PROFIRAG_EMBEDDING_PROVIDER": "fastembed",
        "PROFIRAG_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
    }, clear=False)
    def test_config_from_env_fastembed(self):
        """Test loading FastEmbed config from environment variables"""
        config = RAGConfig.from_env()

        assert config.embedding.provider == "fastembed"
        assert config.embedding.model == "BAAI/bge-base-en-v1.5"
        # Dimension should be auto-detected (768 for bge-base)
        assert config.embedding.dimension == 768

    @patch.dict(os.environ, {
        "PROFIRAG_EMBEDDING_PROVIDER": "fastembed",
        "PROFIRAG_EMBEDDING_MODEL": "unknown-model",
        "PROFIRAG_EMBEDDING_DIMENSION": "512",
    }, clear=False)
    def test_config_dimension_override(self):
        """Test dimension override for unknown models"""
        config = RAGConfig.from_env()

        assert config.embedding.dimension == 512

    @patch.dict(os.environ, {
        "PROFIRAG_EMBEDDING_PROVIDER": "openai",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_EMBEDDING_DIMENSION": "1536",
    }, clear=False)
    def test_config_openai_provider_default(self):
        """Test that openai remains default provider"""
        config = RAGConfig.from_env()

        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.embedding.dimension == 1536
```

- [ ] **Step 2: Commit**

```bash
git add tests/integration/test_fastembed.py
git commit -m "test: add integration tests for FastEmbed pipeline"
```

---

### Task 9: Final verification and documentation

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (integration tests skipped by default)

- [ ] **Step 2: Verify import works**

Run: `python -c "from profirag.embedding import FastEmbedEmbedding; print(FastEmbedEmbedding.__name__)"`
Expected: Output "FastEmbedEmbedding"

- [ ] **Step 3: Final commit with summary**

```bash
git add -A
git status
git commit -m "feat: complete FastEmbed local embedding support

- Add FastEmbedEmbedding class wrapping fastembed.TextEmbedding
- Add provider enum configuration (openai | fastembed)
- Auto-detect embedding dimension for known models
- Lazy model loading with caching
- Async methods via asyncio.to_thread
- Unit and integration tests

Users can now use local embeddings by setting:
PROFIRAG_EMBEDDING_PROVIDER=fastembed
PROFIRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5"
```

---

## Self-Review Checklist

**1. Spec Coverage:**
- [x] FastEmbedEmbedding class - Task 4
- [x] Provider enum - Task 3
- [x] Config auto-detection - Task 3
- [x] Factory pattern - Task 6
- [x] Async wrappers - Task 4
- [x] Error handling - Task 4
- [x] Model dimension mapping - Task 3
- [x] Unit tests - Task 7
- [x] Integration tests - Task 8
- [x] .env.example - Task 2
- [x] pyproject.toml - Task 1

**2. Placeholder Scan:**
- No TBD, TODO, or "implement later" phrases
- All code blocks contain complete code
- No references to undefined functions

**3. Type Consistency:**
- FastEmbedEmbedding.model: str (Task 4, matches Task 3 EmbeddingConfig.model)
- FastEmbedEmbedding.dimension: int (Task 4, matches Task 3)
- FastEmbedEmbedding.cache_dir: Optional[str] (Task 4, matches Task 3)