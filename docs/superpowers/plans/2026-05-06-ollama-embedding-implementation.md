# Ollama Embedding Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as a third embedding provider with auto-configured defaults for local development.

**Architecture:** Thin wrapper around CustomOpenAIEmbedding - adds `ollama` provider type, auto-defaults (localhost:11434/v1, nomic-embed-text, 768 dim), empty API key.

**Tech Stack:** Python 3.10, Pydantic, OpenAI client (for Ollama's OpenAI-compatible endpoint)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tests/config/test_ollama_config.py` | Create | Unit tests for Ollama config creation |
| `src/profirag/config/settings.py` | Modify | Add ollama Literal, ollama settings, from_env logic |
| `src/profirag/pipeline/rag_pipeline.py` | Modify | Add ollama branch in _create_embed_model |
| `.env.example` | Modify | Document Ollama configuration options |
| `CLAUDE.md` | Modify | Update embedding providers section |

---

### Task 1: Ollama Config Tests

**Files:**
- Create: `tests/config/test_ollama_config.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for Ollama embedding provider configuration"""

import pytest
from pathlib import Path
import tempfile

from profirag.config.settings import RAGConfig, EnvSettings, EmbeddingConfig


class TestOllamaEnvSettings:
    """Tests for Ollama settings in EnvSettings"""

    def test_ollama_default_model(self):
        """Test default Ollama model is nomic-embed-text"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_embedding_model == "nomic-embed-text"

    def test_ollama_default_dimension(self):
        """Test default Ollama dimension is 768"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_embedding_dimension == 768

    def test_ollama_default_base_url(self):
        """Test default Ollama base URL"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_base_url == "http://localhost:11434/v1"

    def test_ollama_custom_model(self):
        """Test custom Ollama model override"""
        env = EnvSettings(
            profirag_embedding_provider="ollama",
            ollama_embedding_model="mxbai-embed-large"
        )
        assert env.ollama_embedding_model == "mxbai-embed-large"

    def test_ollama_custom_dimension(self):
        """Test custom Ollama dimension override"""
        env = EnvSettings(
            profirag_embedding_provider="ollama",
            ollama_embedding_dimension=1024
        )
        assert env.ollama_embedding_dimension == 1024

    def test_ollama_custom_base_url(self):
        """Test custom Ollama base URL override"""
        env = EnvSettings(
            profirag_embedding_provider="ollama",
            ollama_base_url="http://192.168.1.100:11434/v1"
        )
        assert env.ollama_base_url == "http://192.168.1.100:11434/v1"


class TestOllamaEmbeddingConfig:
    """Tests for EmbeddingConfig with Ollama provider"""

    def test_embedding_config_ollama_provider(self):
        """Test EmbeddingConfig accepts ollama as provider"""
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434/v1"
        )
        assert config.provider == "ollama"
        assert config.model == "nomic-embed-text"
        assert config.dimension == 768
        assert config.base_url == "http://localhost:11434/v1"


class TestOllamaRAGConfigFromEnv:
    """Tests for RAGConfig.from_env() with Ollama provider"""

    def test_from_env_ollama_defaults(self):
        """Test from_env creates correct config with Ollama defaults"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "nomic-embed-text"
        assert config.embedding.dimension == 768
        assert config.embedding.base_url == "http://localhost:11434/v1"
        assert config.embedding.api_key is None

    def test_from_env_ollama_custom_model(self):
        """Test from_env with custom Ollama model"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("OLLAMA_EMBEDDING_MODEL=mxbai-embed-large\n")
            f.write("OLLAMA_EMBEDDING_DIMENSION=1024\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.model == "mxbai-embed-large"
        assert config.embedding.dimension == 1024

    def test_from_env_ollama_custom_base_url(self):
        """Test from_env with custom Ollama base URL"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("OLLAMA_BASE_URL=http://remote-server:11434/v1\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.base_url == "http://remote-server:11434/v1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_ollama_config.py -v`
Expected: FAIL with errors about missing ollama settings and Literal type

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/config/test_ollama_config.py
git commit -m "test: add failing tests for Ollama embedding provider config

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Add Ollama Settings to EnvSettings

**Files:**
- Modify: `src/profirag/config/settings.py:44-47`

- [ ] **Step 1: Update EnvSettings provider Literal and add Ollama settings**

Find line 44 in `settings.py`:
```python
    profirag_embedding_provider: Literal["openai", "fastembed"] = "openai"
```

Replace with:
```python
    profirag_embedding_provider: Literal["openai", "fastembed", "ollama"] = "openai"

    # Ollama Configuration (only used when PROFIRAG_EMBEDDING_PROVIDER=ollama)
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_embedding_dimension: int = 768
    ollama_base_url: str = "http://localhost:11434/v1"
```

- [ ] **Step 2: Run tests to verify EnvSettings tests pass**

Run: `uv run pytest tests/config/test_ollama_config.py::TestOllamaEnvSettings -v`
Expected: PASS (6 tests)

- [ ] **Step 3: Commit**

```bash
git add src/profirag/config/settings.py
git commit -m "feat: add Ollama embedding settings to EnvSettings

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Update EmbeddingConfig Provider Literal

**Files:**
- Modify: `src/profirag/config/settings.py:159`

- [ ] **Step 1: Update EmbeddingConfig provider Literal**

Find line 159 in `settings.py`:
```python
    provider: Literal["openai", "fastembed"] = "openai"
```

Replace with:
```python
    provider: Literal["openai", "fastembed", "ollama"] = "openai"
```

- [ ] **Step 2: Run tests to verify EmbeddingConfig tests pass**

Run: `uv run pytest tests/config/test_ollama_config.py::TestOllamaEmbeddingConfig -v`
Expected: PASS (1 test)

- [ ] **Step 3: Commit**

```bash
git add src/profirag/config/settings.py
git commit -m "feat: add ollama to EmbeddingConfig provider Literal

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update RAGConfig.from_env() for Ollama

**Files:**
- Modify: `src/profirag/config/settings.py:320-336`

- [ ] **Step 1: Add Ollama branch in from_env() embedding config building**

Find lines 320-336 in `settings.py`:
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

Replace with:
```python
        # Build embedding config based on provider
        if env_settings.profirag_embedding_provider == "fastembed":
            model = env_settings.profirag_embedding_model
            dimension = env_settings.profirag_embedding_dimension or FASTEMBED_MODEL_DIMENSIONS.get(model, 768)
            api_key = None
            base_url = None
        elif env_settings.profirag_embedding_provider == "ollama":
            model = env_settings.ollama_embedding_model
            dimension = env_settings.ollama_embedding_dimension
            api_key = None  # Ollama doesn't require authentication
            base_url = env_settings.ollama_base_url
        else:  # openai
            model = env_settings.openai_embedding_model
            dimension = env_settings.openai_embedding_dimension
            api_key = env_settings.openai_embedding_api_key or env_settings.openai_api_key
            base_url = env_settings.openai_embedding_base_url or env_settings.openai_base_url

        return cls(
            storage=StorageConfig(type=storage_type, config=storage_config),
            embedding=EmbeddingConfig(
                provider=env_settings.profirag_embedding_provider,
                model=model,
                dimension=dimension,
                api_key=api_key,
                base_url=base_url,
                cache_dir=env_settings.profirag_embedding_cache_dir,
            ),
```

- [ ] **Step 2: Run tests to verify RAGConfig.from_env tests pass**

Run: `uv run pytest tests/config/test_ollama_config.py::TestOllamaRAGConfigFromEnv -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Run all config tests to verify no regressions**

Run: `uv run pytest tests/config -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add src/profirag/config/settings.py
git commit -m "feat: add Ollama embedding config logic in RAGConfig.from_env

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add Ollama Branch in Pipeline

**Files:**
- Modify: `src/profirag/pipeline/rag_pipeline.py:135-152`

- [ ] **Step 1: Add ollama branch in _create_embed_model()**

Find lines 135-152 in `rag_pipeline.py`:
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

Replace with:
```python
    def _create_embed_model(self) -> BaseEmbedding:
        """Create embedding model based on provider configuration."""
        if self.config.embedding.provider == "fastembed":
            return FastEmbedEmbedding(
                model=self.config.embedding.model,
                dimension=self.config.embedding.dimension,
                cache_dir=self.config.embedding.cache_dir,
            )
        elif self.config.embedding.provider == "ollama":
            return CustomOpenAIEmbedding(
                model=self.config.embedding.model,
                api_key="",  # Ollama doesn't require authentication
                api_base=self.config.embedding.base_url,
                dimensions=self.config.embedding.dimension,
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

- [ ] **Step 2: Run all tests to verify no regressions**

Run: `uv run pytest tests -v`
Expected: PASS (all tests)

- [ ] **Step 3: Commit**

```bash
git add src/profirag/pipeline/rag_pipeline.py
git commit -m "feat: add Ollama embedding branch in RAGPipeline

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Update .env.example

**Files:**
- Modify: `.env.example:16-26`

- [ ] **Step 1: Add Ollama configuration section**

Find lines 16-26 in `.env.example`:
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

Replace with:
```bash
# ==================== Embedding Provider Configuration ====================
# Provider: openai (API-based), fastembed (local), or ollama (local via Ollama)
PROFIRAG_EMBEDDING_PROVIDER=openai

# FastEmbed Configuration (only used when PROFIRAG_EMBEDDING_PROVIDER=fastembed)
# Model name - dimension auto-detected for known models
PROFIRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# Optional: override dimension (defaults to auto-detected based on model)
PROFIRAG_EMBEDDING_DIMENSION=
# Optional: custom cache directory for model files (defaults to ~/.cache/fastembed)
PROFIRAG_EMBEDDING_CACHE_DIR=

# Ollama Configuration (only used when PROFIRAG_EMBEDDING_PROVIDER=ollama)
# Ensure Ollama is running: ollama serve
# Pull embedding model: ollama pull nomic-embed-text
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
OLLAMA_BASE_URL=http://localhost:11434/v1
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add Ollama embedding configuration to .env.example

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md:38-41`

- [ ] **Step 1: Update embedding providers section**

Find lines 38-41 in `CLAUDE.md`:
```markdown
### Embedding Providers

Two embedding providers are supported:
- **OpenAI**: API-based embedding (default)
- **FastEmbed**: Local embedding via `FastEmbedEmbedding` class

Provider selection via `PROFIRAG_EMBEDDING_PROVIDER` env var.
```

Replace with:
```markdown
### Embedding Providers

Three embedding providers are supported:
- **OpenAI**: API-based embedding (default)
- **FastEmbed**: Local embedding via `FastEmbedEmbedding` class
- **Ollama**: Local embedding via Ollama's OpenAI-compatible API endpoint

Provider selection via `PROFIRAG_EMBEDDING_PROVIDER` env var (openai, fastembed, ollama).

For Ollama:
- Run `ollama serve` to start Ollama server
- Run `ollama pull nomic-embed-text` to download embedding model
- Set `PROFIRAG_EMBEDDING_PROVIDER=ollama` in `.env`
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with Ollama embedding provider info

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests -v`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `uv run ruff check src tests --fix`
Expected: No errors or warnings

- [ ] **Step 3: Format code**

Run: `uv run ruff format src tests`
Expected: Code formatted successfully

- [ ] **Step 4: Verify git status**

Run: `git status`
Expected: Clean working tree (all changes committed)

- [ ] **Step 5: Review commit history**

Run: `git log --oneline -10`
Expected: 7 commits for this feature (test, 3 settings changes, pipeline, .env.example, CLAUDE.md)