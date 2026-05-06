# Ollama Embedding Provider Support

**Date**: 2026-05-06
**Status**: Approved
**Approach**: Dedicated "ollama" provider type

## Overview

Add Ollama as a third embedding provider option for local development/testing. Ollama exposes an OpenAI-compatible `/v1/embeddings` endpoint, allowing reuse of existing `CustomOpenAIEmbedding` class with Ollama-specific defaults.

## Requirements

- Support `PROFIRAG_EMBEDDING_PROVIDER=ollama` configuration
- Auto-configure sensible defaults for local Ollama usage
- Primary use case: local development/testing with Ollama running on `localhost:11434`
- Default model: `nomic-embed-text` (768 dimensions)

## Architecture

The `ollama` provider is a convenience wrapper around `CustomOpenAIEmbedding`:

- Adds `ollama` to provider Literal types in config
- Sets auto-defaults: localhost URL, nomic-embed-text model, 768 dimensions, empty API key
- Reuses `CustomOpenAIEmbedding` internally (Ollama has OpenAI-compatible endpoint)
- No new embedding class needed

## Configuration Changes

### EnvSettings class (settings.py)

```python
# Update provider Literal
profirag_embedding_provider: Literal["openai", "fastembed", "ollama"] = "openai"

# Add Ollama-specific settings
ollama_embedding_model: str = "nomic-embed-text"
ollama_embedding_dimension: int = 768
ollama_base_url: str = "http://localhost:11434/v1"
```

### EmbeddingConfig class

```python
provider: Literal["openai", "fastembed", "ollama"] = "openai"
```

### RAGConfig.from_env() logic

When `profirag_embedding_provider == "ollama"`:
- `model` = `env.ollama_embedding_model` (default: nomic-embed-text)
- `dimension` = `env.ollama_embedding_dimension` (default: 768)
- `api_key` = `None` (Ollama doesn't require authentication)
- `base_url` = `env.ollama_base_url` (default: http://localhost:11434/v1)
- `cache_dir` = `None` (not used for Ollama)

### .env.example additions

```bash
# ==================== Embedding Provider Configuration ====================
# Provider: openai (API-based), fastembed (local), or ollama (local via Ollama)
PROFIRAG_EMBEDDING_PROVIDER=openai

# Ollama Configuration (only used when PROFIRAG_EMBEDDING_PROVIDER=ollama)
# Defaults: localhost:11434/v1, nomic-embed-text, 768 dimensions
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Pipeline Implementation

### RAGPipeline._create_embed_model()

Add branch for `ollama` provider:

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
            model=self.config.embedding.model,        # nomic-embed-text
            api_key="",                               # Empty, Ollama doesn't need auth
            api_base=self.config.embedding.base_url,  # localhost:11434/v1
            dimensions=self.config.embedding.dimension,  # 768
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

## Files Modified

1. `src/profirag/config/settings.py`
   - Add `ollama` to provider Literal in `EnvSettings`
   - Add `ollama_embedding_model`, `ollama_embedding_dimension`, `ollama_base_url` settings
   - Add `ollama` to provider Literal in `EmbeddingConfig`
   - Update `RAGConfig.from_env()` to handle `ollama` provider

2. `src/profirag/pipeline/rag_pipeline.py`
   - Add `ollama` branch in `_create_embed_model()`

3. `.env.example`
   - Add Ollama configuration section

4. `CLAUDE.md`
   - Update "Embedding Providers" section to include Ollama option

## Testing

### Unit tests (tests/config/test_settings.py or tests/embedding/)

- Test `from_env()` creates correct EmbeddingConfig when provider=ollama
- Test defaults applied correctly
- Test user overrides work (e.g., custom OLLAMA_BASE_URL)

### Integration test (optional)

- Would require running Ollama locally with `nomic-embed-text` pulled
- Test actual embedding generation through pipeline
- Skip by default in CI (no Ollama available)

## Error Handling

No special validation needed. Connection errors handled by OpenAI client:
- If Ollama not running: connection refused error
- If model not pulled: 404/error from Ollama API

Same behavior as other providers - user must ensure service is running.

## Usage Example

```bash
# .env
PROFIRAG_EMBEDDING_PROVIDER=ollama
# Optional overrides (defaults shown):
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_DIMENSION=768
OLLAMA_BASE_URL=http://localhost:11434/v1
```

```bash
# Ensure Ollama is running with model pulled
ollama pull nomic-embed-text
ollama serve  # or already running
```

```python
from profirag import RAGPipeline

pipeline = RAGPipeline.from_env()
# Uses Ollama embeddings automatically
```