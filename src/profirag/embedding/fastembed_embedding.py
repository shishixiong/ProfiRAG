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