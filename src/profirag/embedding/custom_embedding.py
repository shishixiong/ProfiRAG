"""Custom embedding models for non-OpenAI providers"""

import logging
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Model-specific max context lengths (in characters)
# These are fallback defaults when OLLAMA_EMBEDDING_MAX_LENGTH is not set
# Users should configure OLLAMA_EMBEDDING_MAX_LENGTH based on their model
EMBEDDING_MAX_CONTEXT_LENGTHS = {
    "nomic-embed-text": 2048,  # ~512 tokens, conservative default
    "mxbai-embed-large": 2048,
    "all-minilm": 1024,  # ~256 tokens
}


class CustomOpenAIEmbedding(BaseEmbedding):
    """Embedding model for OpenAI-compatible APIs without model validation.

    This class bypasses llama_index's model name validation, allowing use
    of custom embedding providers like DashScope, MiniMax, Ollama, etc.

    Args:
        model: Model name (e.g., "text-embedding-v4", "nomic-embed-text")
        api_key: API key for the embedding provider
        api_base: Base URL for the API endpoint
        dimensions: Embedding dimensions (optional)
        embed_batch_size: Batch size for embedding requests
        max_length: Maximum text length in characters (optional, auto-detected for known models)
    """

    model: str
    api_key: str
    api_base: str | None = None
    dimensions: int | None = None
    embed_batch_size: int = 10  # DashScope requires batch size <= 10
    max_length: int | None = None  # Max text length in characters
    _client: OpenAI | None = None
    _aclient: AsyncOpenAI | None = None

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        dimensions: int | None = None,
        embed_batch_size: int = 10,  # DashScope requires batch size <= 10
        max_length: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            dimensions=dimensions,
            embed_batch_size=embed_batch_size,
            max_length=max_length,
            **kwargs,
        )
        self._client = None
        self._aclient = None

        # Auto-detect max_length for known models if not specified
        if self.max_length is None:
            # Check model name against known limits
            for known_model, length in EMBEDDING_MAX_CONTEXT_LENGTHS.items():
                if known_model in self.model:
                    self.max_length = length
                    logger.info(
                        f"Auto-detected max_length={length} for model '{self.model}'"
                    )
                    break

    def _truncate_text(self, text: str) -> str:
        """Truncate text if it exceeds max_length.

        Args:
            text: Text to potentially truncate

        Returns:
            Truncated text if needed, otherwise original text
        """
        if self.max_length is None:
            return text

        if len(text) > self.max_length:
            logger.warning(
                f"Truncating text from {len(text)} to {self.max_length} chars "
                f"for model '{self.model}'"
            )
            return text[:self.max_length]
        return text

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._aclient

    @classmethod
    def class_name(cls) -> str:
        return "CustomOpenAIEmbedding"

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        client = self._get_client()
        text = self._truncate_text(text.replace("\n", " "))

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(input=[text], model=self.model, **kwargs)
        return response.data[0].embedding

    async def _aget_embedding(self, text: str) -> list[float]:
        """Get embedding asynchronously."""
        client = self._get_aclient()
        text = self._truncate_text(text.replace("\n", " "))

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = await client.embeddings.create(input=[text], model=self.model, **kwargs)
        return response.data[0].embedding

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        client = self._get_client()
        texts = [self._truncate_text(text.replace("\n", " ")) for text in texts]

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(input=texts, model=self.model, **kwargs)
        return [d.embedding for d in response.data]

    async def _aget_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings asynchronously for multiple texts."""
        client = self._get_aclient()
        texts = [self._truncate_text(text.replace("\n", " ")) for text in texts]

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = await client.embeddings.create(input=texts, model=self.model, **kwargs)
        return [d.embedding for d in response.data]

    # Required BaseEmbedding method implementations
    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get query embedding asynchronously."""
        return await self._aget_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._get_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get text embedding asynchronously."""
        return await self._aget_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings in batch."""
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            embeddings = self._get_embeddings(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings asynchronously in batch."""
        all_embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            embeddings = await self._aget_embeddings(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings
