"""Custom embedding models for non-OpenAI providers"""

from typing import List, Any, Optional
from openai import OpenAI, AsyncOpenAI
from llama_index.core.base.embeddings.base import BaseEmbedding


class CustomOpenAIEmbedding(BaseEmbedding):
    """Embedding model for OpenAI-compatible APIs without model validation.

    This class bypasses llama_index's model name validation, allowing use
    of custom embedding providers like DashScope, MiniMax, etc.

    Args:
        model: Model name (e.g., "text-embedding-v4")
        api_key: API key for the embedding provider
        api_base: Base URL for the API endpoint
        dimensions: Embedding dimensions (optional)
        embed_batch_size: Batch size for embedding requests
    """

    model: str
    api_key: str
    api_base: Optional[str] = None
    dimensions: Optional[int] = None
    embed_batch_size: int = 10  # DashScope requires batch size <= 10
    _client: Optional[OpenAI] = None
    _aclient: Optional[AsyncOpenAI] = None

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        embed_batch_size: int = 10,  # DashScope requires batch size <= 10
        **kwargs: Any
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            dimensions=dimensions,
            embed_batch_size=embed_batch_size,
            **kwargs
        )
        self._client = None
        self._aclient = None

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

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        client = self._get_client()
        text = text.replace("\n", " ")

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(
            input=[text],
            model=self.model,
            **kwargs
        )
        return response.data[0].embedding

    async def _aget_embedding(self, text: str) -> List[float]:
        """Get embedding asynchronously."""
        client = self._get_aclient()
        text = text.replace("\n", " ")

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = await client.embeddings.create(
            input=[text],
            model=self.model,
            **kwargs
        )
        return response.data[0].embedding

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        client = self._get_client()
        texts = [text.replace("\n", " ") for text in texts]

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embeddings.create(
            input=texts,
            model=self.model,
            **kwargs
        )
        return [d.embedding for d in response.data]

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings asynchronously for multiple texts."""
        client = self._get_aclient()
        texts = [text.replace("\n", " ") for text in texts]

        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = await client.embeddings.create(
            input=texts,
            model=self.model,
            **kwargs
        )
        return [d.embedding for d in response.data]

    # Required BaseEmbedding method implementations
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously."""
        return await self._aget_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously."""
        return await self._aget_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings in batch."""
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            embeddings = self._get_embeddings(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings asynchronously in batch."""
        all_embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            embeddings = await self._aget_embeddings(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings