"""Re-ranking component for post-retrieval processing"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import httpx
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field, PrivateAttr


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


class CrossEncoderReranker(BaseNodePostprocessor):
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
                   Default: ms-marco-MiniLM-L-6-v2 (fast and effective)
                   Alternatives: ms-marco-MiniLM-L-12-v2 (more accurate)
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

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes based on cross-encoder scores.

        Args:
            nodes: List of NodeWithScore objects to rerank
            query_bundle: QueryBundle containing the query string

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not nodes or not query_bundle:
            return nodes

        query_str = query_bundle.query_str
        self._load_model()

        # Prepare query-document pairs
        pairs = [(query_str, node.node.text) for node in nodes]

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


class Reranker:
    """Flexible reranker supporting multiple reranking strategies.

    Provides a unified interface for different reranking backends.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        enabled: bool = True,
        **kwargs
    ):
        """Initialize reranker.

        Args:
            model: Reranker model name
            top_n: Number of results to return
            enabled: Whether reranking is enabled
            **kwargs: Additional arguments for the reranker
        """
        self.model = model
        self.top_n = top_n
        self.enabled = enabled
        self.kwargs = kwargs
        self._reranker: Optional[CrossEncoderReranker] = None

        if enabled:
            self._reranker = CrossEncoderReranker(
                model=model,
                top_n=top_n,
                **kwargs
            )

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
        if not self.enabled or not self._reranker:
            return nodes[:self.top_n]

        if not nodes:
            return nodes

        return self._reranker.postprocess_nodes(nodes, query_str=query, **kwargs)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable reranking.

        Args:
            enabled: Whether to enable reranking
        """
        self.enabled = enabled
        if enabled and self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model=self.model,
                top_n=self.top_n,
                **self.kwargs
            )

    def set_top_n(self, top_n: int) -> None:
        """Update number of results to return.

        Args:
            top_n: New top_n value
        """
        self.top_n = top_n
        if self._reranker:
            self._reranker.top_n = top_n


class LLMReranker:
    """LLM-based reranker for more sophisticated relevance scoring.

    Uses an LLM to evaluate query-document relevance.
    More accurate but slower than cross-encoder reranking.
    """

    def __init__(
        self,
        llm: Any,
        top_n: int = 5,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize LLM reranker.

        Args:
            llm: LLM instance
            top_n: Number of results to return
            prompt_template: Custom prompt template for scoring
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.top_n = top_n
        self.prompt_template = prompt_template or self._default_prompt()
        self.kwargs = kwargs

    def _default_prompt(self) -> str:
        """Default prompt template for relevance scoring."""
        return """Evaluate the relevance of the following document to the query.
Score from 0 to 10, where 0 is completely irrelevant and 10 is highly relevant.

Query: {query}

Document: {document}

Relevance score (0-10):"""

    def _score_document(self, query: str, document: str) -> float:
        """Get relevance score from LLM.

        Args:
            query: Query string
            document: Document text

        Returns:
            Relevance score (0-10)
        """
        prompt = self.prompt_template.format(query=query, document=document)
        response = self.llm.complete(prompt)

        # Parse score
        try:
            score_str = response.text.strip()
            # Extract first number found
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", score_str)
            if match:
                score = float(match.group(1))
                return min(max(score, 0), 10)  # Clamp to 0-10
        except (ValueError, AttributeError):
            pass

        return 0.0

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes using LLM scoring.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not nodes:
            return nodes

        scored_nodes = []
        for node in nodes:
            score = self._score_document(query, node.node.text)
            scored_nodes.append(NodeWithScore(node=node.node, score=score))

        # Sort by score
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        return scored_nodes[:self.top_n]