"""Re-ranking component for post-retrieval processing"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union
import httpx
from llama_index.core.schema import NodeWithScore, QueryBundle

from profirag.config.settings import RerankingConfig


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
        url = f"{self.base_url}/reranks"
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


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers.

    Uses a cross-encoder model to compute relevance scores for
    query-document pairs and reorder results.
    """

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
        self.model = model
        self.top_n = top_n
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs
        self._model: Any = None

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
            **kwargs: Additional arguments

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
        """Rerank nodes based on cross-encoder scores.

        Args:
            nodes: List of NodeWithScore objects to rerank
            query_bundle: QueryBundle containing the query string

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not query_bundle:
            return nodes

        return self.rerank(query_bundle.query_str, nodes)


class Reranker:
    """Flexible reranker supporting multiple reranking strategies.

    Factory class that creates the appropriate reranker implementation
    based on the provider configuration.
    """

    def __init__(
        self,
        config: RerankingConfig,
    ):
        """Initialize reranker from configuration.

        Args:
            config: RerankingConfig object specifying provider and settings

        Raises:
            ValueError: If API provider is selected but api_key is missing
        """
        self.config = config
        self.enabled = config.enabled
        self.top_n = config.top_n
        self.model = config.model
        self._impl: Optional[BaseReranker] = None

        if self.enabled:
            self._impl = self._create_impl(config)

    def _create_impl(self, config: RerankingConfig) -> BaseReranker:
        """Create the appropriate reranker implementation based on provider.

        Args:
            config: RerankingConfig object

        Returns:
            Appropriate BaseReranker implementation

        Raises:
            ValueError: If API provider is selected but api_key is missing
        """
        if config.provider == "local":
            return CrossEncoderReranker(
                model=config.model,
                top_n=config.top_n,
            )
        elif config.provider == "cohere":
            if not config.api_key:
                raise ValueError("api_key is required for Cohere reranker")
            if not config.base_url:
                raise ValueError("base_url is required for Cohere reranker")
            return CohereReranker(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                top_n=config.top_n,
                timeout=config.timeout,
            )
        elif config.provider == "dashscope":
            if not config.api_key:
                raise ValueError("api_key is required for DashScope reranker")
            if not config.base_url:
                raise ValueError("base_url is required for DashScope reranker")
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
            self._impl = self._create_impl(self.config)

    def set_top_n(self, top_n: int) -> None:
        """Update number of results to return.

        Args:
            top_n: New top_n value
        """
        self.top_n = top_n
        if self._impl:
            self._impl.top_n = top_n


