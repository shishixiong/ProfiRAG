"""Re-ranking component for post-retrieval processing"""

from typing import List, Optional, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class CrossEncoderReranker(BaseNodePostprocessor):
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
        super().__init__(**kwargs)
        self.model_name = model
        self.top_n = top_n
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Load cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_str: Optional[str] = None,
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes based on cross-encoder scores.

        Args:
            nodes: List of NodeWithScore objects to rerank
            query_str: Query string for scoring
            **kwargs: Additional arguments

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not nodes or not query_str:
            return nodes

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