"""Hybrid retrieval with BM25 support"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from rank_bm25 import BM25Okapi
import jieba  # Chinese tokenizer


class BM25Index:
    """BM25 keyword search index.

    Provides BM25-based keyword retrieval for hybrid search.
    Supports Chinese text with jieba tokenization.
    """

    def __init__(
        self,
        tokenizer: str = "jieba",
        language: str = "zh",
        **kwargs
    ):
        """Initialize BM25 index.

        Args:
            tokenizer: Tokenizer type ("jieba" for Chinese, "space" for simple split)
            language: Language code for tokenization
            **kwargs: Additional arguments for BM25Okapi
        """
        self.tokenizer = tokenizer
        self.language = language
        self.bm25_kwargs = kwargs
        self.nodes: List[TextNode] = []
        self.node_id_to_idx: Dict[str, int] = {}
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on configured tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if self.tokenizer == "jieba":
            return list(jieba.cut(text))
        else:
            # Simple space-based tokenization
            return text.split()

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add nodes and rebuild BM25 index.

        Args:
            nodes: List of TextNode objects to add
        """
        start_idx = len(self.nodes)
        for i, node in enumerate(nodes):
            idx = start_idx + i
            self.nodes.append(node)
            self.node_id_to_idx[node.node_id] = idx

        # Rebuild BM25 index
        self._build_index()

    def _build_index(self) -> None:
        """Build BM25 index from current nodes."""
        if not self.nodes:
            self._bm25 = None
            return

        corpus = [self._tokenize(node.text) for node in self.nodes]
        # Filter out empty tokenized texts to avoid ZeroDivisionError
        corpus = [tokens for tokens in corpus if tokens]
        if not corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(corpus, **self.bm25_kwargs)

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[NodeWithScore]:
        """Retrieve nodes using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of NodeWithScore objects
        """
        if self._bm25 is None or not self.nodes:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # Filter out zero-score results
        ]

    def get_node_ids(self) -> List[str]:
        """Get all node IDs in the index.

        Returns:
            List of node IDs
        """
        return list(self.node_id_to_idx.keys())

    def clear(self) -> None:
        """Clear all nodes from the index."""
        self.nodes.clear()
        self.node_id_to_idx.clear()
        self._bm25 = None

    def count(self) -> int:
        """Count nodes in the index.

        Returns:
            Number of nodes
        """
        return len(self.nodes)


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both retrieval methods.
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25_index: Optional[BM25Index] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
        **kwargs
    ):
        """Initialize hybrid retriever.

        Args:
            vector_index: LlamaIndex VectorStoreIndex for vector search
            bm25_index: BM25Index for keyword search (optional)
            alpha: Weight for vector search (1-alpha for BM25)
                   Default 0.5 means equal weight
            rrf_k: RRF constant for smoothing (default 60)
            **kwargs: Additional arguments
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.kwargs = kwargs

        # Create vector retriever
        self._vector_retriever = vector_index.as_retriever(**kwargs)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[NodeWithScore]:
        """Perform hybrid retrieval.

        Args:
            query: Query string
            top_k: Number of final results to return
            **kwargs: Additional arguments

        Returns:
            List of NodeWithScore objects after RRF fusion
        """
        # Vector retrieval
        vector_nodes = self._vector_retriever.retrieve(query)

        # BM25 retrieval (if available)
        bm25_nodes = []
        if self.bm25_index is not None:
            bm25_nodes = self.bm25_index.retrieve(query, top_k=top_k)

        # If no BM25, just return vector results
        if not bm25_nodes:
            return vector_nodes[:top_k]

        # If no vector results, just return BM25 results
        if not vector_nodes:
            return bm25_nodes[:top_k]

        # Apply RRF fusion
        fused_results = self._rrf_fusion(vector_nodes, bm25_nodes)

        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Apply Reciprocal Rank Fusion (RRF) to combine results.

        RRF score = alpha / (k + rank_vector) + (1-alpha) / (k + rank_bm25)

        Args:
            vector_nodes: Results from vector search
            bm25_nodes: Results from BM25 search

        Returns:
            Fused and sorted results
        """
        scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}

        # Process vector results
        for rank, node_with_score in enumerate(vector_nodes):
            node_id = node_with_score.node.node_id
            rrf_score = self.alpha / (self.rrf_k + rank + 1)
            scores[node_id] = scores.get(node_id, 0) + rrf_score
            node_map[node_id] = node_with_score

        # Process BM25 results
        for rank, node_with_score in enumerate(bm25_nodes):
            node_id = node_with_score.node.node_id
            rrf_score = (1 - self.alpha) / (self.rrf_k + rank + 1)
            scores[node_id] = scores.get(node_id, 0) + rrf_score
            if node_id not in node_map:
                node_map[node_id] = node_with_score

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [node_map[node_id] for node_id in sorted_ids]

    def add_nodes_to_bm25(self, nodes: List[TextNode]) -> None:
        """Add nodes to BM25 index.

        Args:
            nodes: List of TextNode objects to add
        """
        if self.bm25_index is None:
            self.bm25_index = BM25Index()
        self.bm25_index.add_nodes(nodes)

    def clear_bm25(self) -> None:
        """Clear BM25 index."""
        if self.bm25_index is not None:
            self.bm25_index.clear()

    def update_bm25_from_vector_index(self) -> None:
        """Sync BM25 index with nodes from vector index.

        This requires iterating through all nodes in the vector store.
        """
        if self.bm25_index is None:
            self.bm25_index = BM25Index()

        # Get ref docs from vector store
        ref_docs = self.vector_index.ref_doc_info
        if ref_docs:
            for ref_doc_id, ref_doc_info in ref_docs.items():
                for node_id in ref_doc_info.node_ids:
                    node = self.vector_index.docstore.get_node(node_id)
                    if node:
                        self.bm25_index.add_nodes([node])