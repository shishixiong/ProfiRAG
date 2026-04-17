"""Hybrid retrieval with BM25 support"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from rank_bm25 import BM25Okapi
import jieba  # Chinese tokenizer

from ..ingestion.image_processor import ImageResult, RetrievalResult


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
    both retrieval methods. When vector_store has native BM25 (Qdrant),
    delegates to vector_store.query() which handles hybrid search internally.
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        bm25_index: Optional[BM25Index] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
        vector_store: Optional[Any] = None,
        **kwargs
    ):
        """Initialize hybrid retriever.

        Args:
            vector_index: LlamaIndex VectorStoreIndex for vector search
            bm25_index: BM25Index for keyword search (optional)
            alpha: Weight for vector search (1-alpha for BM25)
                   Default 0.5 means equal weight
            rrf_k: RRF constant for smoothing (default 60)
            vector_store: Optional BaseVectorStore reference.
                         If it has native BM25 (use_bm25=True), retrieval
                         is delegated to vector_store.query().
            **kwargs: Additional arguments
        """
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.vector_store = vector_store
        self.kwargs = kwargs

        # Create vector retriever (only if vector_index is available)
        self._vector_retriever = vector_index.as_retriever(**kwargs) if vector_index is not None else None

    @property
    def _use_native_bm25(self) -> bool:
        """Check if vector store handles BM25 natively."""
        if self.vector_store is not None:
            return getattr(self.vector_store, "has_native_bm25", lambda: False)()
        return False

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
        # Check if vector store handles BM25 natively (Qdrant with use_bm25=True)
        if self._use_native_bm25:
            # Delegate to vector store which does hybrid search internally
            from llama_index.core.schema import QueryBundle
            query_embedding = kwargs.get("query_embedding")
            query_bundle = QueryBundle(query_str=query, embedding=query_embedding)
            return self.vector_store.query(query_bundle, similarity_top_k=top_k, **kwargs)

        # Vector retrieval (only if vector retriever is available)
        vector_nodes = []
        if self._vector_retriever is not None:
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

    def retrieve_with_images(
        self,
        query: str,
        top_k: int = 10,
        include_images: bool = True,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve text chunks and associated images.

        Args:
            query: Query string
            top_k: Number of final results to return
            include_images: Whether to collect associated images
            **kwargs: Additional arguments

        Returns:
            RetrievalResult containing text nodes and images
        """
        # Standard text retrieval
        text_nodes = self.retrieve(query, top_k=top_k)

        # Collect images from retrieved chunks
        images = []
        if include_images:
            for node_with_score in text_nodes:
                node = node_with_score.node
                # Get image paths directly from chunk metadata (not from image_map)
                image_paths = node.metadata.get("image_paths", [])
                chunk_images = node.metadata.get("chunk_images", [])

                for i, img_id in enumerate(chunk_images):
                    # Get image path - either from image_paths list or fallback
                    if i < len(image_paths):
                        img_path = image_paths[i]
                    else:
                        img_path = img_id  # Fallback

                    if img_path:
                        image_result = ImageResult(
                            image_path=img_path,
                            description="",  # Description stored separately
                            score=node_with_score.score,
                            source_chunk_id=node.node_id,
                            metadata={"image_id": img_id},
                        )
                        images.append(image_result)

        # Deduplicate images by path
        unique_images = self._deduplicate_images(images)

        return RetrievalResult(
            text_nodes=text_nodes,
            images=unique_images,
        )

    def _deduplicate_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """Deduplicate images by path, keeping highest score.

        Args:
            images: List of ImageResult objects

        Returns:
            Deduplicated list of ImageResult objects
        """
        if not images:
            return []

        # Group by path, keep highest score
        path_to_image: Dict[str, ImageResult] = {}
        for img in images:
            path = img.image_path
            if path not in path_to_image or img.score > path_to_image[path].score:
                path_to_image[path] = img

        return list(path_to_image.values())