"""Hybrid retrieval with BM25 support"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from ..ingestion.image_processor import ImageResult, RetrievalResult

class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    both retrieval methods. When vector_store has native BM25 (Qdrant),
    delegates to vector_store.query() which handles hybrid search internally.
    """

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        alpha: float = 0.5,
        rrf_k: int = 60,
        vector_store: Optional[Any] = None,
        retrieve_mode: str = "hybrid",
        **kwargs
    ):
        """Initialize hybrid retriever.

        Args:
            vector_index: LlamaIndex VectorStoreIndex for vector search
            alpha: Weight for vector search (1-alpha for BM25)
                   Default 0.5 means equal weight
            rrf_k: RRF constant for smoothing (default 60)
            vector_store: Optional BaseVectorStore reference.
                         If it has native BM25 (use_bm25=True), retrieval
                         is delegated to vector_store.query().
            retrieve_mode: Retrieval mode - "hybrid" (dense+BM25), "sparse" (BM25 only),
                          or "vector" (dense only). Default is "hybrid".
            **kwargs: Additional arguments passed to as_retriever
        """
        self.vector_index = vector_index
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.vector_store = vector_store
        self.retrieve_mode = retrieve_mode
        self.kwargs = kwargs

        # Map retrieve_mode to VectorStoreQueryMode
        self._query_mode = self._map_retrieve_mode(retrieve_mode)

        # Create retriever with native LlamaIndex support
        retriever_kwargs = kwargs.copy()
        retriever_kwargs["vector_store_query_mode"] = self._query_mode
        retriever_kwargs["alpha"] = alpha

        self._vector_retriever = vector_index.as_retriever(**retriever_kwargs) if vector_index is not None else None

    @staticmethod
    def _map_retrieve_mode(mode: Optional[str]) -> VectorStoreQueryMode:
        """Map retrieve_mode string to VectorStoreQueryMode enum.

        Args:
            mode: Retrieve mode string ("hybrid", "sparse", "vector")

        Returns:
            VectorStoreQueryMode enum value
        """
        mode_map = {
            "hybrid": VectorStoreQueryMode.HYBRID,
            "sparse": VectorStoreQueryMode.SPARSE,
            "vector": VectorStoreQueryMode.DEFAULT,
        }
        return mode_map.get(mode, VectorStoreQueryMode.HYBRID)

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
            # If no embedding provided, generate it from query text using the index's embed model
            if query_embedding is None and self.vector_index is not None:
                query_embedding = self.vector_index._embed_model.get_text_embedding(query)
            query_bundle = QueryBundle(query_str=query, embedding=query_embedding)
            return self.vector_store.query(query_bundle, similarity_top_k=top_k, **kwargs)

        # Vector retrieval (only if vector retriever is available)
        vector_nodes = []
        if self._vector_retriever is not None:
            vector_nodes = self._vector_retriever.retrieve(query)

        return vector_nodes[:top_k]

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