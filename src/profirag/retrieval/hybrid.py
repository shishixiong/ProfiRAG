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

        # Vector retrieval (only if vector retriever is available)
        retriever_kwargs = kwargs.copy()
        retriever_kwargs["vector_store_query_mode"] = self._query_mode
        retriever_kwargs["alpha"] = self.alpha
        retriever_kwargs["similarity_top_k"] = top_k
        retriever_kwargs["sparse_top_k"] = top_k
        retriever_kwargs["hybrid_top_k"] = top_k

        vector_nodes = self.vector_index.as_retriever(**retriever_kwargs).retrieve(query)

        return vector_nodes[:top_k]
    
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