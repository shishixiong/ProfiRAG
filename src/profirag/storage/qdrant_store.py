"""Qdrant vector store implementation with native BM25 (sparse vector) support"""

from typing import List, Optional, Dict, Any, Tuple
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus, SparseVectorParams
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import models

from .base import BaseVectorStore
from .registry import StorageRegistry


@StorageRegistry.register("qdrant")
class QdrantStore(BaseVectorStore):
    """Qdrant vector store backend implementation.

    Supports both local and cloud Qdrant deployments.
    When use_bm25=True, enables Qdrant native sparse vector storage and hybrid retrieval.
    """

    # Sparse vector configuration
    SPARSE_VECTOR_NAME = "sparse-text"

    def __init__(
        self,
        collection_name: str,
        client: QdrantClient,
        dimension: int = 1536,
        distance: str = "Cosine",
        prefer_grpc: bool = False,
        aclient: Optional[AsyncQdrantClient] = None,
        use_bm25: bool = False,
        sparse_vectorizer: Optional[Any] = None,
        **kwargs
    ):
        """Initialize Qdrant vector store.

        Args:
            collection_name: Name of the Qdrant collection
            client: QdrantClient instance
            dimension: Vector dimension (default 1536 for OpenAI embeddings)
            distance: Distance metric ("Cosine", "Euclidean", "Dot")
            prefer_grpc: Whether to use gRPC for communication
            aclient: Optional AsyncQdrantClient for async operations
            use_bm25: Enable Qdrant native BM25 (sparse vector) support
            sparse_vectorizer: SparseVectorizer instance for BM25 (optional).
                               If not provided but use_bm25=True, creates one internally.
            **kwargs: Additional arguments
        """
        self.collection_name = collection_name
        self._client = client
        self._aclient = aclient
        self.dimension = dimension
        self.distance = Distance[distance.upper()]
        self.use_bm25 = use_bm25

        # Sparse vectorizer (for BM25)
        self._sparse_vectorizer = sparse_vectorizer

        # Create collection if not exists
        self._ensure_collection_exists()

        # Create LlamaIndex vector store wrapper
        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            prefer_grpc=prefer_grpc,
            aclient=aclient,
            **kwargs
        )

        # Try to load IDF from existing collection if use_bm25=True
        if self.use_bm25 and self._sparse_vectorizer is None:
            self._sparse_vectorizer = self._try_load_idf_from_collection()
        elif self._sparse_vectorizer is None:
            # Create an empty vectorizer for tokenization-only use
            from ..retrieval.sparse_vectorizer import SparseVectorizer
            self._sparse_vectorizer = SparseVectorizer()

    @property
    def client(self) -> QdrantClient:
        """Get the underlying Qdrant client."""
        return self._client

    def has_native_bm25(self) -> bool:
        """Check if Qdrant native BM25 (sparse vector) is enabled and ready.

        Returns:
            True if use_bm25=True and IDF has been computed, False otherwise.
        """
        return self.use_bm25 and self._sparse_vectorizer is not None and self._sparse_vectorizer.has_idf

    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if not."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            if self.use_bm25:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.dimension, distance=self.distance),
                        self.SPARSE_VECTOR_NAME: VectorParams(size=self.dimension, distance=Distance.DOT),
                    },
                    sparse_vectors_config={
                        self.SPARSE_VECTOR_NAME: SparseVectorParams()
                    },
                )
            else:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=self.distance
                    )
                )

    def add(self, nodes: List[TextNode], **kwargs) -> List[str]:
        """Add nodes to the Qdrant collection.

        Args:
            nodes: List of TextNode objects to add
            **kwargs: Additional arguments

        Returns:
            List of node IDs that were added
        """
        if not nodes:
            return []

        if self.use_bm25:
            # Fit vectorizer on new nodes to update IDF
            # This incrementally updates IDF with terms from new nodes
            self._sparse_vectorizer.fit_nodes(nodes)
            self._store_idf_to_payload()

            # Compute sparse vectors for each node and upsert
            points = []
            for node in nodes:
                indices, values = self._sparse_vectorizer.compute_sparse_vector(
                    node.text or "", with_idf=False
                )
                # Build vectors dict with both dense and sparse
                vectors = {
                    self.SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=indices,
                        values=values
                    )
                }
                # Get dense vector from node if available
                if hasattr(node, "embedding") and node.embedding:
                    vectors["dense"] = node.embedding

                point = models.PointStruct(
                    id=node.node_id,
                    vector=vectors,
                    payload={
                        "text": node.text or "",
                        "metadata": node.metadata or {},
                        "ref_doc_id": node.metadata.get("ref_doc_id", "") if node.metadata else "",
                    }
                )
                points.append(point)

            if points:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

            return [n.node_id for n in nodes]
        else:
            # Use LlamaIndex vector store to add nodes
            ids = self._vector_store.add(nodes, **kwargs)
            return ids

    def _store_idf_to_payload(self) -> None:
        """Store IDF values to collection metadata payload.

        Uses a special point to store global IDF values.
        """
        if not self._sparse_vectorizer.has_idf:
            return

        idf_payload = self._sparse_vectorizer.get_idf_payload()
        # Store as a special point with known ID
        idf_point = models.PointStruct(
            id="__idf_meta__",
            vector={
                self.SPARSE_VECTOR_NAME: models.SparseVector(indices=[0], values=[0.0])
            },
            payload={
                "_is_idf_meta": True,
                "idf_data": idf_payload,
            }
        )
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[idf_point],
            )
        except Exception:
            # If the point already exists, just update
            pass

    def _try_load_idf_from_collection(self) -> Optional[Any]:
        """Try to load IDF from existing collection metadata.

        Returns:
            SparseVectorizer with loaded IDF, or None if not found
        """
        from ..retrieval.sparse_vectorizer import SparseVectorizer

        try:
            results = self._client.retrieve(
                collection_name=self.collection_name,
                ids=["__idf_meta__"],
                with_payload=True,
            )
            if results and len(results) > 0:
                payload = results[0].payload
                if payload.get("_is_idf_meta"):
                    idf_data = payload.get("idf_data", {})
                    vectorizer = SparseVectorizer()
                    vectorizer.load_idf_from_payload(idf_data)
                    return vectorizer
        except Exception:
            pass
        return None

    def _query_hybrid(
        self,
        query_str: str,
        query_embedding: Optional[List[float]],
        top_k: int,
    ) -> List[NodeWithScore]:
        """Perform Qdrant native hybrid search (dense + sparse with RRF).

        Args:
            query_str: Query text
            query_embedding: Dense embedding vector (if available)
            top_k: Number of results to return

        Returns:
            List of NodeWithScore objects
        """
        if not self._sparse_vectorizer.has_idf:
            # Fallback to dense only
            if query_embedding:
                return self._query_dense(query_embedding, top_k)
            return []

        # Compute query sparse vector with IDF weighting
        query_indices, query_values = self._sparse_vectorizer.compute_query_sparse_vector(query_str)

        prefetch: List[models.Prefetch] = []

        # Prefetch sparse (BM25)
        if query_indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=query_indices, values=query_values),
                    using=self.SPARSE_VECTOR_NAME,
                    limit=top_k * 2,
                )
            )

        # Prefetch dense (if embedding provided)
        if query_embedding:
            prefetch.append(
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=top_k * 2,
                )
            )

        # Execute hybrid query with RRF fusion
        results = self._client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        # Convert to NodeWithScore
        nodes = []
        for hit in results.points:
            text = hit.payload.get("text", "")
            metadata = hit.payload.get("metadata", {}) or {}
            node = TextNode(
                id_=str(hit.id),
                text=text,
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=hit.score))

        return nodes

    def _query_dense(self, embedding: List[float], top_k: int) -> List[NodeWithScore]:
        """Query using dense vectors only.

        Args:
            embedding: Dense embedding vector
            top_k: Number of results

        Returns:
            List of NodeWithScore objects
        """
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            using="dense",
            limit=top_k,
            with_payload=True,
        )

        nodes = []
        for hit in results.points:
            text = hit.payload.get("text", "")
            metadata = hit.payload.get("metadata", {}) or {}
            node = TextNode(
                id_=str(hit.id),
                text=text,
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=hit.score))

        return nodes

    def delete(self, ref_doc_id: Optional[str] = None, node_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """Delete nodes from Qdrant.

        Args:
            ref_doc_id: Document ID to delete all associated nodes
            node_ids: List of specific node IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful
        """
        if ref_doc_id:
            # Get node IDs for this ref_doc_id and delete them
            ref_info = self.get_ref_doc_info(ref_doc_id)
            if ref_info:
                self._client.delete_points(
                    collection_name=self.collection_name,
                    points=ref_info.node_ids,
                )
        elif node_ids:
            self._client.delete_points(
                collection_name=self.collection_name,
                points=node_ids,
            )
        else:
            # Delete entire collection
            self._client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
        return True

    def query(self, query: QueryBundle, similarity_top_k: int = 10, **kwargs) -> List[NodeWithScore]:
        """Query Qdrant for similar nodes.

        Args:
            query: QueryBundle containing query text and/or embedding
            similarity_top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of NodeWithScore objects
        """
        if self.use_bm25 and self._sparse_vectorizer.has_idf:
            # Use Qdrant native hybrid search
            query_str = query.query_str or ""
            query_embedding = query.embedding if hasattr(query, "embedding") else None
            return self._query_hybrid(query_str, query_embedding, similarity_top_k)
        else:
            # Fallback to LlamaIndex vector store (dense only)
            return self._vector_store.query(query, similarity_top_k=similarity_top_k, **kwargs)

    def get_node(self, node_id: str) -> Optional[TextNode]:
        """Get a specific node by ID from Qdrant.

        Args:
            node_id: The node ID

        Returns:
            TextNode if found, None otherwise
        """
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[node_id],
            with_payload=True,
            with_vectors=False
        )

        if not results:
            return None

        point = results[0]
        return TextNode(
            id_=point.id,
            text=point.payload.get("text", ""),
            metadata=point.payload.get("metadata", {})
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info from Qdrant.

        Args:
            ref_doc_id: Reference document ID

        Returns:
            RefDocInfo if found, None otherwise
        """
        # Query for nodes with this ref_doc_id
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        results = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="ref_doc_id",
                        match=MatchValue(value=ref_doc_id)
                    )
                ]
            ),
            limit=100,
            with_payload=True
        )[0]

        if not results:
            return None

        node_ids = [r.id for r in results]
        return RefDocInfo(node_ids=node_ids)

    def persist(self, persist_path: Optional[str] = None, **kwargs) -> None:
        """Persist Qdrant storage.

        Note: Qdrant handles persistence automatically. This method
        is provided for interface compatibility.

        Args:
            persist_path: Path to persist to (ignored for Qdrant)
            **kwargs: Additional arguments
        """
        # Qdrant handles persistence internally
        # Optionally snapshot the collection
        if persist_path:
            self._client.create_snapshot(collection_name=self.collection_name)

    def count(self) -> int:
        """Count nodes in the Qdrant collection.

        Returns:
            Number of nodes
        """
        info = self._client.get_collection(self.collection_name)
        return info.points_count

    def clear(self) -> None:
        """Clear all nodes from the collection."""
        self._client.delete_collection(self.collection_name)
        self._ensure_collection_exists()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QdrantStore":
        """Create QdrantStore from configuration.

        Args:
            config: Configuration dictionary with keys:
                - collection_name: Name of the collection
                - host: Qdrant server host (default "localhost")
                - port: Qdrant server port (default 6333)
                - api_key: Optional API key for cloud
                - url: Optional URL for cloud deployment
                - dimension: Vector dimension (default 1536)
                - distance: Distance metric (default "Cosine")
                - prefer_grpc: Use gRPC (default False)
                - use_bm25: Enable Qdrant native BM25/sparse vector (default False)

        Returns:
            QdrantStore instance
        """
        # Create client
        if config.get("url"):
            # Cloud deployment
            client = QdrantClient(
                url=config.get("url"),
                api_key=config.get("api_key"),
            )
            aclient = AsyncQdrantClient(
                url=config.get("url"),
                api_key=config.get("api_key"),
            )
        else:
            # Local deployment
            client = QdrantClient(
                host=config.get("host", "localhost"),
                port=config.get("port", 6333),
                api_key=config.get("api_key"),
                prefer_grpc=config.get("prefer_grpc", False),
            )
            aclient = AsyncQdrantClient(
                host=config.get("host", "localhost"),
                port=config.get("port", 6333),
                api_key=config.get("api_key"),
                prefer_grpc=config.get("prefer_grpc", False),
            )

        return cls(
            collection_name=config["collection_name"],
            client=client,
            aclient=aclient,
            dimension=config.get("dimension", 1536),
            distance=config.get("distance", "Cosine"),
            prefer_grpc=config.get("prefer_grpc", False),
            use_bm25=config.get("use_bm25", False),
            **config.get("store_options", {})
        )