"""Qdrant vector store implementation"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from .base import BaseVectorStore
from .registry import StorageRegistry


@StorageRegistry.register("qdrant")
class QdrantStore(BaseVectorStore):
    """Qdrant vector store backend implementation.

    Supports both local and cloud Qdrant deployments.
    """

    def __init__(
        self,
        collection_name: str,
        client: QdrantClient,
        dimension: int = 1536,
        distance: str = "Cosine",
        prefer_grpc: bool = False,
        aclient: Optional[AsyncQdrantClient] = None,
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
            **kwargs: Additional arguments
        """
        self.collection_name = collection_name
        self._client = client
        self._aclient = aclient
        self.dimension = dimension
        self.distance = Distance[distance.upper()]

        # Create collection if not exists
        self._ensure_collection_exists()

        # Create LlamaIndex vector store wrapper
        self._vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            prefer_grpc=prefer_grpc,
            aclient=aclient,  # Pass async client for async operations
            **kwargs
        )

    @property
    def client(self) -> QdrantClient:
        """Get the underlying Qdrant client."""
        return self._client

    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if not."""
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
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

        # Use LlamaIndex vector store to add nodes
        ids = self._vector_store.add(nodes, **kwargs)
        return ids

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
            self._vector_store.delete(ref_doc_id=ref_doc_id, **kwargs)
        elif node_ids:
            self._vector_store.delete(node_ids=node_ids, **kwargs)
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
            **config.get("store_options", {})
        )