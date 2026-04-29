"""Qdrant vector store implementation with hybrid search support"""

from typing import List, Optional, Dict, Any, Tuple, ClassVar
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, MetadataMode, BaseNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.utils import iter_batch
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue
)
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from .base import BaseVectorStore
from .registry import StorageRegistry


class MinimalPayloadQdrantVectorStore(QdrantVectorStore):
    """Custom QdrantVectorStore that stores minimal payload without _node_content.

    This class inherits from LlamaIndex's QdrantVectorStore and overrides
    the _build_points method to store only essential fields:
    - text: The actual text content (for BM25 search)
    - doc_id: Reference document ID
    - _node_type: Node type for reconstruction
    - Essential metadata fields (excluding large/redundant ones)
    """

    # Metadata keys to exclude from storage (too large or redundant)
    LARGE_METADATA_KEYS: ClassVar[frozenset] = frozenset({
        "image_map", "b64_image", "image_data", "original_image",
        "combined_text", "_document", "page_content", "b64_json",
        "base64", "embeddings", "text_vector", "content_vector",
        "chunk_images",  # Image data can be large
    })

    # Maximum metadata value length (50KB per value)
    MAX_METADATA_VALUE_LEN: ClassVar[int] = 50 * 1024

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to exclude large/redundant fields.

        Args:
            metadata: Original metadata dict

        Returns:
            Filtered metadata dict safe for Qdrant storage
        """
        filtered = {}
        for k, v in metadata.items():
            # Skip known large field names
            if k.lower() in {k.lower() for k in self.LARGE_METADATA_KEYS}:
                continue
            # Skip very large values
            if isinstance(v, str) and len(v) > self.MAX_METADATA_VALUE_LEN:
                continue
            # Skip list/dict values that could be large
            if isinstance(v, (dict, list)):
                v_str = str(v)
                if len(v_str) > self.MAX_METADATA_VALUE_LEN:
                    continue
            filtered[k] = v
        return filtered

    def _build_minimal_payload(self, node: BaseNode) -> Dict[str, Any]:
        """Build minimal payload for Qdrant storage without _node_content.

        Args:
            node: BaseNode to build payload for

        Returns:
            Minimal payload dict with essential fields only
        """
        payload = {
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "_node_type": node.class_name(),
        }

        # Add ref_doc_id if present (for document filtering)
        if node.ref_doc_id:
            payload["doc_id"] = node.ref_doc_id
            payload["ref_doc_id"] = node.ref_doc_id

        # Add filtered metadata directly to payload (not nested)
        if node.metadata:
            filtered_metadata = self._filter_metadata(node.metadata)
            for k, v in filtered_metadata.items():
                payload[k] = v

        return payload

    def _build_points(
        self, nodes: List[BaseNode], sparse_vector_name: str
    ) -> Tuple[List[Any], List[str]]:
        """Build Qdrant points with minimal payload (no _node_content).

        This overrides the parent method to store minimal payload instead of
        the full node serialized in _node_content.

        Args:
            nodes: List of nodes to build points for
            sparse_vector_name: Name of the sparse vector field

        Returns:
            Tuple of (points list, node IDs list)
        """
        ids = []
        points = []

        for node_batch in iter_batch(nodes, self.batch_size):
            node_ids = []
            vectors: List[Any] = []
            sparse_vectors: List[List[float]] = []
            sparse_indices: List[List[int]] = []
            payloads = []

            if self.enable_hybrid and self._sparse_doc_fn is not None:
                sparse_indices, sparse_vectors = self._sparse_doc_fn(
                    [
                        node.get_content(metadata_mode=MetadataMode.EMBED)
                        for node in node_batch
                    ],
                )

            for i, node in enumerate(node_batch):
                assert isinstance(node, BaseNode)
                node_ids.append(node.node_id)

                if self.enable_hybrid:
                    if (
                        len(sparse_vectors) > 0
                        and len(sparse_indices) > 0
                        and len(sparse_vectors) == len(sparse_indices)
                    ):
                        vectors.append(
                            {
                                sparse_vector_name: rest.SparseVector(
                                    indices=sparse_indices[i],
                                    values=sparse_vectors[i],
                                ),
                                self.dense_vector_name: node.get_embedding(),
                            }
                        )
                    else:
                        vectors.append(
                            {
                                self.dense_vector_name: node.get_embedding(),
                            }
                        )
                else:
                    vectors.append({self.dense_vector_name: node.get_embedding()})

                # Use minimal payload instead of node_to_metadata_dict
                payloads.append(self._build_minimal_payload(node))

            points.extend(
                [
                    rest.PointStruct(id=node_id, payload=payload, vector=vector)
                    for node_id, payload, vector in zip(node_ids, payloads, vectors)
                ]
            )

            ids.extend(node_ids)

        return points, ids


@StorageRegistry.register("qdrant")
class QdrantStore(BaseVectorStore):
    """Qdrant vector store backend implementation.

    Supports both local and cloud Qdrant deployments.
    Supports hybrid search (dense + sparse BM25) and pure vector search.
    """

    # LlamaIndex compatibility properties
    stores_text: ClassVar[bool] = True  # Indicates that text is stored in the vector store
    flat_metadata: ClassVar[bool] = False  # Metadata is not flattened
    is_embedding_query: ClassVar[bool] = True  # Requires embedding for query

    # Vector names for Qdrant collection
    SPARSE_VECTOR_NAME: ClassVar[str] = "text-sparse"
    DENSE_VECTOR_NAME: ClassVar[str] = "text-dense"

    def __init__(
        self,
        collection_name: str,
        client: QdrantClient,
        dimension: int = 1536,
        distance: str = "Cosine",
        prefer_grpc: bool = False,
        aclient: Optional[AsyncQdrantClient] = None,
        index_mode: str = "hybrid",
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
            index_mode: Index mode - "hybrid" (dense + BM25) or "vector" (dense only)
            **kwargs: Additional arguments
        """
        self.collection_name = collection_name
        self._client = client
        self._aclient = aclient
        self.dimension = dimension
        self.distance = Distance[distance.upper()]
        self.index_mode = index_mode

        # Create collection if not exists
        # self._ensure_collection_exists()

        # Initialize minimal payload vector store (custom implementation)
        if self.index_mode == "hybrid":
            # Hybrid mode: dense vectors + BM25 sparse vectors
            self._vector_store = MinimalPayloadQdrantVectorStore(
                client=client,
                collection_name=collection_name,
                prefer_grpc=prefer_grpc,
                aclient=aclient,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
                **kwargs
            )
        else:
            # Vector-only mode: dense vectors only
            self._vector_store = MinimalPayloadQdrantVectorStore(
                client=client,
                collection_name=collection_name,
                prefer_grpc=prefer_grpc,
                aclient=aclient,
                enable_hybrid=False,
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
            if self.index_mode == "hybrid":
                # Hybrid mode: dense vectors + sparse vectors (BM25)
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_VECTOR_NAME: VectorParams(size=self.dimension, distance=self.distance),
                    },
                    sparse_vectors_config={
                        self.SPARSE_VECTOR_NAME: SparseVectorParams()
                    },
                )
            else:
                # Vector-only mode: dense vectors only
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_VECTOR_NAME: VectorParams(size=self.dimension, distance=self.distance),
                    },
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
        # Use the minimal payload vector store to add nodes
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
            # Delete by ref_doc_id using filter
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="ref_doc_id",
                            match=MatchValue(value=ref_doc_id)
                        )
                    ]
                ),
            )
        elif node_ids:
            # Delete by node IDs directly
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=node_ids,
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
        # Use LlamaIndex vector store for query (handles both hybrid and vector modes)
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
            id_=str(point.id),
            text=point.payload.get("text", ""),
            metadata=point.payload.get("metadata", {}) or {}
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info from Qdrant.

        Args:
            ref_doc_id: Reference document ID

        Returns:
            RefDocInfo if found, None otherwise
        """
        try:
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

            node_ids = [str(r.id) for r in results]
            return RefDocInfo(node_ids=node_ids)
        except Exception as ex:
            if hasattr(ex, 'status_code') and ex.status_code == 404:
                return None
            if "404" in str(ex) or "Not found" in str(ex):
                return None
            raise

    def persist(self, persist_path: Optional[str] = None, **kwargs) -> None:
        """Persist Qdrant storage.

        Note: Qdrant handles persistence automatically. This method
        is provided for interface compatibility.

        Args:
            persist_path: Path to persist to (ignored for Qdrant)
            **kwargs: Additional arguments
        """
        # Qdrant handles persistence internally
        if persist_path:
            self._client.create_snapshot(collection_name=self.collection_name)

    def count(self) -> int:
        """Count nodes in the Qdrant collection.

        Returns:
            Number of nodes (0 if collection doesn't exist)
        """
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            # Collection doesn't exist yet
            return 0

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
                - index_mode: Index mode - "hybrid" or "vector" (default "hybrid")

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
            index_mode=config.get("index_mode", "hybrid"),
            **config.get("store_options", {})
        )