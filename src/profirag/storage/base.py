"""Vector store abstraction base class"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import RefDocInfo


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations.

    This class defines the interface that all vector store backends must implement.
    Supports multiple backends: Qdrant, Local file storage, PostgreSQL/pgvector.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the vector store."""
        pass

    @property
    @abstractmethod
    def client(self) -> Any:
        """Get the underlying client for the vector store."""
        pass

    @abstractmethod
    def add(self, nodes: List[TextNode], **kwargs) -> List[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of TextNode objects to add
            **kwargs: Additional arguments for the add operation

        Returns:
            List of node IDs that were added
        """
        pass

    @abstractmethod
    def delete(self, ref_doc_id: Optional[str] = None, node_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """Delete nodes from the vector store.

        Args:
            ref_doc_id: Document ID to delete all associated nodes
            node_ids: List of specific node IDs to delete
            **kwargs: Additional arguments for the delete operation

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def query(self, query: QueryBundle, similarity_top_k: int = 10, **kwargs) -> List[NodeWithScore]:
        """Query the vector store for similar nodes.

        Args:
            query: QueryBundle containing the query text and/or embedding
            similarity_top_k: Number of top similar nodes to return
            **kwargs: Additional arguments for the query operation

        Returns:
            List of NodeWithScore objects containing the similar nodes and their scores
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[TextNode]:
        """Get a specific node by ID.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            The TextNode if found, None otherwise
        """
        pass

    @abstractmethod
    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info by document ID.

        Args:
            ref_doc_id: The reference document ID

        Returns:
            RefDocInfo if found, None otherwise
        """
        pass

    @abstractmethod
    def persist(self, persist_path: Optional[str] = None, **kwargs) -> None:
        """Persist the vector store to storage.

        Args:
            persist_path: Path to persist to (optional)
            **kwargs: Additional arguments for persistence
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Count the number of nodes in the vector store.

        Returns:
            Number of nodes in the store
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes from the vector store."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseVectorStore":
        """Create an instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Instance of the vector store
        """
        pass

    def to_llamaindex_vector_store(self) -> Any:
        """Convert to LlamaIndex VectorStore interface.

        Returns:
            LlamaIndex VectorStore compatible object
        """
        return self._vector_store

    # Internal LlamaIndex vector store wrapper
    _vector_store: Any = None