"""Local file vector store implementation using LlamaIndex SimpleVectorStore"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

from .base import BaseVectorStore
from .registry import StorageRegistry


@StorageRegistry.register("local")
class LocalStore(BaseVectorStore):
    """Local file vector store backend implementation.

    Uses LlamaIndex's SimpleVectorStore with file persistence.
    Suitable for development, testing, and small-scale deployments.
    """

    def __init__(
        self,
        persist_path: str = "./storage",
        collection_name: str = "default",
        dimension: int = 1536,
        **kwargs
    ):
        """Initialize local file vector store.

        Args:
            persist_path: Directory path for storage files
            collection_name: Name for the collection (used as subdirectory)
            dimension: Vector dimension (default 1536 for OpenAI)
            **kwargs: Additional arguments
        """
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create storage directory
        self._storage_dir = self.persist_path / collection_name
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._vector_store_file = self._storage_dir / "vector_store.json"
        self._doc_store_file = self._storage_dir / "doc_store.json"
        self._index_store_file = self._storage_dir / "index_store.json"

        # Initialize or load storage
        self._initialize_storage()

    @property
    def client(self) -> Any:
        """Get the underlying storage context."""
        return self._storage_context

    def _initialize_storage(self) -> None:
        """Initialize or load existing storage."""
        if self._vector_store_file.exists():
            # Load existing storage
            self._vector_store = SimpleVectorStore.from_persist_path(
                str(self._vector_store_file)
            )
        else:
            # Create new storage
            self._vector_store = SimpleVectorStore()

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
            persist_dir=str(self._storage_dir)
        )

        # Initialize document storage
        self._doc_store: Dict[str, Dict] = {}
        if self._doc_store_file.exists():
            with open(self._doc_store_file, "r") as f:
                self._doc_store = json.load(f)

        # Initialize node storage
        self._node_store: Dict[str, Dict] = {}
        self._node_file = self._storage_dir / "nodes.json"
        if self._node_file.exists():
            with open(self._node_file, "r") as f:
                self._node_store = json.load(f)

    def add(self, nodes: List[TextNode], **kwargs) -> List[str]:
        """Add nodes to local storage.

        Args:
            nodes: List of TextNode objects to add
            **kwargs: Additional arguments

        Returns:
            List of node IDs that were added
        """
        if not nodes:
            return []

        ids = self._vector_store.add(nodes, **kwargs)

        # Store node content and metadata
        for node in nodes:
            self._node_store[node.node_id] = {
                "id": node.node_id,
                "text": node.text,
                "metadata": node.metadata or {},
                "embedding": node.embedding,
                "ref_doc_id": node.ref_doc_id,
            }

            # Track ref_doc_id
            if node.ref_doc_id:
                if node.ref_doc_id not in self._doc_store:
                    self._doc_store[node.ref_doc_id] = {"node_ids": []}
                self._doc_store[node.ref_doc_id]["node_ids"].append(node.node_id)

        # Persist changes
        self.persist()

        return ids

    def delete(self, ref_doc_id: Optional[str] = None, node_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """Delete nodes from local storage.

        Args:
            ref_doc_id: Document ID to delete all associated nodes
            node_ids: List of specific node IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful
        """
        if ref_doc_id:
            # Delete all nodes for this document
            if ref_doc_id in self._doc_store:
                node_ids_to_delete = self._doc_store[ref_doc_id].get("node_ids", [])
                for nid in node_ids_to_delete:
                    if nid in self._node_store:
                        del self._node_store[nid]
                del self._doc_store[ref_doc_id]
                self._vector_store.delete(ref_doc_id=ref_doc_id, **kwargs)
        elif node_ids:
            # Delete specific nodes
            for nid in node_ids:
                if nid in self._node_store:
                    node_info = self._node_store[nid]
                    ref_doc = node_info.get("ref_doc_id")
                    if ref_doc and ref_doc in self._doc_store:
                        self._doc_store[ref_doc]["node_ids"] = [
                            x for x in self._doc_store[ref_doc]["node_ids"] if x != nid
                        ]
                    del self._node_store[nid]
            self._vector_store.delete(node_ids=node_ids, **kwargs)
        else:
            # Clear all
            self.clear()

        self.persist()
        return True

    def query(self, query: QueryBundle, similarity_top_k: int = 10, **kwargs) -> List[NodeWithScore]:
        """Query local storage for similar nodes.

        Args:
            query: QueryBundle containing query text and/or embedding
            similarity_top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of NodeWithScore objects
        """
        results = self._vector_store.query(query, similarity_top_k=similarity_top_k, **kwargs)

        # Reconstruct full nodes with text
        full_results = []
        for node_with_score in results:
            node_id = node_with_score.node.node_id
            if node_id in self._node_store:
                node_info = self._node_store[node_id]
                full_node = TextNode(
                    id_=node_id,
                    text=node_info["text"],
                    metadata=node_info.get("metadata", {}),
                    embedding=node_info.get("embedding"),
                    ref_doc_id=node_info.get("ref_doc_id"),
                )
                full_results.append(NodeWithScore(node=full_node, score=node_with_score.score))
            else:
                full_results.append(node_with_score)

        return full_results

    def get_node(self, node_id: str) -> Optional[TextNode]:
        """Get a specific node by ID from local storage.

        Args:
            node_id: The node ID

        Returns:
            TextNode if found, None otherwise
        """
        if node_id not in self._node_store:
            return None

        node_info = self._node_store[node_id]
        return TextNode(
            id_=node_id,
            text=node_info["text"],
            metadata=node_info.get("metadata", {}),
            embedding=node_info.get("embedding"),
            ref_doc_id=node_info.get("ref_doc_id"),
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info from local storage.

        Args:
            ref_doc_id: Reference document ID

        Returns:
            RefDocInfo if found, None otherwise
        """
        if ref_doc_id not in self._doc_store:
            return None

        node_ids = self._doc_store[ref_doc_id].get("node_ids", [])
        return RefDocInfo(node_ids=node_ids)

    def persist(self, persist_path: Optional[str] = None, **kwargs) -> None:
        """Persist local storage to files.

        Args:
            persist_path: Optional alternative path to persist to
            **kwargs: Additional arguments
        """
        target_dir = Path(persist_path) if persist_path else self._storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Persist vector store
        self._vector_store.persist(str(target_dir / "vector_store.json"))

        # Persist doc store
        with open(target_dir / "doc_store.json", "w") as f:
            json.dump(self._doc_store, f, indent=2)

        # Persist node store
        with open(target_dir / "nodes.json", "w") as f:
            json.dump(self._node_store, f, indent=2)

    def count(self) -> int:
        """Count nodes in local storage.

        Returns:
            Number of nodes
        """
        return len(self._node_store)

    def clear(self) -> None:
        """Clear all nodes from local storage."""
        self._node_store.clear()
        self._doc_store.clear()
        self._vector_store = SimpleVectorStore()
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self.persist()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LocalStore":
        """Create LocalStore from configuration.

        Args:
            config: Configuration dictionary with keys:
                - persist_path: Storage directory path (default "./storage")
                - collection_name: Collection name (default "default")
                - dimension: Vector dimension (default 1536)

        Returns:
            LocalStore instance
        """
        return cls(
            persist_path=config.get("persist_path", "./storage"),
            collection_name=config.get("collection_name", "default"),
            dimension=config.get("dimension", 1536),
            **config.get("store_options", {})
        )