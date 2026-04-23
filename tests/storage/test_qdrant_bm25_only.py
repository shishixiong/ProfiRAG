"""Tests for QdrantStore index_mode functionality"""

import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from src.profirag.storage.qdrant_store import QdrantStore


class TestQdrantStoreIndexMode:
    """Test QdrantStore index_mode functionality (hybrid vs vector-only)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient."""
        client = MagicMock(spec=QdrantClient)
        client.get_collections.return_value.collections = []
        client.upsert.return_value = None
        return client

    def test_index_mode_hybrid_enables_hybrid_search(self, mock_client):
        """Test that index_mode='hybrid' creates hybrid-enabled vector store."""
        store = QdrantStore(
            collection_name="test_hybrid",
            client=mock_client,
            dimension=1536,
            index_mode="hybrid",
        )

        # Verify the internal vector store is configured for hybrid
        assert store.index_mode == "hybrid"
        # MinimalPayloadQdrantVectorStore should have enable_hybrid=True
        assert store._vector_store.enable_hybrid is True

    def test_index_mode_vector_disables_hybrid_search(self, mock_client):
        """Test that index_mode='vector' creates vector-only store."""
        store = QdrantStore(
            collection_name="test_vector",
            client=mock_client,
            dimension=1536,
            index_mode="vector",
        )

        # Verify the internal vector store is configured for vector-only
        assert store.index_mode == "vector"
        # MinimalPayloadQdrantVectorStore should have enable_hybrid=False
        assert store._vector_store.enable_hybrid is False

    def test_default_index_mode_is_hybrid(self, mock_client):
        """Test that default index_mode is 'hybrid'."""
        store = QdrantStore(
            collection_name="test_default",
            client=mock_client,
            dimension=1536,
        )

        assert store.index_mode == "hybrid"
        assert store._vector_store.enable_hybrid is True

    def test_add_with_hybrid_mode(self, mock_client):
        """Test that add() works with hybrid mode."""
        from llama_index.core.schema import TextNode

        store = QdrantStore(
            collection_name="test_hybrid_add",
            client=mock_client,
            dimension=1536,
            index_mode="hybrid",
        )

        # Create a mock node with both text and embedding
        node = TextNode(
            id_="test-node-1",
            text="这是一个测试文档",
            embedding=[0.1, 0.2, 0.3] * 512,  # Mock embedding
        )

        # The add should succeed (delegated to MinimalPayloadQdrantVectorStore)
        # We're testing that the parameters are correctly passed
        result = store.add([node])
        # Result depends on MinimalPayloadQdrantVectorStore.add()
        # In a mock scenario, this may return empty or mock values