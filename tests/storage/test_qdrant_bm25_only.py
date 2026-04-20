"""Tests for BM25-only (sparse vector) ingestion in QdrantStore"""

import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from src.profirag.storage.qdrant_store import QdrantStore


class TestQdrantStoreBM25Only:
    """Test BM25-only mode (no dense vectors)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient."""
        client = MagicMock(spec=QdrantClient)
        client.get_collections.return_value.collections = []
        client.upsert.return_value = None
        return client

    def test_ensure_collection_creates_sparse_only(self, mock_client):
        """Test that collection is created with sparse vectors only when dense_vector_name=None."""
        store = QdrantStore(
            collection_name="test_bm25_only",
            client=mock_client,
            dimension=1536,
            use_bm25=True,
            dense_vector_name=None,
        )

        # Verify create_collection was called with sparse only
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs

        assert "vectors_config" not in call_kwargs
        assert "sparse_vectors_config" in call_kwargs
        assert "sparse-text" in call_kwargs["sparse_vectors_config"]

    def test_add_stores_sparse_vectors_only(self, mock_client):
        """Test that add() stores only sparse vectors, not dense."""
        store = QdrantStore(
            collection_name="test_bm25_only",
            client=mock_client,
            dimension=1536,
            use_bm25=True,
            dense_vector_name=None,
        )

        # Create a mock node with both text and embedding
        from llama_index.core.schema import TextNode

        node = TextNode(
            id_="test-node-1",
            text="这是一个测试文档",
            embedding=[0.1, 0.2, 0.3] * 512,  # Mock embedding
        )

        store.add([node])

        # Verify upsert was called (twice: once for IDF metadata, once for the node)
        assert mock_client.upsert.call_count == 2

        # Get all points from both calls
        all_points = []
        for call in mock_client.upsert.call_args_list:
            all_points.extend(call.kwargs["points"])

        # Find the actual data point (not the IDF metadata point)
        data_points = [p for p in all_points if p.id != "00000000-0000-0000-0000-000000000001"]
        assert len(data_points) == 1
        point = data_points[0]

        # Should have sparse vector
        assert "sparse-text" in point.vector

        # Should NOT have dense vector
        assert "dense" not in point.vector
        assert len(point.vector) == 1  # Only sparse

    def test_dense_vector_name_default_creates_hybrid(self, mock_client):
        """Test that default dense_vector_name='dense' creates hybrid collection."""
        store = QdrantStore(
            collection_name="test_hybrid",
            client=mock_client,
            dimension=1536,
            use_bm25=True,
            # dense_vector_name defaults to "dense"
        )

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs

        assert "vectors_config" in call_kwargs
        assert "dense" in call_kwargs["vectors_config"]
        assert "sparse_vectors_config" in call_kwargs

    def test_has_native_bm25_returns_true_for_bm25_only(self, mock_client):
        """Test has_native_bm25() works correctly in BM25-only mode."""
        store = QdrantStore(
            collection_name="test_bm25_only",
            client=mock_client,
            dimension=1536,
            use_bm25=True,
            dense_vector_name=None,
        )

        # After adding nodes, has_native_bm25 should be True (if IDF computed)
        from llama_index.core.schema import TextNode

        node = TextNode(id_="test-node-1", text="测试文本内容")
        store.add([node])

        assert store.has_native_bm25() is True