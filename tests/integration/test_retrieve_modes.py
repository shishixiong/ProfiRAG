"""Integration tests for retrieve_mode functionality"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import NodeWithScore, TextNode

from profirag.config.settings import RAGConfig, RetrievalConfig
from profirag.retrieval.hybrid import HybridRetriever


class TestRetrieveModesIntegration:
    """Test that all three retrieve modes work correctly."""

    @pytest.fixture
    def mock_vector_index(self):
        """Create a mock VectorStoreIndex with retriever."""
        index = MagicMock()

        # Mock retriever that returns different results based on mode
        def mock_retrieve(query):
            return [
                NodeWithScore(node=TextNode(id_="node-1", text=f"Result for {query}"), score=0.9)
            ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = mock_retrieve
        index.as_retriever = MagicMock(return_value=mock_retriever)
        index._embed_model = MagicMock()
        index._embed_model.get_text_embedding = MagicMock(return_value=[0.1] * 1536)

        return index

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store with native BM25 support."""
        store = MagicMock()
        store.has_native_bm25 = MagicMock(return_value=True)
        store.query = MagicMock(return_value=[
            NodeWithScore(node=TextNode(id_="node-2", text="BM25 result"), score=0.85)
        ])
        return store

    @pytest.fixture
    def mock_vector_store_no_bm25(self):
        """Create a mock vector store without native BM25 support."""
        store = MagicMock()
        store.has_native_bm25 = MagicMock(return_value=False)
        return store

    def test_hybrid_mode_creates_hybrid_retriever(self, mock_vector_index, mock_vector_store):
        """Test that hybrid mode uses HYBRID query mode when retrieve() is called."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="hybrid",
        )

        # Call retrieve to trigger as_retriever call
        retriever.retrieve("test query", top_k=5)

        # Verify as_retriever was called with HYBRID mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.HYBRID

    def test_sparse_mode_creates_sparse_retriever(self, mock_vector_index, mock_vector_store):
        """Test that sparse mode uses SPARSE query mode when retrieve() is called."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="sparse",
        )

        # Call retrieve to trigger as_retriever call
        retriever.retrieve("test query", top_k=5)

        # Verify as_retriever was called with SPARSE mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE

    def test_vector_mode_creates_default_retriever(self, mock_vector_index, mock_vector_store):
        """Test that vector mode uses DEFAULT query mode when retrieve() is called."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="vector",
        )

        # Call retrieve to trigger as_retriever call
        retriever.retrieve("test query", top_k=5)

        # Verify as_retriever was called with DEFAULT mode
        mock_vector_index.as_retriever.assert_called_once()
        call_kwargs = mock_vector_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.DEFAULT

    def test_retrieve_delegates_to_vector_retriever(self, mock_vector_index, mock_vector_store_no_bm25):
        """Test that retrieve() delegates to the internal vector retriever when no native BM25."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store_no_bm25,
            retrieve_mode="hybrid",
        )

        results = retriever.retrieve("test query", top_k=5)

        # Should have called the retriever
        assert len(results) == 1
        assert results[0].node.text == "Result for test query"

    def test_retrieve_delegates_to_native_bm25(self, mock_vector_index, mock_vector_store):
        """Test that retrieve() delegates to vector_index.as_retriever() for hybrid mode."""
        retriever = HybridRetriever(
            vector_index=mock_vector_index,
            vector_store=mock_vector_store,
            retrieve_mode="hybrid",
        )

        results = retriever.retrieve("test query", top_k=5)

        # Should have called vector_index.as_retriever, not vector_store.query directly
        mock_vector_index.as_retriever.assert_called_once()
        assert len(results) == 1
        assert results[0].node.text == "Result for test query"

    def test_config_integration(self):
        """Test that RAGConfig properly passes retrieve_mode."""
        config = RAGConfig(
            storage={"type": "qdrant", "config": {"collection_name": "test"}},
            retrieval=RetrievalConfig(retrieve_mode="sparse"),
        )

        assert config.retrieval.retrieve_mode == "sparse"