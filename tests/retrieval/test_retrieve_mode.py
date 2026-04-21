"""Tests for retrieve_mode in HybridRetriever"""

import pytest
from unittest.mock import MagicMock
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from profirag.retrieval.hybrid import HybridRetriever


class TestRetrieveModeMapping:
    """Test retrieve_mode to VectorStoreQueryMode mapping."""

    def test_map_retrieve_mode_hybrid(self):
        """Test hybrid mode maps to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode("hybrid")
        assert mode == VectorStoreQueryMode.HYBRID

    def test_map_retrieve_mode_sparse(self):
        """Test sparse mode maps to SPARSE."""
        mode = HybridRetriever._map_retrieve_mode("sparse")
        assert mode == VectorStoreQueryMode.SPARSE

    def test_map_retrieve_mode_vector(self):
        """Test vector mode maps to DEFAULT."""
        mode = HybridRetriever._map_retrieve_mode("vector")
        assert mode == VectorStoreQueryMode.DEFAULT

    def test_map_retrieve_mode_invalid_defaults_to_hybrid(self):
        """Test invalid mode defaults to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode("invalid")
        assert mode == VectorStoreQueryMode.HYBRID

    def test_map_retrieve_mode_none_defaults_to_hybrid(self):
        """Test None defaults to HYBRID."""
        mode = HybridRetriever._map_retrieve_mode(None)
        assert mode == VectorStoreQueryMode.HYBRID


class TestHybridRetrieverRetrieveMode:
    """Test HybridRetriever initialization with retrieve_mode."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        index = MagicMock()
        index.as_retriever = MagicMock(return_value=MagicMock())
        return index

    def test_init_with_retrieve_mode_hybrid(self, mock_index):
        """Test initialization with hybrid mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="hybrid",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "hybrid"
        assert retriever._query_mode == VectorStoreQueryMode.HYBRID

    def test_init_with_retrieve_mode_sparse(self, mock_index):
        """Test initialization with sparse mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="sparse",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "sparse"
        assert retriever._query_mode == VectorStoreQueryMode.SPARSE

    def test_init_with_retrieve_mode_vector(self, mock_index):
        """Test initialization with vector mode."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="vector",
            alpha=0.5,
        )
        assert retriever.retrieve_mode == "vector"
        assert retriever._query_mode == VectorStoreQueryMode.DEFAULT

    def test_init_passes_query_mode_to_as_retriever(self, mock_index):
        """Test that vector_store_query_mode is passed to as_retriever."""
        retriever = HybridRetriever(
            vector_index=mock_index,
            retrieve_mode="sparse",
            alpha=0.7,
        )
        mock_index.as_retriever.assert_called_once()
        call_kwargs = mock_index.as_retriever.call_args.kwargs
        assert call_kwargs["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE
        assert call_kwargs["alpha"] == 0.7