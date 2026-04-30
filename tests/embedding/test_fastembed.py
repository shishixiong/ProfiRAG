"""Unit tests for FastEmbedEmbedding class"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from profirag.embedding import FastEmbedEmbedding
from profirag.config.settings import FASTEMBED_MODEL_DIMENSIONS


class TestFastEmbedEmbeddingInit:
    """Tests for FastEmbedEmbedding initialization"""

    def test_init_with_valid_model(self):
        """Test initialization with valid model name"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        assert embedding.model == "BAAI/bge-small-en-v1.5"
        assert embedding.dimension == 384
        assert embedding._model is None  # Lazy loaded

    def test_init_with_cache_dir(self):
        """Test initialization with custom cache directory"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
            cache_dir="/tmp/fastembed_cache",
        )
        assert embedding.cache_dir == "/tmp/fastembed_cache"

    def test_class_name(self):
        """Test class_name method returns correct name"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        assert embedding.class_name() == "FastEmbedEmbedding"


class TestFastEmbedEmbeddingLoadModel:
    """Tests for model loading"""

    @patch('fastembed.TextEmbedding')
    def test_load_model_success(self, mock_text_embedding):
        """Test successful model loading"""
        mock_instance = Mock()
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        model = embedding._load_model()

        mock_text_embedding.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=None
        )
        assert model == mock_instance

    @patch('fastembed.TextEmbedding')
    def test_load_model_cached(self, mock_text_embedding):
        """Test that model is cached after first load"""
        mock_instance = Mock()
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        # First load
        embedding._load_model()
        # Second load (should use cached)
        embedding._load_model()

        # Should only create once
        mock_text_embedding.assert_called_once()

    def test_load_model_import_error(self):
        """Test error when fastembed not installed"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with patch.dict('sys.modules', {'fastembed': None}):
            with pytest.raises(ImportError) as exc_info:
                embedding._load_model()

            assert "fastembed package not installed" in str(exc_info.value)

    @patch('fastembed.TextEmbedding')
    def test_load_model_invalid_model(self, mock_text_embedding):
        """Test error for invalid model name"""
        mock_text_embedding.side_effect = ValueError("Invalid model")
        mock_text_embedding.list_supported_models = Mock(return_value=[
            {"model": "BAAI/bge-small-en-v1.5"},
            {"model": "BAAI/bge-base-en-v1.5"},
        ])

        embedding = FastEmbedEmbedding(
            model="invalid-model-name",
            dimension=384,
        )

        with pytest.raises(ValueError) as exc_info:
            embedding._load_model()

        assert "Invalid FastEmbed model" in str(exc_info.value)
        assert "invalid-model-name" in str(exc_info.value)

    @patch('fastembed.TextEmbedding')
    def test_load_model_runtime_error(self, mock_text_embedding):
        """Test error for unexpected model loading failure"""
        mock_text_embedding.side_effect = RuntimeError("Network error")

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with pytest.raises(RuntimeError) as exc_info:
            embedding._load_model()

        assert "Failed to load FastEmbed model" in str(exc_info.value)


class TestFastEmbedEmbeddingMethods:
    """Tests for embedding methods"""

    @patch('fastembed.TextEmbedding')
    def test_get_embedding_single(self, mock_text_embedding):
        """Test single text embedding"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embedding("test text")

        assert len(result) == 384
        assert result == [0.1] * 384

    def test_get_embedding_empty_text(self):
        """Test embedding for empty text returns zero vector"""
        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embedding("")

        assert len(result) == 384
        assert result == [0.0] * 384

    @patch('fastembed.TextEmbedding')
    def test_get_embeddings_batch(self, mock_text_embedding):
        """Test batch embedding for multiple texts"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embeddings(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == 384
        assert len(result[1]) == 384
        assert result[0] == [0.1] * 384
        assert result[1] == [0.2] * 384

    @patch('fastembed.TextEmbedding')
    def test_get_embeddings_with_empty_texts(self, mock_text_embedding):
        """Test batch embedding handles empty texts with zero vectors"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384  # Placeholder
        mock_embeddings[2].tolist.return_value = [0.3] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_embeddings(["text one", "", "text three"])

        assert len(result) == 3
        # Empty text should have zero vector
        assert result[1] == [0.0] * 384

    @patch('fastembed.TextEmbedding')
    def test_get_query_embedding(self, mock_text_embedding):
        """Test _get_query_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_query_embedding("query text")

        assert len(result) == 384

    @patch('fastembed.TextEmbedding')
    def test_get_text_embedding(self, mock_text_embedding):
        """Test _get_text_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_text_embedding("document text")

        assert len(result) == 384

    @patch('fastembed.TextEmbedding')
    def test_get_text_embeddings(self, mock_text_embedding):
        """Test _get_text_embeddings batch method"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = embedding._get_text_embeddings(["text 1", "text 2"])

        assert len(result) == 2
        assert len(result[0]) == 384


class TestFastEmbedEmbeddingAsync:
    """Tests for async embedding methods"""

    @pytest.mark.asyncio
    @patch('fastembed.TextEmbedding')
    async def test_aget_query_embedding(self, mock_text_embedding):
        """Test async _aget_query_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_query_embedding("query text")

        assert len(result) == 384

    @pytest.mark.asyncio
    @patch('fastembed.TextEmbedding')
    async def test_aget_text_embedding(self, mock_text_embedding):
        """Test async _aget_text_embedding method"""
        mock_instance = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_instance.embed.return_value = iter([mock_embedding])
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_text_embedding("document text")

        assert len(result) == 384

    @pytest.mark.asyncio
    @patch('fastembed.TextEmbedding')
    async def test_aget_text_embeddings(self, mock_text_embedding):
        """Test async _aget_text_embeddings batch method"""
        mock_instance = Mock()
        mock_embeddings = [Mock(), Mock()]
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        mock_embeddings[1].tolist.return_value = [0.2] * 384
        mock_instance.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
        result = await embedding._aget_text_embeddings(["text 1", "text 2"])

        assert len(result) == 2
        assert len(result[0]) == 384


class TestFastEmbedEmbeddingErrorHandling:
    """Tests for error handling"""

    @patch('fastembed.TextEmbedding')
    def test_embedding_runtime_error(self, mock_text_embedding):
        """Test error handling when embedding fails"""
        mock_instance = Mock()
        mock_instance.embed.side_effect = RuntimeError("GPU memory error")
        mock_text_embedding.return_value = mock_instance

        embedding = FastEmbedEmbedding(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )

        with pytest.raises(RuntimeError) as exc_info:
            embedding._get_embedding("test text")

        assert "FastEmbed embedding failed" in str(exc_info.value)