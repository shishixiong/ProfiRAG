"""Integration tests for FastEmbed with RAGPipeline"""

import pytest
import os
from unittest.mock import patch, Mock

from profirag.config.settings import RAGConfig, EmbeddingConfig, StorageConfig
from profirag.embedding import FastEmbedEmbedding
from profirag.pipeline.rag_pipeline import RAGPipeline


# Skip tests if fastembed not installed or if network unavailable
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_FASTEMBED_TESTS", "true").lower() == "true",
    reason="FastEmbed integration tests skipped (set SKIP_FASTEMBED_TESTS=false to run)"
)


class TestFastEmbedPipelineIntegration:
    """Integration tests for FastEmbed with full pipeline"""

    @pytest.fixture
    def fastembed_config(self):
        """Create RAGConfig with FastEmbed provider"""
        return RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_fastembed_storage",
                    "collection_name": "test_fastembed",
                    "dimension": 384,
                }
            ),
            embedding=EmbeddingConfig(
                provider="fastembed",
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            ),
        )

    def test_pipeline_creates_fastembed_embedding(self, fastembed_config):
        """Test that pipeline creates FastEmbedEmbedding when provider is fastembed"""
        pipeline = RAGPipeline(fastembed_config)
        assert isinstance(pipeline._embed_model, FastEmbedEmbedding)
        assert pipeline._embed_model.model == "BAAI/bge-small-en-v1.5"
        assert pipeline._embed_model.dimension == 384

    def test_pipeline_embed_model_dimension(self, fastembed_config):
        """Test that embedding dimension is correct"""
        pipeline = RAGPipeline(fastembed_config)
        # Dimension should match config
        assert pipeline._embed_model.dimension == 384

    def test_pipeline_get_stats_shows_fastembed(self, fastembed_config):
        """Test that pipeline stats show FastEmbed as embedding provider"""
        pipeline = RAGPipeline(fastembed_config)
        stats = pipeline.get_stats()

        assert stats["embedding"]["model"] == "BAAI/bge-small-en-v1.5"
        assert stats["embedding"]["dimension"] == 384


class TestFastEmbedProviderSelection:
    """Tests for provider selection logic"""

    def test_openai_provider_creates_custom_embedding(self):
        """Test that openai provider creates CustomOpenAIEmbedding"""
        from profirag.embedding import CustomOpenAIEmbedding

        config = RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_openai_storage",
                    "collection_name": "test_openai",
                    "dimension": 1536,
                }
            ),
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                api_key="test-key",
            ),
        )
        pipeline = RAGPipeline(config)
        assert isinstance(pipeline._embed_model, CustomOpenAIEmbedding)

    def test_fastembed_provider_creates_fastembed_embedding(self):
        """Test that fastembed provider creates FastEmbedEmbedding"""
        config = RAGConfig(
            storage=StorageConfig(
                type="local",
                config={
                    "persist_path": "/tmp/test_fastembed_storage2",
                    "collection_name": "test_fastembed2",
                    "dimension": 384,
                }
            ),
            embedding=EmbeddingConfig(
                provider="fastembed",
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            ),
        )
        pipeline = RAGPipeline(config)
        assert isinstance(pipeline._embed_model, FastEmbedEmbedding)


class TestFastEmbedConfigFromEnv:
    """Tests for loading FastEmbed config from environment"""

    @patch.dict(os.environ, {
        "PROFIRAG_STORAGE_TYPE": "local",
        "LOCAL_STORAGE_PATH": "/tmp/test_env_storage",
        "LOCAL_COLLECTION_NAME": "test_env",
        "PROFIRAG_EMBEDDING_PROVIDER": "fastembed",
        "PROFIRAG_EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
    }, clear=False)
    def test_config_from_env_fastembed(self):
        """Test loading FastEmbed config from environment variables"""
        config = RAGConfig.from_env()

        assert config.embedding.provider == "fastembed"
        assert config.embedding.model == "BAAI/bge-base-en-v1.5"
        # Dimension should be auto-detected (768 for bge-base)
        assert config.embedding.dimension == 768

    @patch.dict(os.environ, {
        "PROFIRAG_EMBEDDING_PROVIDER": "fastembed",
        "PROFIRAG_EMBEDDING_MODEL": "unknown-model",
        "PROFIRAG_EMBEDDING_DIMENSION": "512",
    }, clear=False)
    def test_config_dimension_override(self):
        """Test dimension override for unknown models"""
        config = RAGConfig.from_env()

        assert config.embedding.dimension == 512

    @patch.dict(os.environ, {
        "PROFIRAG_EMBEDDING_PROVIDER": "openai",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_EMBEDDING_DIMENSION": "1536",
    }, clear=False)
    def test_config_openai_provider_default(self):
        """Test that openai remains default provider"""
        config = RAGConfig.from_env()

        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.embedding.dimension == 1536