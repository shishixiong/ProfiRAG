"""Tests for retrieve_mode configuration"""

from profirag.config.settings import EnvSettings, RetrievalConfig, RAGConfig


def test_env_settings_retrieve_mode_default():
    """Test that retrieve_mode defaults to 'hybrid'."""
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "hybrid"


def test_env_settings_retrieve_mode_values():
    """Test that retrieve_mode accepts valid values."""
    # Test via environment variable simulation
    import os
    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "sparse"
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "sparse"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]

    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "vector"
    settings = EnvSettings()
    assert settings.profirag_retrieve_index_mode == "vector"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]


def test_retrieval_config_retrieve_mode():
    """Test RetrievalConfig has retrieve_mode field."""
    config = RetrievalConfig(retrieve_mode="sparse")
    assert config.retrieve_mode == "sparse"


def test_rag_config_from_env_includes_retrieve_mode():
    """Test that RAGConfig.from_env passes retrieve_mode."""
    import os
    os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"] = "vector"
    config = RAGConfig.from_env()
    assert config.retrieval.retrieve_mode == "vector"
    del os.environ["PROFIRAG_RETRIEVE_INDEX_MODE"]