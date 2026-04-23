"""Tests for rerank configuration"""

import os
from profirag.config.settings import EnvSettings, RerankingConfig


def test_env_settings_rerank_provider_default():
    """Test that rerank_provider defaults to 'local'."""
    original_provider = os.environ.pop("PROFIRAG_RERANK_PROVIDER", None)
    original_key = os.environ.pop("PROFIRAG_RERANK_API_KEY", None)
    original_url = os.environ.pop("PROFIRAG_RERANK_BASE_URL", None)
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_provider == "local"
    if original_provider:
        os.environ["PROFIRAG_RERANK_PROVIDER"] = original_provider
    if original_key:
        os.environ["PROFIRAG_RERANK_API_KEY"] = original_key
    if original_url:
        os.environ["PROFIRAG_RERANK_BASE_URL"] = original_url


def test_env_settings_rerank_provider_values():
    """Test that rerank_provider accepts valid values."""
    for provider in ["local", "cohere", "dashscope"]:
        os.environ["PROFIRAG_RERANK_PROVIDER"] = provider
        settings = EnvSettings(_env_file=None)
        assert settings.profirag_rerank_provider == provider
        del os.environ["PROFIRAG_RERANK_PROVIDER"]


def test_env_settings_rerank_api_key():
    """Test that rerank_api_key can be set."""
    os.environ["PROFIRAG_RERANK_API_KEY"] = "test-api-key"
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_api_key == "test-api-key"
    del os.environ["PROFIRAG_RERANK_API_KEY"]


def test_env_settings_rerank_base_url():
    """Test that rerank_base_url can be set."""
    os.environ["PROFIRAG_RERANK_BASE_URL"] = "https://api.example.com"
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_base_url == "https://api.example.com"
    del os.environ["PROFIRAG_RERANK_BASE_URL"]


def test_env_settings_rerank_timeout():
    """Test that rerank_timeout defaults to 30."""
    original = os.environ.pop("PROFIRAG_RERANK_TIMEOUT", None)
    settings = EnvSettings(_env_file=None)
    assert settings.profirag_rerank_timeout == 30
    if original:
        os.environ["PROFIRAG_RERANK_TIMEOUT"] = original


def test_reranking_config_provider_field():
    """Test RerankingConfig has provider field."""
    config = RerankingConfig(provider="cohere")
    assert config.provider == "cohere"


def test_reranking_config_api_key_field():
    """Test RerankingConfig has api_key field."""
    config = RerankingConfig(api_key="test-key")
    assert config.api_key == "test-key"


def test_reranking_config_base_url_field():
    """Test RerankingConfig has base_url field."""
    config = RerankingConfig(base_url="https://api.example.com")
    assert config.base_url == "https://api.example.com"


def test_reranking_config_timeout_field():
    """Test RerankingConfig has timeout field."""
    config = RerankingConfig(timeout=60)
    assert config.timeout == 60