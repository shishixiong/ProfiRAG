"""Unit tests for Ollama embedding provider configuration"""

import tempfile
from pathlib import Path

from profirag.config.settings import EmbeddingConfig, EnvSettings, RAGConfig


class TestOllamaEnvSettings:
    """Tests for Ollama settings in EnvSettings"""

    def test_ollama_default_model(self):
        """Test default Ollama model is nomic-embed-text"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_embedding_model == "nomic-embed-text"

    def test_ollama_default_dimension(self):
        """Test default Ollama dimension is 768"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_embedding_dimension == 768

    def test_ollama_default_base_url(self):
        """Test default Ollama base URL"""
        env = EnvSettings(profirag_embedding_provider="ollama")
        assert env.ollama_base_url == "http://localhost:11434/v1"

    def test_ollama_custom_model(self):
        """Test custom Ollama model override"""
        env = EnvSettings(
            profirag_embedding_provider="ollama", ollama_embedding_model="mxbai-embed-large"
        )
        assert env.ollama_embedding_model == "mxbai-embed-large"

    def test_ollama_custom_dimension(self):
        """Test custom Ollama dimension override"""
        env = EnvSettings(profirag_embedding_provider="ollama", ollama_embedding_dimension=1024)
        assert env.ollama_embedding_dimension == 1024

    def test_ollama_custom_base_url(self):
        """Test custom Ollama base URL override"""
        env = EnvSettings(
            profirag_embedding_provider="ollama", ollama_base_url="http://192.168.1.100:11434/v1"
        )
        assert env.ollama_base_url == "http://192.168.1.100:11434/v1"


class TestOllamaEmbeddingConfig:
    """Tests for EmbeddingConfig with Ollama provider"""

    def test_embedding_config_ollama_provider(self):
        """Test EmbeddingConfig accepts ollama as provider"""
        config = EmbeddingConfig(
            provider="ollama",
            model="nomic-embed-text",
            dimension=768,
            base_url="http://localhost:11434/v1",
        )
        assert config.provider == "ollama"
        assert config.model == "nomic-embed-text"
        assert config.dimension == 768
        assert config.base_url == "http://localhost:11434/v1"


class TestOllamaRAGConfigFromEnv:
    """Tests for RAGConfig.from_env() with Ollama provider"""

    def test_from_env_ollama_defaults(self):
        """Test from_env creates correct config with Ollama defaults"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "nomic-embed-text"
        assert config.embedding.dimension == 768
        assert config.embedding.base_url == "http://localhost:11434/v1"
        assert config.embedding.api_key is None

    def test_from_env_ollama_custom_model(self):
        """Test from_env with custom Ollama model"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("OLLAMA_EMBEDDING_MODEL=mxbai-embed-large\n")
            f.write("OLLAMA_EMBEDDING_DIMENSION=1024\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.model == "mxbai-embed-large"
        assert config.embedding.dimension == 1024

    def test_from_env_ollama_custom_base_url(self):
        """Test from_env with custom Ollama base URL"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("PROFIRAG_EMBEDDING_PROVIDER=ollama\n")
            f.write("OLLAMA_BASE_URL=http://remote-server:11434/v1\n")
            f.write("PROFIRAG_STORAGE_TYPE=local\n")
            f.flush()
            env_path = Path(f.name)

        config = RAGConfig.from_env(env_file=str(env_path))
        env_path.unlink()

        assert config.embedding.base_url == "http://remote-server:11434/v1"
