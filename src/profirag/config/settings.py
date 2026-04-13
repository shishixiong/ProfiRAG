"""Configuration management using Pydantic with .env support"""

import os
from pathlib import Path
from typing import Dict, Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Environment settings loaded from .env file"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536
    openai_llm_model: str = "gpt-4-turbo"
    openai_llm_temperature: float = 0.0
    openai_llm_max_tokens: Optional[int] = None

    # Storage Configuration
    profirag_storage_type: Literal["qdrant", "local", "postgres"] = "qdrant"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "profirag"
    qdrant_url: Optional[str] = None

    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "profirag"
    postgres_user: str = "postgres"
    postgres_password: Optional[str] = None

    # Local Storage Configuration
    local_storage_path: str = "./storage"
    local_collection_name: str = "default"

    # Retrieval Configuration
    profirag_top_k: int = 10
    profirag_alpha: float = 0.5
    profirag_use_hybrid: bool = True
    profirag_use_bm25: bool = True

    # Reranking Configuration
    profirag_rerank_enabled: bool = True
    profirag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    profirag_rerank_top_n: int = 5

    # Pre-Retrieval Configuration
    profirag_use_hyde: bool = False
    profirag_use_rewrite: bool = False
    profirag_multi_query: bool = False


class StorageConfig(BaseModel):
    """Vector store configuration"""
    type: Literal["qdrant", "local", "postgres"] = "qdrant"
    config: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """OpenAI Embedding configuration"""
    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    api_key: Optional[str] = None


class LLMConfig(BaseModel):
    """OpenAI LLM configuration"""
    provider: Literal["openai"] = "openai"
    model: str = "gpt-4-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class PreRetrievalConfig(BaseModel):
    """Pre-retrieval configuration"""
    use_hyde: bool = False
    use_rewrite: bool = False
    multi_query: bool = False
    hyde_prompt: Optional[str] = None


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = 10
    alpha: float = 0.5  # Vector search weight (1-alpha for BM25)
    use_hybrid: bool = True
    use_bm25: bool = True


class RerankingConfig(BaseModel):
    """Reranking configuration"""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class GenerationConfig(BaseModel):
    """Generation configuration"""
    response_mode: str = "compact"
    streaming: bool = False


class RAGConfig(BaseModel):
    """Complete RAG configuration"""
    storage: StorageConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    pre_retrieval: PreRetrievalConfig = PreRetrievalConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    generation: GenerationConfig = GenerationConfig()

    class Config:
        extra = "allow"

    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Load configuration from YAML file"""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "RAGConfig":
        """Load configuration from .env file and environment variables.

        Args:
            env_file: Path to .env file (default: ".env" in current directory)

        Returns:
            RAGConfig instance
        """
        # Determine env file path
        if env_file:
            env_path = Path(env_file)
        else:
            env_path = Path.cwd() / ".env"

        # Load environment settings
        env_settings = EnvSettings(
            _env_file=env_path if env_path.exists() else None
        )

        # Build storage config based on type
        storage_type = env_settings.profirag_storage_type
        storage_config = cls._build_storage_config(env_settings, storage_type)

        return cls(
            storage=StorageConfig(type=storage_type, config=storage_config),
            embedding=EmbeddingConfig(
                provider="openai",
                model=env_settings.openai_embedding_model,
                dimension=env_settings.openai_embedding_dimension,
                api_key=env_settings.openai_api_key,
            ),
            llm=LLMConfig(
                provider="openai",
                model=env_settings.openai_llm_model,
                api_key=env_settings.openai_api_key,
                temperature=env_settings.openai_llm_temperature,
                max_tokens=env_settings.openai_llm_max_tokens,
            ),
            pre_retrieval=PreRetrievalConfig(
                use_hyde=env_settings.profirag_use_hyde,
                use_rewrite=env_settings.profirag_use_rewrite,
                multi_query=env_settings.profirag_multi_query,
            ),
            retrieval=RetrievalConfig(
                top_k=env_settings.profirag_top_k,
                alpha=env_settings.profirag_alpha,
                use_hybrid=env_settings.profirag_use_hybrid,
                use_bm25=env_settings.profirag_use_bm25,
            ),
            reranking=RerankingConfig(
                enabled=env_settings.profirag_rerank_enabled,
                model=env_settings.profirag_rerank_model,
                top_n=env_settings.profirag_rerank_top_n,
            ),
        )

    @staticmethod
    def _build_storage_config(env: EnvSettings, storage_type: str) -> Dict[str, Any]:
        """Build storage configuration dictionary based on storage type.

        Args:
            env: Environment settings
            storage_type: Storage backend type

        Returns:
            Storage configuration dictionary
        """
        if storage_type == "qdrant":
            config = {
                "host": env.qdrant_host,
                "port": env.qdrant_port,
                "collection_name": env.qdrant_collection_name,
                "dimension": env.openai_embedding_dimension,
            }
            if env.qdrant_api_key:
                config["api_key"] = env.qdrant_api_key
            if env.qdrant_url:
                config["url"] = env.qdrant_url
            return config

        elif storage_type == "postgres":
            config = {
                "host": env.postgres_host,
                "port": env.postgres_port,
                "database": env.postgres_database,
                "user": env.postgres_user,
                "table_name": env.postgres_database,
                "dimension": env.openai_embedding_dimension,
            }
            if env.postgres_password:
                config["password"] = env.postgres_password
            return config

        elif storage_type == "local":
            return {
                "persist_path": env.local_storage_path,
                "collection_name": env.local_collection_name,
                "dimension": env.openai_embedding_dimension,
            }

        else:
            return {}


def load_config(env_file: Optional[str] = None) -> RAGConfig:
    """Load RAG configuration from .env file.

    This is a convenience function that wraps RAGConfig.from_env().

    Args:
        env_file: Path to .env file (default: ".env" in current directory)

    Returns:
        RAGConfig instance
    """
    return RAGConfig.from_env(env_file)