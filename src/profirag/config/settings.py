"""Configuration management using Pydantic"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional


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
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables"""
        import os
        from pydantic_settings import BaseSettings

        class EnvSettings(BaseSettings):
            openai_api_key: Optional[str] = None
            qdrant_host: str = "localhost"
            qdrant_port: int = 6333
            postgres_host: str = "localhost"
            postgres_port: int = 5432
            postgres_database: str = "profirag"

            class Config:
                env_prefix = "PROFIRAG_"

        env = EnvSettings()

        storage_type = os.getenv("PROFIRAG_STORAGE_TYPE", "qdrant")
        storage_config = {}

        if storage_type == "qdrant":
            storage_config = {
                "host": env.qdrant_host,
                "port": env.qdrant_port,
                "collection_name": os.getenv("PROFIRAG_COLLECTION", "profirag"),
            }
        elif storage_type == "postgres":
            storage_config = {
                "host": env.postgres_host,
                "port": env.postgres_port,
                "database": env.postgres_database,
            }

        return cls(
            storage=StorageConfig(type=storage_type, config=storage_config),
            embedding=EmbeddingConfig(api_key=env.openai_api_key),
            llm=LLMConfig(api_key=env.openai_api_key),
        )