"""Configuration management using Pydantic with .env support"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata

from .credentials import KeyringCredentialStore

logger = logging.getLogger(__name__)


# FastEmbed model dimension mapping for auto-detection
FASTEMBED_MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class EnvSettings(BaseSettings):
    """Environment settings loaded from .env file"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # Custom API endpoint for LLM
    openai_auth_token: Optional[str] = None  # X-Auth-Token for custom API LLM
    openai_auth_user: Optional[str] = None  # Username for auto-refresh token
    openai_auth_password: Optional[str] = None  # Password for auto-refresh token (prefer keyring)
    openai_auth_credential_store: Literal["env", "keyring"] = "keyring"  # Where to read password from
    openai_auth_token_ttl: int = 7200  # Token TTL in seconds before auto-refresh
    openai_llm_provider: Literal["openai", "custom_api"] = "openai"
    openai_llm_verify_ssl: bool = False  # Verify SSL for custom API LLM
    openai_embedding_api_key: Optional[str] = None  # Fallback to openai_api_key if not set
    openai_embedding_base_url: Optional[str] = None  # Custom API endpoint for Embedding, fallback to openai_base_url
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536
    openai_llm_model: str = "gpt-4-turbo"
    openai_llm_temperature: float = 0.0
    openai_llm_max_tokens: Optional[int] = None

    # Embedding Provider Configuration
    profirag_embedding_provider: Literal["openai", "fastembed"] = "openai"
    profirag_embedding_model: str = "BAAI/bge-small-en-v1.5"
    profirag_embedding_dimension: Optional[int] = None  # Auto-detected for FastEmbed
    profirag_embedding_cache_dir: Optional[str] = None

    # MiniMax Vision Configuration (for image understanding)
    minimax_api_key: Optional[str] = None
    minimax_api_host: str = "https://api.minimax.chat"

    # Image Processing Configuration
    profirag_image_processing_enabled: bool = True
    profirag_generate_image_descriptions: bool = True
    profirag_image_storage_path: str = "./images"
    profirag_image_description_prompt: str = "描述这张图片的内容，包括图片中的文字、图形、图表等关键信息"
    profirag_image_provider: Literal["minimax", "openai"] = "minimax"  # Image understanding provider
    profirag_image_openai_api_key: Optional[str] = None  # Fallback to openai_api_key if not set
    profirag_image_openai_base_url: Optional[str] = None  # Fallback to openai_base_url if not set
    profirag_image_openai_model: str = "gpt-4o"  # Vision model for OpenAI provider
    profirag_image_timeout: int = 60  # Image understanding API timeout

    # Storage Configuration
    profirag_storage_type: Literal["qdrant", "local", "postgres"] = "qdrant"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "profirag"
    qdrant_url: Optional[str] = None

    # Dense Vector Configuration
    # profirag_dense_vector_name is deprecated - index_mode controls this

    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "profirag"
    postgres_user: str = "postgres"
    postgres_password: Optional[str] = None

    # Local Storage Configuration
    local_storage_path: str = "./storage"
    local_collection_name: str = "default"

    # Chunking Configuration
    profirag_splitter_type: Literal["sentence", "token", "semantic", "chinese"] = "sentence"
    profirag_chunk_size: int = 512
    profirag_chunk_overlap: int = 50
    profirag_language: Literal["en", "zh"] = "en"

    # Retrieval Configuration
    profirag_top_k: int = 10
    profirag_alpha: float = 0.5
    profirag_use_hybrid: bool = True
    profirag_index_mode: Literal["hybrid", "vector"] = "hybrid"
    profirag_retrieve_index_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"

    # Reranking Configuration
    profirag_rerank_enabled: bool = True
    profirag_rerank_provider: Literal["local", "cohere", "dashscope"] = "local"
    profirag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    profirag_rerank_top_n: int = 5
    profirag_rerank_api_key: Optional[str] = None
    profirag_rerank_base_url: Optional[str] = None
    profirag_rerank_timeout: int = 30

    # Pre-Retrieval Configuration
    profirag_use_hyde: bool = False
    profirag_use_rewrite: bool = False
    profirag_multi_query: bool = False

    # Agent Configuration
    profirag_agent_enabled: bool = False
    profirag_agent_mode: str = "react"
    profirag_agent_max_iterations: int = 10
    profirag_agent_verbose: bool = True
    profirag_agent_markdown_base_path: Optional[str] = None


class CustomOpenAILLM(OpenAI):
    """Custom OpenAI LLM that bypasses model name validation.

    Allows using custom model names (like MiniMax-M2.7, DeepSeek, etc.)
    with OpenAI-compatible APIs.

    Example:
        >>> llm = CustomOpenAILLM(
        >>>     model="MiniMax-M2.7",
        >>>     api_key="your-api-key",
        >>>     api_base="https://api.minimax.chat/v1",
        >>> )
    """

    @property
    def metadata(self) -> LLMMetadata:
        """Override metadata to bypass model validation."""
        model_dict = self.model_dump()

        return LLMMetadata(
            context_window=128000,
            num_output=model_dict.get('max_tokens') or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=model_dict.get('model', 'unknown'),
        )


class TokenRefresher:
    """Auto-refresh X-Auth-Token via secureLogin API when expired."""

    LOGIN_URL = "http://rnd-idea-api.huawei.com/ideaclientservice/login/v4/secureLogin"

    def __init__(
        self,
        user: str,
        password: str,
        token: Optional[str] = None,
        token_ttl_seconds: int = 7200,
    ):
        self.user = user
        self.password = password
        self._token = token
        self._token_time: Optional[float] = None
        self._token_ttl = token_ttl_seconds
        if token:
            self._token_time = __import__("time").time()

    @property
    def token(self) -> Optional[str]:
        if self._is_expired():
            self._refresh()
        return self._token

    def _is_expired(self) -> bool:
        if self._token is None or self._token_time is None:
            return True
        import time
        return (time.time() - self._token_time) >= self._token_ttl

    def _refresh(self) -> None:
        import httpx
        import logging
        logger = logging.getLogger(__name__)

        try:
            resp = httpx.post(
                self.LOGIN_URL,
                json={"user": self.user, "password": self.password, "ideName": "IntelliJ IDE"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            dragon = data.get("cloudDragonTokens", {})
            new_token = dragon.get("authToken")
            if new_token and dragon.get("valid"):
                self._token = new_token
                import time
                self._token_time = time.time()
                logger.info("X-Auth-Token refreshed successfully")
            else:
                logger.warning(f"Token refresh failed: cloudDragonTokens invalid or missing authToken")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")

    def force_refresh(self) -> Optional[str]:
        self._refresh()
        return self._token


class CustomAPILLM(OpenAI):
    """Custom LLM for APIs that use X-Auth-Token authentication.

    Supports OpenAI-compatible chat/completions endpoints that authenticate
    via X-Auth-Token header instead of Bearer token.
    Supports automatic token refresh when expired.
    """

    auth_token: Optional[str] = Field(
        default=None, description="X-Auth-Token for API authentication."
    )
    auth_user: Optional[str] = Field(
        default=None, description="Username for auto-refresh token."
    )
    auth_password: Optional[str] = Field(
        default=None, description="Password for auto-refresh token."
    )
    auth_token_ttl: int = Field(
        default=7200, description="Token TTL in seconds before auto-refresh."
    )
    verify_ssl: bool = Field(
        default=False, description="Whether to verify SSL certificates."
    )
    _token_refresher: Optional[TokenRefresher] = PrivateAttr(default=None)

    def __init__(
        self,
        auth_token: Optional[str] = None,
        auth_user: Optional[str] = None,
        auth_password: Optional[str] = None,
        auth_token_ttl: int = 7200,
        verify_ssl: bool = False,
        **kwargs: Any,
    ):
        import httpx

        refresher = None
        if auth_user and auth_password:
            refresher = TokenRefresher(
                user=auth_user,
                password=auth_password,
                token=auth_token,
                token_ttl_seconds=auth_token_ttl,
            )
            if not auth_token:
                auth_token = refresher.token

        if auth_token:
            default_headers = kwargs.pop("default_headers", None) or {}
            default_headers["X-Auth-Token"] = auth_token
            kwargs["default_headers"] = default_headers

        kwargs.setdefault("api_key", "not-needed")

        ssl_http_client = httpx.Client(verify=verify_ssl)
        ssl_async_http_client = httpx.AsyncClient(verify=verify_ssl)
        kwargs["http_client"] = ssl_http_client
        kwargs["async_http_client"] = ssl_async_http_client

        super().__init__(**kwargs)
        self.auth_token = auth_token
        self.auth_user = auth_user
        self.auth_password = auth_password
        self.auth_token_ttl = auth_token_ttl
        self.verify_ssl = verify_ssl
        self._token_refresher = refresher

    def _get_current_token(self) -> Optional[str]:
        if self._token_refresher:
            return self._token_refresher.token
        return self.auth_token

    def _get_client(self):
        token = self._get_current_token()
        if token and token != self.auth_token:
            self.auth_token = token
            self.default_headers = {**(self.default_headers or {}), "X-Auth-Token": token}
            self._client = None
        return super()._get_client()

    def _get_aclient(self):
        token = self._get_current_token()
        if token and token != self.auth_token:
            self.auth_token = token
            self.default_headers = {**(self.default_headers or {}), "X-Auth-Token": token}
            self._aclient = None
        return super()._get_aclient()

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        kwargs = super()._get_credential_kwargs(is_async=is_async)
        token = self._get_current_token()
        if token:
            headers = kwargs.get("default_headers") or {}
            headers["X-Auth-Token"] = token
            kwargs["default_headers"] = headers
        return kwargs

    @property
    def metadata(self) -> LLMMetadata:
        model_dict = self.model_dump()
        return LLMMetadata(
            context_window=128000,
            num_output=model_dict.get('max_tokens') or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=model_dict.get('model', 'unknown'),
        )


class StorageConfig(BaseModel):
    """Vector store configuration"""
    type: Literal["qdrant", "local", "postgres"] = "qdrant"
    config: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding configuration supporting OpenAI and FastEmbed providers"""
    provider: Literal["openai", "fastembed"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cache_dir: Optional[str] = None  # For FastEmbed model cache


class LLMConfig(BaseModel):
    """OpenAI LLM configuration"""
    provider: Literal["openai", "custom_api"] = "openai"
    model: str = "gpt-4-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    auth_token: Optional[str] = None
    auth_user: Optional[str] = None
    auth_password: Optional[str] = None
    auth_token_ttl: int = 7200
    verify_ssl: bool = False
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class PreRetrievalConfig(BaseModel):
    """Pre-retrieval configuration"""
    use_hyde: bool = False
    use_rewrite: bool = False
    multi_query: bool = False
    hyde_prompt: Optional[str] = None


class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    splitter_type: Literal["sentence", "token", "semantic", "chinese", "ast", "markdown"] = "sentence"
    chunk_size: int = 512
    chunk_overlap: int = 50
    language: Literal["en", "zh"] = "en"

    # AST-specific settings
    ast_language: Literal["python", "java", "cpp", "go"] = "python"


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = 10
    alpha: float = 0.5  # Vector search weight (1-alpha for BM25)
    retrieve_mode: Literal["hybrid", "sparse", "vector"] = "hybrid"


class RerankingConfig(BaseModel):
    """Reranking configuration"""
    enabled: bool = True
    provider: Literal["local", "cohere", "dashscope"] = "local"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30


class GenerationConfig(BaseModel):
    """Generation configuration"""
    response_mode: str = "compact"
    streaming: bool = False


class ImageProcessingConfig(BaseModel):
    """Image processing configuration for PDF image handling"""
    enabled: bool = True
    generate_descriptions: bool = True
    storage_path: str = "./images"
    description_prompt: str = "描述这张图片的内容，包括图片中的文字、图形、图表等关键信息"
    minimax_api_key: Optional[str] = None
    minimax_api_host: str = "https://api.minimax.chat"


class PlanAgentConfig(BaseModel):
    """PlanAgent specific configuration"""
    require_approval: bool = True       # 计划确认
    max_replan_attempts: int = 3        # 失败重规划上限
    show_plan: bool = True              # 显示计划
    verbose_steps: bool = True          # 详细日志
    auto_approve_simple: bool = True    # 简单问题自动批准


class ConversationConfig(BaseModel):
    """ConversationManager configuration."""
    max_history_turns: int = 6      # Turns before summarization
    keep_recent_turns: int = 2      # Turns kept verbatim after summarization
    auto_context: bool = True       # Enable LLM-based context decision


class AgentConfig(BaseModel):
    """Agent configuration for ReAct-based question answering"""
    enabled: bool = False  # 默认关闭，使用Pipeline模式
    mode: str = "react"  # "react", "plan", or "pipeline"
    max_iterations: int = 10
    verbose: bool = True
    markdown_base_path: Optional[str] = None  # Markdown文件目录路径（用于表格索引解析）
    # PlanAgent 配置
    plan_config: PlanAgentConfig = PlanAgentConfig()
    # Conversation 配置
    conversation_config: ConversationConfig = ConversationConfig()
    # 可用的工具列表
    tools: List[str] = [
        "vector_search",
        "keyword_search",
        "multi_query_search",
        "hyde_search",
        "rewrite_query",
        "rerank_results",
        "filter_results",
        "generate_answer",
        "retrieve_and_answer",
    ]


class RAGConfig(BaseModel):
    """Complete RAG configuration"""
    storage: StorageConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    pre_retrieval: PreRetrievalConfig = PreRetrievalConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    generation: GenerationConfig = GenerationConfig()
    image_processing: ImageProcessingConfig = ImageProcessingConfig()
    agent: AgentConfig = AgentConfig()  # Agent配置

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

        # Build embedding config based on provider
        if env_settings.profirag_embedding_provider == "fastembed":
            model = env_settings.profirag_embedding_model
            dimension = env_settings.profirag_embedding_dimension or FASTEMBED_MODEL_DIMENSIONS.get(model, 768)
        else:
            model = env_settings.openai_embedding_model
            dimension = env_settings.openai_embedding_dimension

        auth_password = env_settings.openai_auth_password
        if not auth_password and env_settings.openai_auth_credential_store == "keyring":
            auth_password = KeyringCredentialStore.load("auth_password")
            if auth_password:
                logger.info("Loaded auth_password from OS keyring")
            else:
                logger.warning("Keyring credential store enabled but 'auth_password' not found in keyring. "
                               "Run: python -m profirag.config.credentials set auth_password")

        return cls(
            storage=StorageConfig(type=storage_type, config=storage_config),
            embedding=EmbeddingConfig(
                provider=env_settings.profirag_embedding_provider,
                model=model,
                dimension=dimension,
                api_key=env_settings.openai_embedding_api_key or env_settings.openai_api_key if env_settings.profirag_embedding_provider == "openai" else None,
                base_url=env_settings.openai_embedding_base_url or env_settings.openai_base_url if env_settings.profirag_embedding_provider == "openai" else None,
                cache_dir=env_settings.profirag_embedding_cache_dir,
            ),
            llm=LLMConfig(
                provider=env_settings.openai_llm_provider,
                model=env_settings.openai_llm_model,
                api_key=env_settings.openai_api_key,
                base_url=env_settings.openai_base_url,
                auth_token=env_settings.openai_auth_token,
                auth_user=env_settings.openai_auth_user,
                auth_password=auth_password,
                auth_token_ttl=env_settings.openai_auth_token_ttl,
                verify_ssl=env_settings.openai_llm_verify_ssl,
                temperature=env_settings.openai_llm_temperature,
                max_tokens=env_settings.openai_llm_max_tokens,
            ),
            chunking=ChunkingConfig(
                splitter_type=env_settings.profirag_splitter_type,
                chunk_size=env_settings.profirag_chunk_size,
                chunk_overlap=env_settings.profirag_chunk_overlap,
                language=env_settings.profirag_language,
            ),
            pre_retrieval=PreRetrievalConfig(
                use_hyde=env_settings.profirag_use_hyde,
                use_rewrite=env_settings.profirag_use_rewrite,
                multi_query=env_settings.profirag_multi_query,
            ),
            retrieval=RetrievalConfig(
                top_k=env_settings.profirag_top_k,
                alpha=env_settings.profirag_alpha,
                retrieve_mode=env_settings.profirag_retrieve_index_mode,
            ),
            reranking=RerankingConfig(
                enabled=env_settings.profirag_rerank_enabled,
                provider=env_settings.profirag_rerank_provider,
                model=env_settings.profirag_rerank_model,
                top_n=env_settings.profirag_rerank_top_n,
                api_key=env_settings.profirag_rerank_api_key,
                base_url=env_settings.profirag_rerank_base_url,
                timeout=env_settings.profirag_rerank_timeout,
            ),
            image_processing=ImageProcessingConfig(
                enabled=env_settings.profirag_image_processing_enabled,
                generate_descriptions=env_settings.profirag_generate_image_descriptions,
                storage_path=env_settings.profirag_image_storage_path,
                description_prompt=env_settings.profirag_image_description_prompt,
                minimax_api_key=env_settings.minimax_api_key,
                minimax_api_host=env_settings.minimax_api_host,
            ),
            agent=AgentConfig(
                enabled=env_settings.profirag_agent_enabled,
                mode=env_settings.profirag_agent_mode,
                max_iterations=env_settings.profirag_agent_max_iterations,
                verbose=env_settings.profirag_agent_verbose,
                markdown_base_path=env_settings.profirag_agent_markdown_base_path,
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
                "index_mode": env.profirag_index_mode,
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
