"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class SplitterType(str, Enum):
    SENTENCE = "sentence"
    TOKEN = "token"
    SEMANTIC = "semantic"
    CHINESE = "chinese"
    AST = "ast"
    MARKDOWN = "markdown"


class ASTLanguage(str, Enum):
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"


class IndexMode(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"


class OutputFormat(str, Enum):
    TXT = "txt"
    JSON = "json"
    JSONL = "jsonl"


class ChatMode(str, Enum):
    PIPELINE = "pipeline"
    AGENT = "agent"
    PLAN = "plan"


class ConversationRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID to continue")
    continue_session: bool = Field(False, description="Continue existing conversation")


class ConversationInfo(BaseModel):
    session_id: str
    turn_count: int
    injected_context: bool = False
    reference_detected: bool = False


class PdfConvertRequest(BaseModel):
    pages: Optional[str] = Field(None, description="Page range e.g. '1-5,10,15-20'")
    write_images: bool = Field(False, description="Extract images from PDF")
    exclude_header_footer: bool = Field(False, description="Filter header/footer content")
    header_footer_min_occurrences: int = Field(3, description="Min occurrences for auto-detect")
    extract_tables: bool = Field(False, description="Extract tables to separate files")


class PdfConvertResponse(BaseModel):
    file_id: str
    markdown_content: str
    table_files: List[str] = Field(default_factory=list)
    image_files: List[str] = Field(default_factory=list)


class PdfPreviewResponse(BaseModel):
    file_id: str
    original_pages: int
    markdown_preview: str
    tables_count: int


class SplitPreviewRequest(BaseModel):
    file_id: str = Field(..., description="ID of uploaded file")
    splitter_type: SplitterType = SplitterType.SENTENCE
    chunk_size: int = Field(512, ge=50, le=4000)
    chunk_overlap: int = Field(50, ge=0, le=500)
    language: ASTLanguage = ASTLanguage.PYTHON
    output_format: OutputFormat = OutputFormat.JSON


class ChunkMetadata(BaseModel):
    chunk_index: int
    source_file: str
    total_chunks_in_doc: int
    header_path: Optional[str] = None
    current_heading: Optional[str] = None
    has_code_block: bool = False
    has_table: bool = False
    has_images: bool = False
    char_count: int


class ChunkPreview(BaseModel):
    chunk_index: int
    text_preview: str
    metadata: ChunkMetadata


class SplitPreviewResponse(BaseModel):
    file_id: str
    total_chunks: int
    chunks: List[ChunkPreview]
    summary: Dict[str, Any]


class SplitDownloadRequest(BaseModel):
    file_id: str
    output_format: OutputFormat = OutputFormat.JSON


class ImportConfig(BaseModel):
    splitter_type: SplitterType = SplitterType.MARKDOWN
    chunk_size: int = Field(1024, ge=50, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    ast_language: ASTLanguage = ASTLanguage.PYTHON
    index_mode: IndexMode = IndexMode.HYBRID
    env_file: str = Field(".env", description="Path to .env config file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for imported documents")


class ImportStartRequest(BaseModel):
    file_ids: List[str]
    config: ImportConfig


class WikiImportRequest(BaseModel):
    wiki_url: str = Field(..., description="Wiki URL to import")
    splitter_type: SplitterType = SplitterType.MARKDOWN
    chunk_size: int = Field(512, ge=50, le=4000)
    chunk_overlap: int = Field(50, ge=0, le=500)
    index_mode: IndexMode = IndexMode.HYBRID
    env_file: str = Field(".env", description="Path to .env config file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for imported documents")
    cookie: Optional[str] = Field(None, description="Optional cookie for wiki authentication")


class ImportProgress(BaseModel):
    job_id: str
    status: str
    documents_processed: int
    documents_total: int
    chunks_created: int
    elapsed_seconds: float
    error: Optional[str] = None


class ImportStats(BaseModel):
    job_id: str
    documents_loaded: int
    documents_ingested: int
    chunks_created: int
    vector_store_count: int
    elapsed_seconds: float


class FileInfo(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size_bytes: int
    temp_path: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to retrieve")
    mode: ChatMode = Field(ChatMode.PIPELINE, description="Query mode")
    env_file: str = Field(".env", description="Path to .env config file")
    conversation: Optional[ConversationRequest] = Field(None, description="Conversation context")


class SourceNode(BaseModel):
    node_id: str
    text: str
    score: float
    source_file: Optional[str] = None
    header_path: Optional[str] = None


class ImageInfo(BaseModel):
    path: str
    description: str
    node_id: str


class ChatResponse(BaseModel):
    query: str
    response: str
    source_nodes: List[SourceNode] = Field(default_factory=list)
    images: List[ImageInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    conversation: Optional[ConversationInfo] = None


class RetrieveMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")
    rerank: bool = Field(True, description="Enable reranking")
    use_pre_retrieval: bool = Field(False, description="Enable query transformation")
    retrieve_mode: RetrieveMode = Field(RetrieveMode.HYBRID, description="Search mode: keyword, vector, or hybrid")
    env_file: str = Field(".env", description="Config file path")


class SearchResultFile(BaseModel):
    filename: str
    chunk_count: int


class SearchResultChunk(BaseModel):
    chunk_id: str
    heading: Optional[str] = Field(None, description="Section heading")
    score: float
    text_preview: str = Field(..., description="First 200 characters")
    full_text: str
    source_file: str
    header_path: Optional[str] = Field(None, description="Full header hierarchy")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    files: List[SearchResultFile]
    chunks: List[SearchResultChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)
