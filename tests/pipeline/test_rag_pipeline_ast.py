"""Tests for AST splitter integration into RAG pipeline"""

from profirag.config.settings import ChunkingConfig
from profirag.ingestion.ast_splitter import ASTSplitter


def test_create_ast_splitter():
    # Test that ASTSplitter is created correctly
    config = ChunkingConfig(splitter_type="ast", ast_language="python")
    splitter = ASTSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        language=config.ast_language
    )
    assert splitter.language == "python"
