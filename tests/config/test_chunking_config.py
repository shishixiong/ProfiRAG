from profirag.config.settings import ChunkingConfig

def test_chunking_config_ast_type():
    config = ChunkingConfig(splitter_type="ast", language="en")
    assert config.splitter_type == "ast"

def test_chunking_config_ast_language():
    config = ChunkingConfig(splitter_type="ast", ast_language="java")
    assert config.ast_language == "java"