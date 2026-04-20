# AST Splitter Design

## Context

ProfiRAG needs to support code file chunking for Python, Java, C/C++, and Go. Currently no code-specific parsing exists - code files are treated as plain text. The goal is semantic-aware splitting that preserves function/class boundaries while respecting chunk size constraints.

## Architecture

```
src/profirag/ingestion/
├── ast_splitter.py          # Main splitter + utilities
└── splitters.py             # Existing splitters (unchanged)
```

## Core Components

### 1. ASTSplitter Class

Entry point, implements same interface as existing splitters:
- `split_text(text, language)` → List[TextNode]
- `split_document(document)` → List[TextNode]
- `split_documents(documents)` → List[TextNode]

### 2. Language Parsers (tree-sitter based)

| Parser | tree-sitter Grammar | Key Features |
|--------|---------------------|--------------|
| PythonParser | tree-sitter-python | function, class, async handling |
| JavaParser | tree-sitter-java | generic support |
| CppParser | tree-sitter-cpp | template, namespace support |
| GoParser | tree-sitter-go | goroutine, defer support |

Each parser:
- `parse(source_code)` → List[CodeChunk]
- `extract_entities(tree)` → recursive entity extraction
- `_split_if_needed(chunk)` → handle oversized chunks

### 3. CodeChunk Data Class

```python
@dataclass
class CodeChunk:
    code: str              # source code
    language: str          # python/java/cpp/go
    entity_name: str        # function/class name
    entity_type: str        # "function" | "class" | "method" | "module"
    file_path: str
    start_line: int
    end_line: int
```

## Splitting Algorithm

### Phase 1: Semantic Split

1. Parse source → AST
2. Extract top-level entities (functions, classes)
3. For each entity, create CodeChunk
4. Check size against `chunk_size`

### Phase 2: Recursive Split (if oversized)

For chunks exceeding `chunk_size`:
1. Try splitting by internal blocks (if/while/for with large bodies)
2. Recursively process sub-chunks
3. If no internal splits possible, hard-split by line count

### Phase 3: Merge (if undersized)

Small consecutive chunks (e.g., small helper functions) are merged up to `chunk_size` limit.

## Configuration

```python
class ChunkingConfig(BaseModel):
    splitter_type: Literal["sentence", "token", "semantic", "chinese", "ast"] = "sentence"
    chunk_size: int = 512          # target chunk size in tokens
    chunk_overlap: int = 50
    language: Literal["en", "zh"] = "en"

    # AST-specific
    ast_language: Literal["python", "java", "cpp", "go"] = "python"
    max_function_lines: int = 200   # hard limit for function size
```

## Dependencies

```toml
tree-sitter>=0.20.0
tree-sitter-python>=0.20.0
tree-sitter-java>=0.20.0
tree-sitter-cpp>=0.20.0
tree-sitter-go>=0.20.0
```

## Metadata for Code Chunks

Same enrichment as existing splitters, plus:

```python
{
    "language": "python",
    "function_name": "process_data",
    "class_name": None,
    "file_path": "/src/processor.py",
    "start_line": 10,
    "end_line": 45,
    "entity_type": "function"
}
```

## Verification

1. Unit tests for each language parser
2. Integration test with pipeline
3. Verify chunk size constraints
4. Test oversized function handling