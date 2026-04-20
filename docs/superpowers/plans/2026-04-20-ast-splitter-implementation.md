# AST Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AST-based code splitter supporting Python, Java, C/C++, Go with semantic-aware chunking that preserves function/class boundaries.

**Architecture:** Independent `ast_splitter.py` module using tree-sitter for parsing, integrated via `ChunkingConfig.splitter_type="ast"`. Recursive splitting handles oversized chunks.

**Tech Stack:** tree-sitter, tree-sitter-python, tree-sitter-java, tree-sitter-cpp, tree-sitter-go

---

## File Structure

```
src/profirag/ingestion/
├── ast_splitter.py          # Create: AST splitter + language parsers
└── splitters.py             # Modify: existing splitters (unchanged)

src/profirag/config/
└── settings.py              # Modify: ChunkingConfig - add "ast" type

src/profirag/pipeline/
└── rag_pipeline.py          # Modify: _create_splitter() - add ASTSplitter

tests/ingestion/
└── test_ast_splitter.py     # Create: unit tests
```

---

## Tasks

### Task 1: Project Dependencies & Basic Structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/profirag/ingestion/ast_splitter.py` (skeleton)
- Create: `tests/ingestion/test_ast_splitter.py` (skeleton)

- [ ] **Step 1: Add tree-sitter dependencies to pyproject.toml**

Run: `grep -n "tree-sitter" pyproject.toml`
Expected: empty (not yet added)

Edit `pyproject.toml`, add after `[project.dependencies]`:

```toml
tree-sitter>=0.20.0
tree-sitter-python>=0.20.0
tree-sitter-java>=0.20.0
tree-sitter-cpp>=0.20.0
tree-sitter-go>=0.20.0
```

- [ ] **Step 2: Run test to confirm dependency addition**

Run: `grep -n "tree-sitter" pyproject.toml`
Expected: 5 lines with tree-sitter dependencies

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add tree-sitter dependencies for AST splitter"
```

---

### Task 2: CodeChunk Dataclass & Basic Interface

**Files:**
- Modify: `src/profirag/ingestion/ast_splitter.py`
- Modify: `tests/ingestion/test_ast_splitter.py`

- [ ] **Step 1: Write failing test for CodeChunk**

Run: `pytest tests/ingestion/test_ast_splitter.py::test_code_chunk_dataclass -v`
Expected: FAIL (file not found or test not found)

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import CodeChunk

def test_code_chunk_dataclass():
    chunk = CodeChunk(
        code="def foo(): pass",
        language="python",
        entity_name="foo",
        entity_type="function",
        file_path="/test.py",
        start_line=1,
        end_line=2
    )
    assert chunk.code == "def foo(): pass"
    assert chunk.language == "python"
    assert chunk.entity_name == "foo"
    assert chunk.entity_type == "function"
```

- [ ] **Step 2: Create ast_splitter.py with CodeChunk**

Create `src/profirag/ingestion/ast_splitter.py`:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    code: str
    language: str
    entity_name: str
    entity_type: str  # "function" | "class" | "method" | "module"
    file_path: str
    start_line: int
    end_line: int

    def to_text_node(self):
        """Convert to llama_index TextNode."""
        from llama_index.core.schema import TextNode
        return TextNode(
            text=self.code,
            metadata={
                "language": self.language,
                "function_name": self.entity_name if self.entity_type == "function" else None,
                "class_name": self.entity_name if self.entity_type == "class" else None,
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "entity_type": self.entity_type,
            }
        )
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/ingestion/test_ast_splitter.py::test_code_chunk_dataclass -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/ingestion/ast_splitter.py tests/ingestion/test_ast_splitter.py
git commit -m "feat(ast_splitter): add CodeChunk dataclass"
```

---

### Task 3: BaseLanguageParser Abstract Class

**Files:**
- Modify: `src/profirag/ingestion/ast_splitter.py`
- Modify: `tests/ingestion/test_ast_splitter.py`

- [ ] **Step 1: Write failing test for BaseLanguageParser**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import BaseLanguageParser

def test_base_parser_abstract():
    parser = BaseLanguageParser()
    with pytest.raises(NotImplementedError):
        parser.parse("def foo(): pass")
```

- [ ] **Step 2: Add BaseLanguageParser to ast_splitter.py**

Add after CodeChunk class:

```python
from abc import ABC, abstractmethod

class BaseLanguageParser(ABC):
    """Abstract base class for language-specific AST parsers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse source code and return list of code chunks."""
        pass

    @abstractmethod
    def get_language_name(self) -> str:
        """Return the language name (python, java, cpp, go)."""
        pass

    def _estimate_tokens(self, code: str) -> int:
        """Estimate token count from code string."""
        return len(code) // 4  # rough estimate

    def _split_if_needed(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split oversized chunk into smaller pieces."""
        if self._estimate_tokens(chunk.code) <= self.chunk_size:
            return [chunk]

        # Try to split by logical blocks (subclasses override for language-specific)
        return self._split_by_blocks(chunk)

    @abstractmethod
    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Language-specific block splitting."""
        pass
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/ingestion/test_ast_splitter.py::test_base_parser_abstract -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/ingestion/ast_splitter.py tests/ingestion/test_ast_splitter.py
git commit -m "feat(ast_splitter): add BaseLanguageParser abstract class"
```

---

### Task 4: PythonParser Implementation

**Files:**
- Modify: `src/profirag/ingestion/ast_splitter.py`
- Modify: `tests/ingestion/test_ast_splitter.py`

- [ ] **Step 1: Write failing test for PythonParser - basic parse**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import PythonParser

def test_python_parser_basic_function():
    parser = PythonParser(chunk_size=512)
    code = """
def hello():
    print("hello")

def world():
    print("world")
"""
    chunks = parser.parse(code, "/test.py")
    assert len(chunks) == 2
    assert chunks[0].entity_name == "hello"
    assert chunks[0].entity_type == "function"
    assert chunks[0].start_line == 2
    assert chunks[0].end_line == 4

def test_python_parser_class():
    parser = PythonParser(chunk_size=512)
    code = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass
"""
    chunks = parser.parse(code, "/test.py")
    assert len(chunks) == 1
    assert chunks[0].entity_name == "MyClass"
    assert chunks[0].entity_type == "class"
```

- [ ] **Step 2: Implement PythonParser in ast_splitter.py**

Add after BaseLanguageParser class:

```python
class PythonParser(BaseLanguageParser):
    """Parser for Python code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._parser = None
        self._ensure_parser()

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_python as tspython
                from tree_sitter import Language, Parser
                self._language = Language(tspython.language())
                self._parser = Parser(self._language)
            except ImportError:
                raise ImportError(
                    "tree-sitter-python not installed. Run: pip install tree-sitter-python"
                )

    def get_language_name(self) -> str:
        return "python"

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse Python source and return function/class chunks."""
        tree = self._parser.parse(bytes(source_code, "utf8"))
        chunks = []
        self._extract_entities(tree.root_node, source_code, file_path, chunks)
        return chunks

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk]):
        """Recursively extract functions and classes."""
        for child in node.children:
            if child.type in ("function_definition", "async_generator_function_definition"):
                self._create_chunk_from_node(child, source_code, file_path, chunks, "function")
            elif child.type == "class_definition":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "class")
            elif child.type == "module":
                # Top-level module code - treat as single chunk
                pass
            else:
                self._extract_entities(child, source_code, file_path, chunks)

    def _create_chunk_from_node(self, node, source_code: str, file_path: str,
                                 chunks: List[CodeChunk], entity_type: str):
        """Create CodeChunk from tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        code = source_code[start_byte:end_byte]

        # Get entity name
        name_node = None
        for child in node.children:
            if child.type in ("identifier", "name"):
                name_node = child
                break
        entity_name = name_node.text.decode("utf8") if name_node else "<anonymous>"

        chunk = CodeChunk(
            code=code,
            language="python",
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )

        # Check if needs splitting
        split_chunks = self._split_if_needed(chunk)
        chunks.extend(split_chunks)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split Python chunk by internal blocks."""
        source_code = chunk.code
        tree = self._parser.parse(bytes(source_code, "utf8"))

        # Find if/while/for with large bodies
        sub_chunks = []
        self._split_by_control_flow(tree.root_node, source_code, chunk, sub_chunks)

        if len(sub_chunks) <= 1:
            # No internal splits possible, do hard split
            return self._hard_split(chunk)

        return sub_chunks

    def _split_by_control_flow(self, node, source_code: str, parent_chunk: CodeChunk,
                                sub_chunks: List[CodeChunk], offset: int = 0):
        """Split by if/while/for blocks."""
        for child in node.children:
            if child.type in ("if_statement", "while_statement", "for_statement"):
                body_start = None
                body_end = None
                for c in child.children:
                    if c.type == "block":
                        body_start = c.start_byte
                        body_end = c.end_byte
                        break

                if body_start is not None:
                    body_code = source_code[body_start - offset:body_end - offset]
                    if self._estimate_tokens(body_code) > self.chunk_size * 0.5:
                        # Large body, split recursively
                        sub_chunk = CodeChunk(
                            code=body_code.strip(),
                            language="python",
                            entity_name=f"{parent_chunk.entity_name}_block",
                            entity_type="block",
                            file_path=parent_chunk.file_path,
                            start_line=parent_chunk.start_line,
                            end_line=parent_chunk.end_line
                        )
                        sub_chunks.extend(self._split_if_needed(sub_chunk))
                        continue

            self._split_by_control_flow(child, source_code, parent_chunk, sub_chunks, offset)

    def _hard_split(self, chunk: CodeChunk, max_tokens: int = None) -> List[CodeChunk]:
        """Hard split by lines when no semantic split possible."""
        if max_tokens is None:
            max_tokens = self.chunk_size

        lines = chunk.code.split("\n")
        chunks = []
        current_lines = []
        current_tokens = 0

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens and current_lines:
                # Create chunk
                code = "\n".join(current_lines)
                chunks.append(CodeChunk(
                    code=code,
                    language=chunk.language,
                    entity_name=chunk.entity_name,
                    entity_type=chunk.entity_type,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.start_line + len(current_lines) - 1
                ))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            code = "\n".join(current_lines)
            chunks.append(CodeChunk(
                code=code,
                language=chunk.language,
                entity_name=chunk.entity_name,
                entity_type=chunk.entity_type,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.start_line + len(current_lines) - 1
            ))

        return chunks
```

- [ ] **Step 3: Run tests to verify Python parser**

Run: `pytest tests/ingestion/test_ast_splitter.py::test_python_parser_basic_function tests/ingestion/test_ast_splitter.py::test_python_parser_class -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/ingestion/ast_splitter.py tests/ingestion/test_ast_splitter.py
git commit -m "feat(ast_splitter): implement PythonParser with tree-sitter"
```

---

### Task 5: JavaParser, CppParser, GoParser Implementations

**Files:**
- Modify: `src/profirag/ingestion/ast_splitter.py`
- Modify: `tests/ingestion/test_ast_splitter.py`

- [ ] **Step 1: Write failing tests for JavaParser**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import JavaParser

def test_java_parser_basic_method():
    parser = JavaParser(chunk_size=512)
    code = """
public class MyClass {
    public void hello() {
        System.out.println("hello");
    }

    public void world() {
        System.out.println("world");
    }
}
"""
    chunks = parser.parse(code, "/Test.java")
    assert len(chunks) == 1  # class level
    assert chunks[0].entity_name == "MyClass"
    assert chunks[0].entity_type == "class"
```

- [ ] **Step 2: Write failing tests for CppParser**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import CppParser

def test_cpp_parser_basic_function():
    parser = CppParser(chunk_size=512)
    code = """
void hello() {
    printf("hello");
}

void world() {
    printf("world");
}
"""
    chunks = parser.parse(code, "/test.cpp")
    assert len(chunks) == 2
    assert chunks[0].entity_type == "function"
```

- [ ] **Step 3: Write failing tests for GoParser**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import GoParser

def test_go_parser_basic_function():
    parser = GoParser(chunk_size=512)
    code = """
func hello() {
    println("hello")
}

func world() {
    println("world")
}
"""
    chunks = parser.parse(code, "/test.go")
    assert len(chunks) == 2
    assert chunks[0].entity_name == "hello"
    assert chunks[0].entity_type == "function"
```

- [ ] **Step 4: Implement JavaParser, CppParser, GoParser in ast_splitter.py**

Add similar implementations for Java, C++, Go after PythonParser. Each follows same pattern:
- `_ensure_parser()` loads tree-sitter language
- `parse()` extracts entities
- `_extract_entities()` walks AST for function/method/class definitions
- `_split_by_blocks()` handles oversized chunks
- `_hard_split()` line-based fallback

```python
class JavaParser(BaseLanguageParser):
    """Parser for Java code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._parser = None
        self._ensure_parser()

    def _ensure_parser(self):
        if self._parser is None:
            try:
                import tree_sitter_java as tsjava
                from tree_sitter import Language, Parser
                self._language = Language(tsjava.language())
                self._parser = Parser(self._language)
            except ImportError:
                raise ImportError("tree-sitter-java not installed")

    def get_language_name(self) -> str:
        return "java"

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        tree = self._parser.parse(bytes(source_code, "utf8"))
        chunks = []
        self._extract_entities(tree.root_node, source_code, file_path, chunks)
        return chunks

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk]):
        for child in node.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                self._create_chunk_from_node(child, source_code, file_path, chunks, "method")
            elif child.type == "class_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "class")
            elif child.type == "interface_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "interface")
            else:
                self._extract_entities(child, source_code, file_path, chunks)

    # _create_chunk_from_node, _split_if_needed, _split_by_blocks, _hard_split
    # follow same pattern as PythonParser - copy and adapt for Java node types


class CppParser(BaseLanguageParser):
    """Parser for C/C++ code using tree-sitter."""

    def get_language_name(self) -> str:
        return "cpp"

    # Similar implementation with C++ specific node types:
    # function_definition, class_specifier, struct_specifier, namespace_specifier


class GoParser(BaseLanguageParser):
    """Parser for Go code using tree-sitter."""

    def get_language_name(self) -> str:
        return "go"

    # Similar implementation with Go specific node types:
    # function_declaration, method_declaration, type_declaration
```

- [ ] **Step 5: Run all parser tests**

Run: `pytest tests/ingestion/test_ast_splitter.py -k "parser" -v`
Expected: All parser tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/profirag/ingestion/ast_splitter.py tests/ingestion/test_ast_splitter.py
git commit -m "feat(ast_splitter): add JavaParser, CppParser, GoParser implementations"
```

---

### Task 6: ASTSplitter Main Class

**Files:**
- Modify: `src/profirag/ingestion/ast_splitter.py`
- Modify: `tests/ingestion/test_ast_splitter.py`

- [ ] **Step 1: Write failing test for ASTSplitter**

Add to `tests/ingestion/test_ast_splitter.py`:

```python
from profirag.ingestion.ast_splitter import ASTSplitter

def test_ast_splitter_basic():
    splitter = ASTSplitter(
        chunk_size=512,
        chunk_overlap=50,
        language="python"
    )
    code = """
def hello():
    print("hello")

def world():
    print("world")
"""
    nodes = splitter.split_text(code, "test.py")
    assert len(nodes) == 2
    assert nodes[0].text == "def hello():\n    print(\"hello\")"
```

- [ ] **Step 2: Implement ASTSplitter class**

Add at the end of `ast_splitter.py`:

```python
class ASTSplitter:
    """Main AST-based splitter for code files."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 language: str = "python"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self._parser = self._create_parser(language)

    def _create_parser(self, language: str) -> BaseLanguageParser:
        """Create language-specific parser."""
        parsers = {
            "python": PythonParser,
            "java": JavaParser,
            "cpp": CppParser,
            "go": GoParser,
        }
        parser_class = parsers.get(language)
        if parser_class is None:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(parsers.keys())}")
        return parser_class(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def split_text(self, text: str, file_path: str = "") -> List:
        """Split code text into TextNode chunks."""
        chunks = self._parser.parse(text, file_path)
        return [chunk.to_text_node() for chunk in chunks]

    def split_document(self, document) -> List:
        """Split a document into code chunks."""
        code = document.text
        file_path = document.metadata.get("file_path", "")
        return self.split_text(code, file_path)

    def split_documents(self, documents: List) -> List:
        """Split multiple documents."""
        nodes = []
        for doc in documents:
            nodes.extend(self.split_document(doc))
        return nodes
```

- [ ] **Step 3: Run test to verify ASTSplitter**

Run: `pytest tests/ingestion/test_ast_splitter.py::test_ast_splitter_basic -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/ingestion/ast_splitter.py tests/ingestion/test_ast_splitter.py
git commit -m "feat(ast_splitter): add ASTSplitter main class"
```

---

### Task 7: Configuration Integration

**Files:**
- Modify: `src/profirag/config/settings.py`
- Create: `tests/config/test_chunking_config.py`

- [ ] **Step 1: Write failing test for ast splitter config**

Run: `ls tests/config/ 2>/dev/null || echo "dir not found"`

If dir not found:
```bash
mkdir -p tests/config
touch tests/config/__init__.py
```

Add to `tests/config/test_chunking_config.py`:

```python
from profirag.config.settings import ChunkingConfig

def test_chunking_config_ast_type():
    config = ChunkingConfig(splitter_type="ast", language="python")
    assert config.splitter_type == "ast"

def test_chunking_config_ast_language():
    config = ChunkingConfig(splitter_type="ast", ast_language="java")
    assert config.ast_language == "java"
```

- [ ] **Step 2: Modify ChunkingConfig in settings.py**

Run: `grep -n "class ChunkingConfig" src/profirag/config/settings.py`

Read the ChunkingConfig class and add:

```python
class ChunkingConfig(BaseModel):
    splitter_type: Literal["sentence", "token", "semantic", "chinese", "ast"] = "sentence"
    chunk_size: int = 512
    chunk_overlap: int = 50
    language: Literal["en", "zh"] = "en"

    # AST-specific settings
    ast_language: Literal["python", "java", "cpp", "go"] = "python"
```

- [ ] **Step 3: Run config tests**

Run: `pytest tests/config/test_chunking_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/config/settings.py tests/config/test_chunking_config.py
git commit -m "feat(config): add AST splitter type and ast_language setting"
```

---

### Task 8: Pipeline Integration

**Files:**
- Modify: `src/profirag/pipeline/rag_pipeline.py`
- Create: `tests/pipeline/test_rag_pipeline_ast.py`

- [ ] **Step 1: Write failing test for pipeline AST integration**

Add to `tests/pipeline/test_rag_pipeline_ast.py`:

```python
def test_create_ast_splitter():
    from profirag.config.settings import ChunkingConfig
    from profirag.ingestion.ast_splitter import ASTSplitter

    config = ChunkingConfig(splitter_type="ast", ast_language="python")
    # Test that ASTSplitter is created correctly
    splitter = ASTSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        language=config.ast_language
    )
    assert splitter.language == "python"
```

- [ ] **Step 2: Modify _create_splitter in rag_pipeline.py**

Run: `grep -n "_create_splitter" src/profirag/pipeline/rag_pipeline.py`

Read the method and update it:

```python
def _create_splitter(self):
    chunking = self.config.chunking
    if chunking.splitter_type == "ast":
        from profirag.ingestion.ast_splitter import ASTSplitter
        return ASTSplitter(
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
            language=chunking.ast_language,
        )
    elif chunking.splitter_type == "chinese" or chunking.language == "zh":
        return ChineseTextSplitter(
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
        )
    else:
        return TextSplitter(
            splitter_type=chunking.splitter_type,
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
            embed_model=self._embed_model if chunking.splitter_type == "semantic" else None,
        )
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/pipeline/test_rag_pipeline_ast.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/profirag/pipeline/rag_pipeline.py tests/pipeline/test_rag_pipeline_ast.py
git commit -m "feat(pipeline): integrate AST splitter into RAG pipeline"
```

---

### Task 9: Integration Test with Real Files

**Files:**
- Create: `tests/ingestion/test_ast_splitter_integration.py`
- Create: `tests/fixtures/sample.py`, `tests/fixtures/sample.java`, etc.

- [ ] **Step 1: Create sample code fixtures**

```bash
mkdir -p tests/fixtures
```

Create `tests/fixtures/sample.py`:
```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
```

- [ ] **Step 2: Write integration test**

Add to `tests/ingestion/test_ast_splitter_integration.py`:

```python
from pathlib import Path
from profirag.ingestion.ast_splitter import ASTSplitter

def test_ast_splitter_with_python_file():
    fixture_path = Path("tests/fixtures/sample.py")
    code = fixture_path.read_text()

    splitter = ASTSplitter(chunk_size=512, language="python")
    nodes = splitter.split_text(code, str(fixture_path))

    assert len(nodes) >= 3  # at least 2 functions + 1 class
    entity_types = [n.metadata.get("entity_type") for n in nodes]
    assert "function" in entity_types
    assert "class" in entity_types
```

- [ ] **Step 3: Run integration test**

Run: `pytest tests/ingestion/test_ast_splitter_integration.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/ingestion/test_ast_splitter_integration.py tests/fixtures/
git commit -m "test(ast_splitter): add integration tests with real code files"
```

---

## Verification

1. **Unit tests**: `pytest tests/ingestion/test_ast_splitter.py -v`
2. **Config tests**: `pytest tests/config/test_chunking_config.py -v`
3. **Pipeline tests**: `pytest tests/pipeline/test_rag_pipeline_ast.py -v`
4. **Integration tests**: `pytest tests/ingestion/test_ast_splitter_integration.py -v`
5. **All tests**: `pytest tests/ -v`

---

## Dependencies

All tree-sitter packages need to be installed:
```bash
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-cpp tree-sitter-go
```