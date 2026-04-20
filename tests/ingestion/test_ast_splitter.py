"""Tests for the AST-based code splitter."""

import pytest

from profirag.ingestion.ast_splitter import (
    ASTSplitter,
    BaseLanguageParser,
    CodeChunk,
    PythonParser,
    JavaParser,
    CppParser,
    GoParser,
)


class TestCodeChunk:
    """Unit tests for the CodeChunk dataclass."""

    def test_code_chunk_creation(self):
        """CodeChunk stores all required fields."""
        chunk = CodeChunk(
            code="def hello(): pass",
            language="python",
            entity_name="hello",
            entity_type="function",
            file_path="/test.py",
            start_line=1,
            end_line=2,
        )
        assert chunk.code == "def hello(): pass"
        assert chunk.language == "python"
        assert chunk.entity_name == "hello"
        assert chunk.entity_type == "function"
        assert chunk.file_path == "/test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 2

    def test_code_chunk_to_text_node(self):
        """CodeChunk converts to TextNode with correct metadata."""
        chunk = CodeChunk(
            code="def hello(): pass",
            language="python",
            entity_name="hello",
            entity_type="function",
            file_path="/test.py",
            start_line=1,
            end_line=2,
        )
        node = chunk.to_text_node()
        assert node.text == "def hello(): pass"
        assert node.metadata["language"] == "python"
        assert node.metadata["function_name"] == "hello"
        assert node.metadata["entity_type"] == "function"


class TestLanguageRegistry:
    """Unit tests for language parser registry and extension mapping."""

    def test_parsers_registered(self):
        """All four target language parsers are available."""
        from profirag.ingestion.ast_splitter import LANGUAGE_PARSERS
        assert "python" in LANGUAGE_PARSERS
        assert "java" in LANGUAGE_PARSERS
        assert "cpp" in LANGUAGE_PARSERS
        assert "go" in LANGUAGE_PARSERS


class TestASTSplitter:
    """Unit tests for the ASTSplitter class."""

    def test_splitter_init_defaults(self):
        """Default constructor sets sensible values."""
        splitter = ASTSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50
        assert splitter.language == "python"

    def test_splitter_init_custom(self):
        """Custom constructor values are stored."""
        splitter = ASTSplitter(chunk_size=256, chunk_overlap=20, language="java")
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20
        assert splitter.language == "java"

    def test_unsupported_language_raises(self):
        """ASTSplitter raises ValueError for unsupported languages."""
        with pytest.raises(ValueError, match="Unsupported language"):
            ASTSplitter(language="ruby")

    def test_split_text_returns_list(self):
        """split_text returns a list of TextNode objects."""
        splitter = ASTSplitter(language="python")
        nodes = splitter.split_text("def foo(): pass", "test.py")
        assert isinstance(nodes, list)
        assert len(nodes) >= 1

    def test_split_text_with_file_path(self):
        """file_path is passed to parser correctly."""
        splitter = ASTSplitter(language="python")
        nodes = splitter.split_text("def foo(): pass", "/path/test.py")
        assert len(nodes) >= 1


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
    assert chunk.file_path == "/test.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 2


def test_base_parser_abstract():
    parser = BaseLanguageParser()
    with pytest.raises(NotImplementedError):
        parser.parse("def foo(): pass", "/test.py")


class TestPythonParser:
    """Unit tests for the PythonParser class."""

    def test_python_parser_basic_function(self):
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
        assert chunks[0].end_line == 3

    def test_python_parser_class(self):
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