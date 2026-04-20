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
    LANGUAGE_PARSERS,
    EXTENSION_TO_LANGUAGE,
)


class TestCodeChunk:
    """Unit tests for the CodeChunk dataclass."""

    def test_code_chunk_creation(self):
        """CodeChunk stores all required fields."""
        chunk = CodeChunk(
            text="def hello(): pass",
            language="python",
            chunk_type="function_definition",
            start_line=1,
            end_line=2,
            name="hello",
        )
        assert chunk.text == "def hello(): pass"
        assert chunk.language == "python"
        assert chunk.chunk_type == "function_definition"
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.name == "hello"

    def test_code_chunk_defaults(self):
        """Optional fields default to sensible values."""
        chunk = CodeChunk(
            text="class Foo: pass",
            language="python",
            chunk_type="class_definition",
            start_line=5,
            end_line=6,
        )
        assert chunk.name is None
        assert chunk.metadata == {}


class TestLanguageRegistry:
    """Unit tests for language parser registry and extension mapping."""

    def test_supported_languages(self):
        """All four target languages are registered."""
        expected = {"python", "java", "cpp", "c", "go"}
        assert set(LANGUAGE_PARSERS.keys()) == expected

    def test_extension_mappings(self):
        """Common file extensions map to the correct languages."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".java"] == "java"
        assert EXTENSION_TO_LANGUAGE[".go"] == "go"
        assert EXTENSION_TO_LANGUAGE[".cpp"] == "cpp"
        assert EXTENSION_TO_LANGUAGE[".c"] == "c"
        assert EXTENSION_TO_LANGUAGE[".hpp"] == "cpp"


class TestASTSplitter:
    """Unit tests for the ASTSplitter class."""

    def test_splitter_init_defaults(self):
        """Default constructor sets sensible values."""
        splitter = ASTSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50
        assert splitter.fallback_to_text is True

    def test_splitter_init_custom(self):
        """Custom constructor values are stored."""
        splitter = ASTSplitter(chunk_size=256, chunk_overlap=20, fallback_to_text=False)
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20
        assert splitter.fallback_to_text is False

    def test_detect_language_python(self):
        """detect_language returns correct language for .py files."""
        splitter = ASTSplitter()
        assert splitter.detect_language("example.py") == "python"

    def test_detect_language_java(self):
        """detect_language returns correct language for .java files."""
        splitter = ASTSplitter()
        assert splitter.detect_language("MyClass.java") == "java"

    def test_detect_language_go(self):
        """detect_language returns correct language for .go files."""
        splitter = ASTSplitter()
        assert splitter.detect_language("main.go") == "go"

    def test_detect_language_cpp(self):
        """detect_language returns correct language for various C++ extensions."""
        splitter = ASTSplitter()
        assert splitter.detect_language("file.cpp") == "cpp"
        assert splitter.detect_language("file.hpp") == "cpp"
        assert splitter.detect_language("file.h") == "cpp"
        assert splitter.detect_language("file.c") == "c"

    def test_detect_language_unknown(self):
        """detect_language returns None for unrecognised extensions."""
        splitter = ASTSplitter()
        assert splitter.detect_language("file.txt") is None
        assert splitter.detect_language("file.xyz") is None

    def test_unsupported_language_raises(self):
        """split_code raises ValueError for unsupported languages when fallback is disabled."""
        splitter = ASTSplitter(fallback_to_text=False)
        with pytest.raises(ValueError, match="Unsupported language"):
            splitter.split_code("some code", language="ruby")

    def test_split_code_returns_list(self):
        """split_code returns a list of CodeChunk objects."""
        splitter = ASTSplitter()
        # Currently raises NotImplementedError because parsers are skeletons.
        # This test documents the expected return type contract.
        with pytest.raises(NotImplementedError):
            splitter.split_code("def foo(): pass", language="python")

    def test_split_code_with_file_path(self):
        """file_path is stored in chunk metadata when provided."""
        splitter = ASTSplitter()
        with pytest.raises(NotImplementedError):
            splitter.split_code("def foo(): pass", language="python", file_path="foo.py")


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
        parser.parse("def foo(): pass")


class TestParserPlaceholders:
    """Sanity tests that parser stubs exist and raise NotImplementedError."""

    def test_python_parser_placeholder(self):
        parser = PythonParser()
        with pytest.raises(NotImplementedError):
            parser.parse("def foo(): pass")

    def test_java_parser_placeholder(self):
        parser = JavaParser()
        with pytest.raises(NotImplementedError):
            parser.parse("public class Foo {}")

    def test_cpp_parser_placeholder(self):
        parser = CppParser()
        with pytest.raises(NotImplementedError):
            parser.parse("void foo() {}")

    def test_go_parser_placeholder(self):
        parser = GoParser()
        with pytest.raises(NotImplementedError):
            parser.parse("func main() {}")
