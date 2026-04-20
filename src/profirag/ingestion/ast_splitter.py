"""AST-based code splitter using tree-sitter for semantic-aware chunking.

Supports Python, Java, C/C++, and Go source files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """Represents a semantically meaningful chunk of source code.

    Attributes:
        text: The source text of the chunk.
        language: Programming language identifier (e.g. "python", "java").
        chunk_type: AST node type that produced this chunk
                    (e.g. "function_definition", "class_definition").
        start_line: 1-based starting line number in the original file.
        end_line: 1-based ending line number in the original file.
        name: Optional qualified name of the symbol (e.g. "MyClass.my_method").
        metadata: Arbitrary extra metadata.
    """

    text: str
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract parser base
# ---------------------------------------------------------------------------

class BaseLanguageParser(ABC):
    """Abstract base class for language-specific AST parsers."""

    @abstractmethod
    def parse(self, source: str) -> List[CodeChunk]:
        """Parse *source* and return a list of semantic code chunks.

        Args:
            source: Raw source code text.

        Returns:
            Ordered list of :class:`CodeChunk` objects.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Language-specific parser placeholders
# (full implementations added in subsequent tasks)
# ---------------------------------------------------------------------------

class PythonParser(BaseLanguageParser):
    """AST parser for Python source files."""

    def parse(self, source: str) -> List[CodeChunk]:
        raise NotImplementedError("PythonParser.parse not yet implemented")


class JavaParser(BaseLanguageParser):
    """AST parser for Java source files."""

    def parse(self, source: str) -> List[CodeChunk]:
        raise NotImplementedError("JavaParser.parse not yet implemented")


class CppParser(BaseLanguageParser):
    """AST parser for C and C++ source files."""

    def parse(self, source: str) -> List[CodeChunk]:
        raise NotImplementedError("CppParser.parse not yet implemented")


class GoParser(BaseLanguageParser):
    """AST parser for Go source files."""

    def parse(self, source: str) -> List[CodeChunk]:
        raise NotImplementedError("GoParser.parse not yet implemented")


# ---------------------------------------------------------------------------
# Supported language registry
# ---------------------------------------------------------------------------

LANGUAGE_PARSERS: Dict[str, type] = {
    "python": PythonParser,
    "java": JavaParser,
    "cpp": CppParser,
    "c": CppParser,
    "go": GoParser,
}

# File-extension to language mapping
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".go": "go",
}


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

class ASTSplitter:
    """Semantic code splitter backed by tree-sitter AST parsing.

    Usage::

        splitter = ASTSplitter()
        chunks = splitter.split_code(source, language="python")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        fallback_to_text: bool = True,
    ) -> None:
        """Initialise the splitter.

        Args:
            chunk_size: Soft maximum number of characters per chunk.
            chunk_overlap: Character overlap between adjacent chunks when
                           the parser falls back to plain-text splitting.
            fallback_to_text: When *True*, fall back to line-based splitting
                              for unsupported languages instead of raising.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_to_text = fallback_to_text
        self._parser_cache: Dict[str, BaseLanguageParser] = {}

    def _get_parser(self, language: str) -> BaseLanguageParser:
        """Return (and cache) a parser for *language*.

        Args:
            language: Normalised language identifier.

        Returns:
            A :class:`BaseLanguageParser` instance.

        Raises:
            ValueError: If *language* is unsupported and
                        :attr:`fallback_to_text` is *False*.
        """
        if language not in self._parser_cache:
            parser_cls = LANGUAGE_PARSERS.get(language)
            if parser_cls is None:
                raise ValueError(
                    f"Unsupported language: {language!r}. "
                    f"Supported: {sorted(LANGUAGE_PARSERS)}"
                )
            self._parser_cache[language] = parser_cls()
        return self._parser_cache[language]

    def split_code(
        self,
        source: str,
        language: str,
        file_path: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Split *source* into semantic chunks.

        Args:
            source: Raw source code.
            language: Language identifier (e.g. ``"python"``).
            file_path: Optional origin path stored in chunk metadata.

        Returns:
            List of :class:`CodeChunk` objects.
        """
        language = language.lower()
        parser = self._get_parser(language)
        chunks = parser.parse(source)
        if file_path:
            for chunk in chunks:
                chunk.metadata.setdefault("file_path", file_path)
        return chunks

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect the programming language from a file extension.

        Args:
            file_path: Path to the source file (only the extension is used).

        Returns:
            A language identifier string, or *None* if unrecognised.
        """
        import pathlib
        suffix = pathlib.Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(suffix)
