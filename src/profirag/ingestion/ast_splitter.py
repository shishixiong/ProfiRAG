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
    """Represents a chunk of code with metadata."""
    code: str
    language: str
    entity_name: str
    entity_type: str  # "function" | "class" | "method" | "module" | "class_header"
    file_path: str
    start_line: int
    end_line: int
    parent_class: str = ""  # For methods, the class they belong to

    def to_text_node(self):
        """Convert to llama_index TextNode."""
        from llama_index.core.schema import TextNode
        return TextNode(
            text=self.code,
            metadata={
                "language": self.language,
                "function_name": self.entity_name if self.entity_type in ("function", "method") else None,
                "class_name": self.parent_class if self.entity_type == "method" else self.entity_name if self.entity_type in ("class", "class_header") else None,
                "method_name": self.entity_name if self.entity_type == "method" else None,
                "source_file": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "entity_type": self.entity_type,
            }
        )


# ---------------------------------------------------------------------------
# Abstract parser base
# ---------------------------------------------------------------------------

class BaseLanguageParser(ABC):
    """Abstract base class for language-specific AST parsers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 extract_class_methods: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_class_methods = extract_class_methods

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse source code and return list of code chunks."""
        raise NotImplementedError

    def get_language_name(self) -> str:
        """Return the language name (python, java, cpp, go)."""
        raise NotImplementedError

    def _estimate_tokens(self, code: str) -> int:
        """Estimate token count from code string."""
        return len(code) // 4  # rough estimate

    def _split_if_needed(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split oversized chunk into smaller pieces.

        Functions, methods, constructors, and classes are kept intact regardless of size.
        Only large modules or other structures may be split if too large.
        """
        # Keep code entities intact - don't split them
        if chunk.entity_type in ("function", "method", "constructor", "class"):
            return [chunk]

        if self._estimate_tokens(chunk.code) <= self.chunk_size:
            return [chunk]

        # Try to split by logical blocks (subclasses override for language-specific)
        return self._split_by_blocks(chunk)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Language-specific block splitting."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Language-specific parser placeholders
# (full implementations added in subsequent tasks)
# ---------------------------------------------------------------------------

class PythonParser(BaseLanguageParser):
    """Parser for Python code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 extract_class_methods: bool = True):
        super().__init__(chunk_size, chunk_overlap, extract_class_methods)
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

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk],
                          in_class: bool = False):
        """Recursively extract functions and classes.

        Args:
            in_class: Whether we're currently inside a class definition.
                      When True, methods are extracted as separate chunks.
        """
        for child in node.children:
            if child.type in ("function_definition", "async_generator_function_definition"):
                # If inside a class, this is a method
                entity_type = "method" if in_class else "function"
                self._create_chunk_from_node(child, source_code, file_path, chunks, entity_type)
            elif child.type == "class_definition":
                # If extract_class_methods is True, extract methods separately
                if self.extract_class_methods:
                    # Extract class header (without method bodies) as a chunk
                    self._extract_class_with_methods(child, source_code, file_path, chunks)
                else:
                    # Extract entire class as single chunk
                    self._create_chunk_from_node(child, source_code, file_path, chunks, "class")
            elif child.type == "module":
                self._extract_entities(child, source_code, file_path, chunks)
            else:
                # Recurse into other structures to find nested functions
                self._extract_entities(child, source_code, file_path, chunks, in_class)

    def _extract_class_with_methods(self, class_node, source_code: str, file_path: str,
                                     chunks: List[CodeChunk]):
        """Extract class and its methods as separate chunks.

        This approach:
        1. Creates a chunk for the class definition (class header + docstring)
        2. Creates separate chunks for each method
        """
        # Get class name
        class_name = "<anonymous>"
        for child in class_node.children:
            if child.type in ("identifier", "name"):
                class_name = child.text.decode("utf8")
                break

        # Extract each method in the class
        for child in class_node.children:
            if child.type in ("function_definition", "async_generator_function_definition"):
                self._create_chunk_from_node(child, source_code, file_path, chunks, "method",
                                              parent_class=class_name)

        # Optionally, create a class overview chunk (class header without method bodies)
        # This is useful for understanding the class structure
        class_header_parts = []
        for child in class_node.children:
            # Include class header elements but not method bodies
            if child.type in ("identifier", "name", "argument_list", "parenthesized_list",
                              "expression_list", "simple_statement"):
                class_header_parts.append(source_code[child.start_byte:child.end_byte])
            elif child.type == "block":
                # Look for docstring or class-level statements (not methods)
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        # Could be a docstring (string literal)
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                class_header_parts.append(
                                    source_code[block_child.start_byte:block_child.end_byte])
                    elif block_child.type in ("assignment", "expression_statement"):
                        # Class-level variable/constant
                        if not any(c.type in ("function_definition", "async_generator_function_definition")
                                   for c in block_child.children):
                            class_header_parts.append(
                                source_code[block_child.start_byte:block_child.end_byte])

        if class_header_parts:
            header_code = "\n".join(class_header_parts)
            if header_code.strip():
                chunks.append(CodeChunk(
                    code=f"class {class_name}:\n{header_code}" if not header_code.startswith("class") else header_code,
                    language="python",
                    entity_name=class_name,
                    entity_type="class_header",
                    file_path=file_path,
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.start_point[0] + len(class_header_parts) + 1
                ))

    def _create_chunk_from_node(self, node, source_code: str, file_path: str,
                                 chunks: List[CodeChunk], entity_type: str):
        """Create CodeChunk from tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        code = source_code[start_byte:end_byte]

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

        split_chunks = self._split_if_needed(chunk)
        chunks.extend(split_chunks)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split Python chunk by internal blocks."""
        source_code = chunk.code
        tree = self._parser.parse(bytes(source_code, "utf8"))

        sub_chunks = []
        self._split_by_control_flow(tree.root_node, source_code, chunk, sub_chunks)

        if len(sub_chunks) <= 1:
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


class JavaParser(BaseLanguageParser):
    """Parser for Java code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._parser = None
        self._ensure_parser()

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_java as tsjava
                from tree_sitter import Language, Parser
                self._language = Language(tsjava.language())
                self._parser = Parser(self._language)
            except ImportError:
                raise ImportError(
                    "tree-sitter-java not installed. Run: pip install tree-sitter-java"
                )

    def get_language_name(self) -> str:
        return "java"

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse Java source and return class/method/constructor chunks."""
        tree = self._parser.parse(bytes(source_code, "utf8"))
        chunks = []
        self._extract_entities(tree.root_node, source_code, file_path, chunks)
        return chunks

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk]):
        """Recursively extract classes, methods, constructors, interfaces."""
        for child in node.children:
            if child.type == "class_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "class")
            elif child.type == "method_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "method")
            elif child.type == "constructor_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "constructor")
            elif child.type == "interface_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "interface")
            else:
                self._extract_entities(child, source_code, file_path, chunks)

    def _create_chunk_from_node(self, node, source_code: str, file_path: str,
                                chunks: List[CodeChunk], entity_type: str):
        """Create CodeChunk from tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        code = source_code[start_byte:end_byte]

        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break
        entity_name = name_node.text.decode("utf8") if name_node else "<anonymous>"

        chunk = CodeChunk(
            code=code,
            language="java",
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )

        split_chunks = self._split_if_needed(chunk)
        chunks.extend(split_chunks)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split Java chunk by internal blocks."""
        source_code = chunk.code
        tree = self._parser.parse(bytes(source_code, "utf8"))

        sub_chunks = []
        self._split_by_control_flow(tree.root_node, source_code, chunk, sub_chunks)

        if len(sub_chunks) <= 1:
            return self._hard_split(chunk)

        return sub_chunks

    def _split_by_control_flow(self, node, source_code: str, parent_chunk: CodeChunk,
                                sub_chunks: List[CodeChunk], offset: int = 0):
        """Split by if/while/for blocks."""
        for child in node.children:
            if child.type in ("if_statement", "while_statement", "for_statement",
                              "enhanced_for_statement", "do_statement"):
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
                        sub_chunk = CodeChunk(
                            code=body_code.strip(),
                            language="java",
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


class CppParser(BaseLanguageParser):
    """Parser for C and C++ code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._parser = None
        self._ensure_parser()

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_cpp as tscpp
                from tree_sitter import Language, Parser
                self._language = Language(tscpp.language())
                self._parser = Parser(self._language)
            except ImportError:
                raise ImportError(
                    "tree-sitter-cpp not installed. Run: pip install tree-sitter-cpp"
                )

    def get_language_name(self) -> str:
        return "cpp"

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse C/C++ source and return function/class/struct/namespace chunks."""
        tree = self._parser.parse(bytes(source_code, "utf8"))
        chunks = []
        self._extract_entities(tree.root_node, source_code, file_path, chunks)
        return chunks

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk]):
        """Recursively extract functions, classes, structs, namespaces."""
        for child in node.children:
            if child.type == "function_definition":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "function")
            elif child.type == "class_specifier":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "class")
            elif child.type == "struct_specifier":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "struct")
            elif child.type == "namespace_specifier":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "namespace")
            else:
                self._extract_entities(child, source_code, file_path, chunks)

    def _create_chunk_from_node(self, node, source_code: str, file_path: str,
                                chunks: List[CodeChunk], entity_type: str):
        """Create CodeChunk from tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        code = source_code[start_byte:end_byte]

        name_node = None
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                name_node = child
                break
        entity_name = name_node.text.decode("utf8") if name_node else "<anonymous>"

        chunk = CodeChunk(
            code=code,
            language="cpp",
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )

        split_chunks = self._split_if_needed(chunk)
        chunks.extend(split_chunks)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split C/C++ chunk by internal blocks."""
        source_code = chunk.code
        tree = self._parser.parse(bytes(source_code, "utf8"))

        sub_chunks = []
        self._split_by_control_flow(tree.root_node, source_code, chunk, sub_chunks)

        if len(sub_chunks) <= 1:
            return self._hard_split(chunk)

        return sub_chunks

    def _split_by_control_flow(self, node, source_code: str, parent_chunk: CodeChunk,
                                sub_chunks: List[CodeChunk], offset: int = 0):
        """Split by if/while/for/switch blocks."""
        for child in node.children:
            if child.type in ("if_statement", "while_statement", "for_statement",
                              "switch_statement", "do_statement"):
                body_start = None
                body_end = None
                for c in child.children:
                    if c.type == "compound_statement":
                        body_start = c.start_byte
                        body_end = c.end_byte
                        break

                if body_start is not None:
                    body_code = source_code[body_start - offset:body_end - offset]
                    if self._estimate_tokens(body_code) > self.chunk_size * 0.5:
                        sub_chunk = CodeChunk(
                            code=body_code.strip(),
                            language="cpp",
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


class GoParser(BaseLanguageParser):
    """Parser for Go code using tree-sitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self._parser = None
        self._ensure_parser()

    def _ensure_parser(self):
        """Lazy-load tree-sitter parser."""
        if self._parser is None:
            try:
                import tree_sitter_go as tsgo
                from tree_sitter import Language, Parser
                self._language = Language(tsgo.language())
                self._parser = Parser(self._language)
            except ImportError:
                raise ImportError(
                    "tree-sitter-go not installed. Run: pip install tree-sitter-go"
                )

    def get_language_name(self) -> str:
        return "go"

    def parse(self, source_code: str, file_path: str = "") -> List[CodeChunk]:
        """Parse Go source and return function/method/type chunks."""
        tree = self._parser.parse(bytes(source_code, "utf8"))
        chunks = []
        self._extract_entities(tree.root_node, source_code, file_path, chunks)
        return chunks

    def _extract_entities(self, node, source_code: str, file_path: str, chunks: List[CodeChunk]):
        """Recursively extract functions, methods, type declarations."""
        for child in node.children:
            if child.type == "function_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "function")
            elif child.type == "method_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "method")
            elif child.type == "type_declaration":
                self._create_chunk_from_node(child, source_code, file_path, chunks, "type")
            else:
                self._extract_entities(child, source_code, file_path, chunks)

    def _create_chunk_from_node(self, node, source_code: str, file_path: str,
                                chunks: List[CodeChunk], entity_type: str):
        """Create CodeChunk from tree-sitter node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        code = source_code[start_byte:end_byte]

        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break
        entity_name = name_node.text.decode("utf8") if name_node else "<anonymous>"

        chunk = CodeChunk(
            code=code,
            language="go",
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )

        split_chunks = self._split_if_needed(chunk)
        chunks.extend(split_chunks)

    def _split_by_blocks(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split Go chunk by internal blocks."""
        source_code = chunk.code
        tree = self._parser.parse(bytes(source_code, "utf8"))

        sub_chunks = []
        self._split_by_control_flow(tree.root_node, source_code, chunk, sub_chunks)

        if len(sub_chunks) <= 1:
            return self._hard_split(chunk)

        return sub_chunks

    def _split_by_control_flow(self, node, source_code: str, parent_chunk: CodeChunk,
                                sub_chunks: List[CodeChunk], offset: int = 0):
        """Split by if/for/switch blocks."""
        for child in node.children:
            if child.type in ("if_statement", "for_statement", "switch_statement"):
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
                        sub_chunk = CodeChunk(
                            code=body_code.strip(),
                            language="go",
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
                chunk.metadata.setdefault("source_file", file_path)
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
        # Support both source_file and file_path for compatibility
        file_path = document.metadata.get("source_file") or document.metadata.get("file_path", "")
        return self.split_text(code, file_path)

    def split_documents(self, documents: List) -> List:
        """Split multiple documents."""
        nodes = []
        for doc in documents:
            nodes.extend(self.split_document(doc))
        return nodes
