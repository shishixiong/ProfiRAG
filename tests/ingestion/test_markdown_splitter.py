"""Tests for the Markdown splitter."""

import pytest
from profirag.ingestion.splitters import (
    extract_markdown_elements,
    build_header_chain,
    build_sections,
    Section,
    chunk_sections,
)
from llama_index.core.node_parser.relational.base_element import Element


class TestExtractMarkdownElements:
    """Unit tests for element extraction."""

    def test_extract_simple_headers(self):
        """Headers are extracted with correct levels."""
        text = "# Title 1\n## Title 2\n### Title 3\nContent here"
        elements = extract_markdown_elements(text)
        assert len(elements) == 4
        assert elements[0].type == "title"
        assert elements[0].title_level == 1
        assert elements[0].element.strip() == "Title 1"

    def test_extract_code_block(self):
        """Code blocks are extracted intact."""
        text = "```python\ndef hello():\n    pass\n```\nText after"
        elements = extract_markdown_elements(text)
        assert len(elements) == 2
        assert elements[0].type == "code"
        assert "def hello()" in elements[0].element

    def test_extract_table(self):
        """Markdown tables are extracted intact."""
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n\nText after"
        elements = extract_markdown_elements(text)
        assert len(elements) == 2
        assert elements[0].type == "table"

    def test_header_inside_code_block_ignored(self):
        """Headers inside code blocks are not parsed as titles."""
        text = "```markdown\n# Fake Header\n```\n# Real Header"
        elements = extract_markdown_elements(text)
        title_elements = [e for e in elements if e.type == "title"]
        assert len(title_elements) == 1
        assert title_elements[0].element.strip() == "Real Header"


class TestBuildHeaderChain:
    """Unit tests for header chain building."""

    def test_single_header(self):
        """Single header produces one line."""
        heading_stack = [(1, "Title")]
        result = build_header_chain(heading_stack)
        assert result == "# Title"

    def test_multiple_headers(self):
        """Multiple headers produce multi-line chain."""
        heading_stack = [(1, "API"), (2, "Users"), (3, "Login")]
        result = build_header_chain(heading_stack)
        assert result == "# API\n## Users\n### Login"

    def test_empty_stack(self):
        """Empty stack produces empty string."""
        result = build_header_chain([])
        assert result == ""

    def test_header_level_correctness(self):
        """Header level determines number of # symbols."""
        heading_stack = [(2, "Section"), (4, "Subsubsection")]
        result = build_header_chain(heading_stack)
        assert "## Section" in result
        assert "#### Subsubsection" in result


class TestSection:
    """Unit tests for Section dataclass."""

    def test_section_creation(self):
        """Section stores heading_stack and elements."""
        section = Section(heading_stack=[(1, "Title")])
        assert section.heading_stack == [(1, "Title")]
        assert section.elements == []

    def test_section_add_element(self):
        """Elements can be added to section."""
        section = Section(heading_stack=[(1, "Title")])
        elem = Element(id="test", type="text", element="Content")
        section.add_element(elem)
        assert len(section.elements) == 1
        assert section.elements[0].type == "text"

    def test_section_has_content(self):
        """has_content returns True when elements exist."""
        section = Section(heading_stack=[(1, "Title")])
        assert not section.has_content()
        section.add_element(Element(id="test", type="text", element="x"))
        assert section.has_content()

    def test_section_heading_stack_copy(self):
        """Heading stack is independent per section."""
        section1 = Section(heading_stack=[(1, "A"), (2, "B")])
        section2 = Section(heading_stack=[(1, "A")])
        assert len(section1.heading_stack) == 2
        assert len(section2.heading_stack) == 1


class TestBuildSections:
    """Unit tests for section building from elements."""

    def test_single_section(self):
        """Single header creates one section."""
        elements = [
            Element(id="0", type="title", element="Title", title_level=1),
            Element(id="1", type="text", element="Content"),
        ]
        sections = build_sections(elements)
        assert len(sections) == 1
        assert sections[0].heading_stack == [(1, "Title")]

    def test_multiple_sections(self):
        """Multiple headers create multiple sections."""
        elements = [
            Element(id="0", type="title", element="A", title_level=1),
            Element(id="1", type="text", element="Content A"),
            Element(id="2", type="title", element="B", title_level=1),
            Element(id="3", type="text", element="Content B"),
        ]
        sections = build_sections(elements)
        assert len(sections) == 2
        assert sections[0].heading_stack == [(1, "A")]
        assert sections[1].heading_stack == [(1, "B")]

    def test_nested_headers(self):
        """Nested headers maintain hierarchy stack."""
        elements = [
            Element(id="0", type="title", element="API", title_level=1),
            Element(id="1", type="text", element="Intro"),
            Element(id="2", type="title", element="Users", title_level=2),
            Element(id="3", type="text", element="User content"),
            Element(id="4", type="title", element="Login", title_level=3),
            Element(id="5", type="text", element="Login content"),
        ]
        sections = build_sections(elements)
        assert len(sections) == 3
        assert sections[0].heading_stack == [(1, "API")]
        assert sections[1].heading_stack == [(1, "API"), (2, "Users")]
        assert sections[2].heading_stack == [(1, "API"), (2, "Users"), (3, "Login")]

    def test_header_level_jump(self):
        """Jumping header levels (H1 to H3) pops intermediate levels."""
        elements = [
            Element(id="0", type="title", element="A", title_level=1),
            Element(id="1", type="title", element="B", title_level=3),
        ]
        sections = build_sections(elements)
        assert sections[0].heading_stack == [(1, "A")]
        # H3 pops H2 (not in stack), keeps H1, adds H3
        assert sections[1].heading_stack == [(1, "A"), (3, "B")]

    def test_no_headers(self):
        """No headers creates single section with empty heading_stack."""
        elements = [
            Element(id="0", type="text", element="Plain content"),
            Element(id="1", type="text", element="More content"),
        ]
        sections = build_sections(elements)
        assert len(sections) == 1
        assert sections[0].heading_stack == []


class TestChunkSections:
    """Unit tests for chunking sections into TextNode chunks."""

    def test_small_section_single_chunk(self):
        """Small section produces single chunk."""
        section = Section(
            heading_stack=[(1, "Title")],
            elements=[Element(id="0", type="text", element="Short content")]
        )
        chunks = chunk_sections([section], chunk_size=512)
        assert len(chunks) == 1
        assert "# Title" in chunks[0].text
        assert "Short content" in chunks[0].text

    def test_header_chain_included(self):
        """Each chunk starts with header chain."""
        section = Section(
            heading_stack=[(1, "API"), (2, "Users")],
            elements=[Element(id="0", type="text", element="Content")]
        )
        chunks = chunk_sections([section], chunk_size=512)
        assert "# API" in chunks[0].text
        assert "## Users" in chunks[0].text

    def test_empty_section_no_chunks(self):
        """Empty section produces no chunks."""
        section = Section(heading_stack=[(1, "Title")], elements=[])
        chunks = chunk_sections([section], chunk_size=512)
        assert len(chunks) == 0

    def test_metadata_header_path(self):
        """Chunk has header_path metadata."""
        section = Section(
            heading_stack=[(1, "API"), (2, "Users")],
            elements=[Element(id="0", type="text", element="Content")]
        )
        chunks = chunk_sections([section], chunk_size=512)
        assert chunks[0].metadata["header_path"] == "/API/Users/"
        assert chunks[0].metadata["current_heading"] == "Users"
        assert chunks[0].metadata["heading_level"] == 2


class TestChunkSectionsAtomic:
    """Tests for atomic element preservation."""

    def test_code_block_preserved_intact(self):
        """Code blocks are preserved intact even when chunk splits."""
        long_text = "x" * 1000  # ~250 tokens
        section = Section(
            heading_stack=[(1, "Title")],
            elements=[
                Element(id="0", type="text", element="Intro"),
                Element(id="1", type="code", element=f"```python\n{long_text}\n```"),
            ]
        )
        chunks = chunk_sections([section], chunk_size=100)
        assert len(chunks) >= 2
        code_chunks = [c for c in chunks if "```python" in c.text]
        assert len(code_chunks) >= 1
        for c in code_chunks:
            assert long_text in c.text  # Code is intact

    def test_table_preserved_intact(self):
        """Tables are preserved intact."""
        section = Section(
            heading_stack=[(1, "Title")],
            elements=[
                Element(id="0", type="table", element="| A | B |\n|---|---|\n| 1 | 2 |"),
            ]
        )
        chunks = chunk_sections([section], chunk_size=50)
        assert len(chunks) == 1
        assert "| A | B |" in chunks[0].text
        assert chunks[0].metadata["has_table"] == True


class TestChunkSectionsRepetition:
    """Tests for header chain repetition."""

    def test_header_chain_repeats_in_split_chunks(self):
        """When section splits, header chain repeats in each chunk."""
        long_content = "Paragraph one with enough text. " * 50  # ~400 tokens
        long_content2 = "Paragraph two with more text. " * 50  # ~400 tokens
        section = Section(
            heading_stack=[(1, "API"), (2, "Users")],
            elements=[
                Element(id="0", type="text", element=long_content),
                Element(id="1", type="text", element=long_content2),
            ]
        )
        chunks = chunk_sections([section], chunk_size=300)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "# API" in chunk.text
            assert "## Users" in chunk.text
