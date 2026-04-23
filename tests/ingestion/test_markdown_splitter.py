"""Tests for the Markdown splitter."""

import pytest
from profirag.ingestion.splitters import extract_markdown_elements, build_header_chain


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
