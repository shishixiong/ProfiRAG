# Markdown Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a structured Markdown splitter that preserves header hierarchy, keeps code blocks and tables intact, and repeats header chains in each chunk for RAG retrieval.

**Architecture:** Add `MarkdownSplitter` class to `splitters.py`, reusing LlamaIndex's `MarkdownElementNodeParser.extract_elements()` for element parsing. Three-phase algorithm: extract elements → build sections → chunk sections.

**Tech Stack:** Python, LlamaIndex (MarkdownElementNodeParser, TextNode, Document), pytest

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/profirag/ingestion/splitters.py` | MarkdownSplitter class + helper functions |
| `tests/ingestion/test_markdown_splitter.py` | Unit tests for MarkdownSplitter |

---

### Task 1: Test File Setup and Element Extraction Helper

**Files:**
- Create: `tests/ingestion/test_markdown_splitter.py`
- Modify: `src/profirag/ingestion/splitters.py:1-20`

- [ ] **Step 1: Write the failing test for element extraction**

```python
"""Tests for the Markdown splitter."""

import pytest
from profirag.ingestion.splitters import MarkdownSplitter, extract_markdown_elements


class TestExtractMarkdownElements:
    """Unit tests for element extraction."""

    def test_extract_simple_headers(self):
        """Headers are extracted with correct levels."""
        text = "# Title 1\n## Title 2\n### Title 3\nContent here"
        elements = extract_markdown_elements(text)
        assert len(elements) == 4
        assert elements[0].type == "title"
        assert elements[0].title_level == 1
        assert elements[0].element == "Title 1"

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
        assert title_elements[0].element == "Real Header"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py -v`
Expected: FAIL with "cannot import name 'extract_markdown_elements'"

- [ ] **Step 3: Write minimal implementation - extract_markdown_elements function**

Add to `src/profirag/ingestion/splitters.py` after imports:

```python
from llama_index.core.node_parser.relational.markdown_element import MarkdownElementNodeParser
from llama_index.core.node_parser.relational.base_element import Element


def extract_markdown_elements(text: str) -> List[Element]:
    """Extract structured elements from Markdown text.

    Uses LlamaIndex's MarkdownElementNodeParser.extract_elements().

    Args:
        text: Markdown text content

    Returns:
        List of Element objects with types: title, code, table, text
    """
    parser = MarkdownElementNodeParser()
    elements = parser.extract_elements(text)
    return elements
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestExtractMarkdownElements -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add extract_markdown_elements helper function"
```

---

### Task 2: Header Chain Builder

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the failing test for build_header_chain**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestBuildHeaderChain -v`
Expected: FAIL with "cannot import name 'build_header_chain'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/profirag/ingestion/splitters.py`:

```python
def build_header_chain(heading_stack: List[tuple]) -> str:
    """Build header chain from heading stack.

    Args:
        heading_stack: List of (level, text) tuples

    Returns:
        Markdown header chain string, e.g. "# API\\n## Users\\n### Login"
    """
    if not heading_stack:
        return ""
    lines = []
    for level, text in heading_stack:
        lines.append("#" * level + " " + text)
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestBuildHeaderChain -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add build_header_chain helper function"
```

---

### Task 3: Section Builder

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the failing test for Section class**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestSection -v`
Expected: FAIL with "cannot import name 'Section'"

- [ ] **Step 3: Write minimal implementation - Section dataclass**

Add to `src/profirag/ingestion/splitters.py` after imports:

```python
from dataclasses import dataclass, field


@dataclass
class Section:
    """Represents a section of Markdown content under a header hierarchy."""
    heading_stack: List[tuple] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)

    def add_element(self, element: Element) -> None:
        """Add an element to this section."""
        self.elements.append(element)

    def has_content(self) -> bool:
        """Check if section has any content elements."""
        return len(self.elements) > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestSection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add Section dataclass for grouping elements"
```

---

### Task 4: Section Builder Function

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the failing test for build_sections**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestBuildSections -v`
Expected: FAIL with "cannot import name 'build_sections'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/profirag/ingestion/splitters.py`:

```python
def build_sections(elements: List[Element]) -> List[Section]:
    """Build sections from elements by grouping by title boundaries.

    Args:
        elements: List of Element objects from extract_markdown_elements

    Returns:
        List of Section objects with heading_stack and elements
    """
    sections = []
    heading_stack: List[tuple] = []
    current_section = Section(heading_stack=heading_stack.copy())

    for element in elements:
        if element.type == "title":
            # Flush current section if non-empty
            if current_section.has_content():
                sections.append(current_section)

            # Update heading stack
            level = element.title_level
            # Pop headers of equal or higher level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, element.element))

            # Create new section with current heading stack
            current_section = Section(heading_stack=heading_stack.copy())
        else:
            current_section.add_element(element)

    # Add final section if non-empty
    if current_section.has_content():
        sections.append(current_section)

    return sections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestBuildSections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add build_sections function for grouping by headers"
```

---

### Task 5: Chunk Assembler - Basic

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the failing test for chunk_sections basic**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
class TestChunkSections:
    """Unit tests for chunk assembly from sections."""

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections -v`
Expected: FAIL with "cannot import name 'chunk_sections'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/profirag/ingestion/splitters.py`:

```python
def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // 4


def chunk_sections(
    sections: List[Section],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[TextNode]:
    """Assemble chunks from sections with chunk_size constraints.

    Args:
        sections: List of Section objects
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of TextNode objects
    """
    from llama_index.core.schema import TextNode

    chunks = []
    for section in sections:
        if not section.has_content():
            continue

        header_chain = build_header_chain(section.heading_stack)
        header_tokens = estimate_tokens(header_chain)

        current_text = header_chain
        current_tokens = header_tokens if header_chain else 0

        for element in section.elements:
            element_text = element.element
            element_tokens = estimate_tokens(element_text)

            # Check if adding element would exceed chunk_size
            if current_tokens + element_tokens + 1 > chunk_size:
                # Flush current chunk if it has content beyond header
                if current_text.strip() and current_text != header_chain:
                    chunks.append(create_chunk_node(current_text, section))
                    current_text = header_chain
                    current_tokens = header_tokens if header_chain else 0

            # Add element to current chunk
            if current_text:
                current_text += "\n" + element_text
            else:
                current_text = element_text
            current_tokens += element_tokens

        # Flush remaining chunk
        if current_text.strip():
            chunks.append(create_chunk_node(current_text, section))

    return chunks


def create_chunk_node(text: str, section: Section) -> TextNode:
    """Create a TextNode from chunk text with metadata."""
    from llama_index.core.schema import TextNode

    header_path = "/" + "/".join(h[1] for h in section.heading_stack) + "/"
    current_heading = section.heading_stack[-1][1] if section.heading_stack else ""
    heading_level = section.heading_stack[-1][0] if section.heading_stack else 0

    has_code = any(e.type == "code" for e in section.elements)
    has_table = any(e.type == "table" for e in section.elements)

    return TextNode(
        text=text,
        metadata={
            "header_path": header_path,
            "current_heading": current_heading,
            "heading_level": heading_level,
            "has_code_block": has_code,
            "has_table": has_table,
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_small_section_single_chunk -v`
Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_header_chain_included -v`
Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_empty_section_no_chunks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add chunk_sections and create_chunk_node functions"
```

---

### Task 6: Chunk Assembler - Atomic Elements (Code/Table)

**Files:**
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the test for atomic element preservation**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
    def test_code_block_preserved_intact(self):
        """Code blocks are preserved intact even when chunk splits."""
        # Create a section where code block would exceed chunk_size
        long_text = "x" * 1000  # ~250 tokens
        section = Section(
            heading_stack=[(1, "Title")],
            elements=[
                Element(id="0", type="text", element="Intro"),
                Element(id="1", type="code", element=f"```python\n{long_text}\n```"),
            ]
        )
        chunks = chunk_sections([section], chunk_size=100)  # small chunk_size
        # Code block should be in its own chunk with header
        assert len(chunks) >= 2
        # Find chunk with code
        code_chunks = [c for c in chunks if "has_code_block" in c.metadata and c.metadata["has_code_block"]]
        assert len(code_chunks) >= 1
        # Code should be intact
        for c in code_chunks:
            if "```python" in c.text:
                assert long_text in c.text

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
```

- [ ] **Step 2: Run test to verify behavior**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_code_block_preserved_intact -v`
Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_table_preserved_intact -v`
Expected: May PASS or need adjustment

- [ ] **Step 3: Fix implementation if needed**

The existing `chunk_sections` should handle this, but if tests fail, update the logic to ensure atomic elements are always added intact regardless of size. The key is: when encountering code/table, flush current chunk first (if has content beyond header), then add the atomic element to a fresh chunk.

- [ ] **Step 4: Run test to verify passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "test: add tests for atomic element preservation"
```

---

### Task 7: Chunk Assembler - Header Chain Repetition

**Files:**
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the test for header chain repetition**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
    def test_header_chain_repeats_in_split_chunks(self):
        """When section splits, header chain repeats in each chunk."""
        # Content that exceeds chunk_size
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
        # Each chunk should start with header chain
        for chunk in chunks:
            assert "# API" in chunk.text
            assert "## Users" in chunk.text
```

- [ ] **Step 2: Run test to verify behavior**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_header_chain_repeats_in_split_chunks -v`
Expected: Should PASS if implementation is correct

- [ ] **Step 3: Fix implementation if needed**

If test fails, the header repetition logic needs adjustment. Current logic should already repeat headers when flushing a chunk and starting a new one.

- [ ] **Step 4: Run test to verify passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestChunkSections::test_header_chain_repeats_in_split_chunks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "test: add test for header chain repetition across chunks"
```

---

### Task 8: MarkdownSplitter Class

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the failing test for MarkdownSplitter class**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
class TestMarkdownSplitter:
    """Unit tests for MarkdownSplitter class."""

    def test_splitter_init_defaults(self):
        """Default constructor sets sensible values."""
        splitter = MarkdownSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_splitter_init_custom(self):
        """Custom constructor values are stored."""
        splitter = MarkdownSplitter(chunk_size=256, chunk_overlap=20)
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 20

    def test_split_text_returns_nodes(self):
        """split_text returns list of TextNode."""
        splitter = MarkdownSplitter()
        text = "# Title\nContent here"
        nodes = splitter.split_text(text)
        assert isinstance(nodes, list)
        assert len(nodes) >= 1
        assert all(hasattr(n, "text") for n in nodes)

    def test_split_text_with_headers(self):
        """split_text handles headers correctly."""
        splitter = MarkdownSplitter()
        text = "# API\n## Users\nUser content here"
        nodes = splitter.split_text(text)
        assert len(nodes) >= 1
        assert nodes[0].metadata["header_path"] == "/API/Users/"
        assert nodes[0].metadata["current_heading"] == "Users"

    def test_split_document(self):
        """split_document handles Document objects."""
        splitter = MarkdownSplitter()
        from llama_index.core.schema import Document
        doc = Document(text="# Title\nContent", metadata={"file_path": "/test.md"})
        nodes = splitter.split_document(doc)
        assert len(nodes) >= 1
        assert nodes[0].metadata.get("source_doc_id") == doc.doc_id or "file_path" in nodes[0].metadata

    def test_split_documents(self):
        """split_documents handles multiple Documents."""
        splitter = MarkdownSplitter()
        from llama_index.core.schema import Document
        docs = [
            Document(text="# A\nContent A"),
            Document(text="# B\nContent B"),
        ]
        nodes = splitter.split_documents(docs)
        assert len(nodes) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestMarkdownSplitter -v`
Expected: FAIL with "cannot import name 'MarkdownSplitter'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/profirag/ingestion/splitters.py`:

```python
class MarkdownSplitter:
    """Markdown splitter for structured chunking.

    Preserves header hierarchy, keeps code blocks and tables intact,
    and repeats header chains in each chunk for RAG retrieval.

    Usage::

        splitter = MarkdownSplitter()
        nodes = splitter.split_text("# Title\\nContent")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """Initialize the Markdown splitter.

        Args:
            chunk_size: Maximum estimated tokens per chunk
            chunk_overlap: Overlap between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[TextNode]:
        """Split Markdown text into nodes.

        Args:
            text: Markdown text content

        Returns:
            List of TextNode objects
        """
        elements = extract_markdown_elements(text)
        sections = build_sections(elements)
        return chunk_sections(sections, self.chunk_size, self.chunk_overlap)

    def split_document(self, document: Document) -> List[TextNode]:
        """Split a Document into nodes.

        Args:
            document: Document with Markdown text

        Returns:
            List of TextNode objects
        """
        nodes = self.split_text(document.text)

        # Copy base metadata to each node
        base_metadata = {k: v for k, v in document.metadata.items()}
        for node in nodes:
            node.metadata.update(base_metadata)
            if document.doc_id:
                node.metadata["source_doc_id"] = document.doc_id

        return nodes

    def split_documents(self, documents: List[Document]) -> List[TextNode]:
        """Split multiple Documents into nodes.

        Args:
            documents: List of Documents

        Returns:
            List of TextNode objects
        """
        all_nodes = []
        for doc in documents:
            nodes = self.split_document(doc)
            all_nodes.extend(nodes)
        return all_nodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestMarkdownSplitter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py src/profirag/ingestion/splitters.py
git commit -m "feat: add MarkdownSplitter class with split_text/document methods"
```

---

### Task 9: Edge Cases - No Headers

**Files:**
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write the test for no headers case**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
class TestMarkdownSplitterEdgeCases:
    """Edge case tests for MarkdownSplitter."""

    def test_no_headers_plain_text(self):
        """Document without headers is split by chunk_size."""
        splitter = MarkdownSplitter(chunk_size=100)
        text = "Plain text paragraph one. " * 20 + "\n" + "Plain text paragraph two. " * 20
        nodes = splitter.split_text(text)
        assert len(nodes) >= 2
        # No header metadata
        for node in nodes:
            assert node.metadata["header_path"] == "/"
            assert node.metadata["current_heading"] == ""
            assert node.metadata["heading_level"] == 0

    def test_no_headers_with_code_block(self):
        """Code block in headerless doc is preserved."""
        splitter = MarkdownSplitter(chunk_size=100)
        text = "Some intro text.\n```python\ndef foo(): pass\n```"
        nodes = splitter.split_text(text)
        code_nodes = [n for n in nodes if n.metadata.get("has_code_block")]
        assert len(code_nodes) >= 1
```

- [ ] **Step 2: Run test to verify behavior**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestMarkdownSplitterEdgeCases -v`
Expected: PASS (existing implementation should handle this)

- [ ] **Step 3: Fix if needed and commit**

```bash
git add tests/ingestion/test_markdown_splitter.py
git commit -m "test: add edge case tests for headerless documents"
```

---

### Task 10: Integration Test - Full Pipeline

**Files:**
- Modify: `tests/ingestion/test_markdown_splitter.py`

- [ ] **Step 1: Write integration test**

Add to `tests/ingestion/test_markdown_splitter.py`:

```python
class TestMarkdownSplitterIntegration:
    """Integration tests for realistic Markdown documents."""

    def test_full_api_document(self):
        """Test realistic API documentation structure."""
        splitter = MarkdownSplitter(chunk_size=300)
        text = """
# API Documentation

This document describes the API endpoints.

## User Module

### Login Endpoint

POST /api/login

Request body:
```json
{
    "username": "string",
    "password": "string"
}
```

Response:
| Field | Type | Description |
|-------|------|-------------|
| token | string | Auth token |
| expires | int | Expiry timestamp |

### Logout Endpoint

POST /api/logout

## Admin Module

### List Users

GET /api/admin/users

Returns a list of all users.
"""
        nodes = splitter.split_text(text)

        # Should have multiple chunks
        assert len(nodes) >= 3

        # Check header metadata on various nodes
        user_nodes = [n for n in nodes if "Users" in n.metadata.get("header_path", "")]
        assert len(user_nodes) >= 1

        # Check code block and table metadata
        code_nodes = [n for n in nodes if n.metadata.get("has_code_block")]
        table_nodes = [n for n in nodes if n.metadata.get("has_table")]
        assert len(code_nodes) >= 1
        assert len(table_nodes) >= 1

        # Header chain should appear in nodes
        login_nodes = [n for n in nodes if "Login" in n.metadata.get("current_heading", "") or "Login" in n.text]
        assert len(login_nodes) >= 1
        for node in login_nodes:
            assert "# API Documentation" in node.text or "## User Module" in node.text
```

- [ ] **Step 2: Run test**

Run: `pytest tests/ingestion/test_markdown_splitter.py::TestMarkdownSplitterIntegration -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/ingestion/test_markdown_splitter.py
git commit -m "test: add integration test for realistic API documentation"
```

---

### Task 11: Final Verification and Export

**Files:**
- Modify: `src/profirag/ingestion/splitters.py`

- [ ] **Step 1: Ensure all exports are in place**

Check that `splitters.py` exports `MarkdownSplitter` properly. Add to top of file if needed:

```python
# Ensure MarkdownSplitter is exported
__all__ = [
    "TextSplitter",
    "ChineseTextSplitter",
    "MarkdownSplitter",
    # ... other exports
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ingestion/test_markdown_splitter.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing splitter tests to ensure no breakage**

Run: `pytest tests/ingestion/ -v`
Expected: All tests PASS (including existing ast_splitter tests)

- [ ] **Step 4: Commit**

```bash
git add src/profirag/ingestion/splitters.py
git commit -m "feat: finalize MarkdownSplitter export"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Each spec section maps to a task
  - Element extraction → Task 1
  - Header chain → Task 2
  - Section builder → Task 3, 4
  - Chunk assembler → Task 5, 6, 7
  - MarkdownSplitter class → Task 8
  - Edge cases → Task 9
  - Integration → Task 10
- [x] **No placeholders:** All code and commands are concrete
- [x] **Type consistency:** Element, Section, TextNode types match across tasks