# Markdown Splitter Design

## Context

ProfiRAG needs a structured Markdown splitter for RAG retrieval scenarios. Current splitters (TextSplitter, ChineseTextSplitter, ASTSplitter) don't handle Markdown-specific elements like headers, code blocks, and tables. The goal is semantic-aware splitting that preserves header hierarchy, keeps code blocks and tables intact, and repeats header chains in each chunk for better retrieval.

## Architecture

```
src/profirag/ingestion/
├── splitters.py              # Add MarkdownSplitter alongside existing splitters
└── ast_splitter.py           # Code splitter (unchanged)
```

Reuses LlamaIndex's `MarkdownElementNodeParser.extract_elements()` for element parsing.

## Core Components

### 1. MarkdownSplitter Class

Entry point, implements same interface as existing splitters:

```python
class MarkdownSplitter:
    def __init__(
        self,
        chunk_size: int = 512,       # token estimate limit
        chunk_overlap: int = 50,     # overlap between chunks
    )

    def split_text(self, text: str) -> List[TextNode]
    def split_document(self, document: Document) -> List[TextNode]
    def split_documents(self, documents: List[Document]) -> List[TextNode]
```

### 2. Element Types (from MarkdownElementNodeParser)

| Element Type | Split Strategy |
|--------------|----------------|
| `title` | Section boundary, header chain repeated in each chunk |
| `code` | Atomic unit, preserved intact |
| `table` | Atomic unit, preserved intact |
| `text` | Splittable (by paragraph/sentence) |

### 3. Section Builder

Groups elements by title boundaries, maintains heading stack for hierarchy tracking.

### 4. Chunk Assembler

Accumulates elements within a section, flushes when exceeding `chunk_size`:
- New chunk starts with header chain
- Code blocks and tables preserved intact even if oversized
- Text elements split by paragraph/sentence if needed

## Splitting Algorithm

### Phase 1: Extract Elements

```python
elements = extract_elements(text)
# Uses MarkdownElementNodeParser.extract_elements()
# Returns List[Element] with types: title, code, table, text
```

### Phase 2: Build Sections

```python
sections = []
current_section = Section()
heading_stack = []  # List of (level, text)

for element in elements:
    if element.type == "title":
        # Flush current section if non-empty
        if current_section.has_content():
            sections.append(current_section)
        # Update heading stack
        level = element.title_level
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, element.element))
        current_section = Section(heading_stack=heading_stack.copy())
    else:
        current_section.add_element(element)

# Add final section
if current_section.has_content():
    sections.append(current_section)
```

### Phase 3: Chunk Sections

```python
chunks = []
for section in sections:
    header_chain = build_header_chain(section.heading_stack)
    current_chunk_text = header_chain
    current_tokens = estimate_tokens(header_chain)

    for element in section.elements:
        element_tokens = estimate_tokens(element.element)

        # Atomic elements (code/table): always add intact
        if element.type in ("code", "table"):
            if current_tokens + element_tokens > chunk_size and current_chunk_text != header_chain:
                # Flush current chunk first
                chunks.append(create_chunk(current_chunk_text, section))
                current_chunk_text = header_chain
                current_tokens = estimate_tokens(header_chain)
            current_chunk_text += "\n" + element.element
            current_tokens += element_tokens

        # Text elements: may split
        else:
            if current_tokens + element_tokens > chunk_size:
                # Flush current chunk
                chunks.append(create_chunk(current_chunk_text, section))
                current_chunk_text = header_chain
                current_tokens = estimate_tokens(header_chain)

                # If element still exceeds, split it
                if element_tokens > chunk_size:
                    sub_texts = split_text_element(element.element, chunk_size - estimate_tokens(header_chain))
                    for sub_text in sub_texts:
                        chunks.append(create_chunk(header_chain + "\n" + sub_text, section))
                else:
                    current_chunk_text += "\n" + element.element
                    current_tokens += element_tokens
            else:
                current_chunk_text += "\n" + element.element
                current_tokens += element_tokens

    # Flush remaining chunk
    if current_chunk_text.strip():
        chunks.append(create_chunk(current_chunk_text, section))

return chunks
```

## Token Estimation

```python
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # Consistent with ASTSplitter
```

## Header Chain Format

```python
def build_header_chain(heading_stack: List[tuple]) -> str:
    """
    Build header chain from heading stack.

    Args:
        heading_stack: List of (level, text) tuples

    Returns:
        Markdown header chain string, e.g.:
        "# API 文档\n## 用户模块\n### 登录接口"
    """
    lines = []
    for level, text in heading_stack:
        lines.append("#" * level + " " + text)
    return "\n".join(lines)
```

## Metadata

Each TextNode includes:

```python
{
    "header_path": "/H1/H2/",           # Header path (consistent with MarkdownNodeParser)
    "current_heading": "登录接口",       # Most recent heading text
    "heading_level": 3,                 # Current heading level (1-6)
    "has_code_block": True,             # Contains code block
    "has_table": False,                 # Contains table
    "source_doc_id": "...",             # Document ID (if from Document)
}
```

## Edge Cases

1. **No headers** - Treat entire document as one section, split by chunk_size
2. **Oversized code/table** - Preserve intact, log warning if exceeds chunk_size
3. **Empty section** - Skip, don't generate empty chunks
4. **Deep hierarchy** - Header chain may be long; user accepts this tradeoff for RAG retrieval

## Dependencies

No new dependencies. Uses existing LlamaIndex components:
- `llama_index.core.node_parser.relational.markdown_element.MarkdownElementNodeParser`
- `llama_index.core.schema.TextNode, Document`

## Verification

1. Unit tests for element extraction
2. Unit tests for section building with various header hierarchies
3. Unit tests for chunk assembly with chunk_size constraints
4. Integration test with existing pipeline
5. Test edge cases: no headers, oversized elements, empty sections