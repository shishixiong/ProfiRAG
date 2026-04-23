"""Text splitters for chunking documents"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
)
from llama_index.core.node_parser.relational.markdown_element import MarkdownElementNodeParser
from llama_index.core.node_parser.relational.base_element import Element
from llama_index.core.schema import TextNode, Document


# Pattern for markdown image references
IMAGE_REFERENCE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# Pattern for markdown headings
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')


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


def extract_markdown_elements(text: str) -> List[Element]:
    """Extract structured elements from Markdown text.

    Uses LlamaIndex's MarkdownElementNodeParser.extract_elements().
    Post-processes title elements to set title_level based on heading depth.

    Args:
        text: Markdown text content

    Returns:
        List of Element objects with types: title, code, table, text
    """
    parser = MarkdownElementNodeParser()
    elements = parser.extract_elements(text)

    # Post-process: set title_level for title elements
    for element in elements:
        if element.type == "title" and element.element:
            # Find the heading in the original text to determine level
            # Element text has leading space, e.g., " Title 1"
            element_text = element.element.lstrip()
            lines = text.split("\n")
            for line in lines:
                match = HEADING_PATTERN.match(line)
                if match:
                    heading_text = match.group(2).strip()
                    if heading_text == element_text:
                        element.title_level = len(match.group(1))
                        break

    return elements


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


def extract_heading_chain(text: str) -> List[tuple]:
    """Extract hierarchical heading structure from markdown text.

    Args:
        text: Markdown text content

    Returns:
        List of (line_number, level, heading_text) tuples, in document order.
        e.g., [(0, 1, "概述"), (5, 2, "工具介绍"), ...]
    """
    results = []
    lines = text.split("\n")
    for line_num, line in enumerate(lines):
        match = HEADING_PATTERN.match(line.strip())
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            results.append((line_num, level, heading_text))
    return results


def get_heading_chain_for_position(
    heading_chain: List[tuple],
    position_line: int
) -> tuple:
    """Get the heading chain (ancestors) for content at a given line position.

    Args:
        heading_chain: List of (line_number, level, heading_text) from extract_heading_chain
        position_line: The line number to find heading context for

    Returns:
        Tuple of (current_heading, heading_chain_list)
        - current_heading: The most recent heading text (or "")
        - heading_chain_list: List of ancestor heading texts, e.g., ["1 概述", "1.1 工具介绍"]
    """
    if not heading_chain:
        return "", []

    # Find the last heading that appears before or at position_line
    current_heading = ""
    ancestors: List[str] = []

    for line_num, level, heading_text in heading_chain:
        if line_num > position_line:
            break
        current_heading = heading_text
        # Rebuild ancestors: headings at levels 1 through current_level-1
        ancestors = [h for _, l, h in heading_chain if l < level and _ <= position_line]

    return current_heading, ancestors


def find_images_in_chunk(chunk_text: str, image_map: Dict[str, Dict]) -> List[str]:
    """Find image IDs referenced in chunk text.

    Args:
        chunk_text: Text content of the chunk
        image_map: Document's image_map dictionary

    Returns:
        List of image IDs found in the chunk
    """
    if not image_map:
        return []

    found_ids = []
    for match in IMAGE_REFERENCE_PATTERN.finditer(chunk_text):
        image_path = match.group(2)
        # Find matching image_id from image_map
        for img_id, img_info in image_map.items():
            if img_info.get("path") == image_path or \
               img_info.get("filename") in image_path:
                found_ids.append(img_id)
                break

    return found_ids


class TextSplitter:
    """Text splitter for document chunking.

    Supports multiple splitting strategies:
    - sentence: Split by sentences with overlap
    - token: Split by token count
    - semantic: Split by semantic similarity
    """

    def __init__(
        self,
        splitter_type: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embed_model: Optional[Any] = None,
        **kwargs
    ):
        """Initialize text splitter.

        Args:
            splitter_type: Type of splitter ("sentence", "token", "semantic")
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            embed_model: Embedding model for semantic splitter
            **kwargs: Additional arguments
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model = embed_model
        self.kwargs = kwargs

        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create appropriate splitter based on type."""
        if self.splitter_type == "sentence":
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **self.kwargs
            )
        elif self.splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **self.kwargs
            )
        elif self.splitter_type == "semantic":
            if self.embed_model is None:
                raise ValueError("Semantic splitter requires embed_model")
            return SemanticSplitterNodeParser(
                embed_model=self.embed_model,
                chunk_size=self.chunk_size,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")

    def split_text(self, text: str) -> List[TextNode]:
        """Split text into nodes.

        Args:
            text: Text to split

        Returns:
            List of TextNode objects
        """
        doc = Document(text=text)
        return self._splitter.get_nodes_from_documents([doc])

    def split_document(self, document: Document) -> List[TextNode]:
        """Split a Document into nodes.

        Args:
            document: Document to split

        Returns:
            List of TextNode objects
        """
        # Get image_map from document BEFORE modifying
        image_map = document.metadata.get("image_map", {})

        # Extract heading chain from original text for heading metadata
        heading_chain = extract_heading_chain(document.text)

        # Create reduced metadata (without large image_map) before splitting
        # LlamaIndex splitter checks metadata length against chunk_size
        reduced_metadata = {
            k: v for k, v in document.metadata.items()
            if k != "image_map"  # Skip the potentially large image_map
        }

        # Temporarily modify document metadata for splitting
        original_metadata = document.metadata.copy()
        document.metadata = reduced_metadata

        # Split the document with reduced metadata
        nodes = self._splitter.get_nodes_from_documents([document])

        # Restore original metadata to document
        document.metadata = original_metadata

        # Add image-related and heading metadata to each node
        for node in nodes:
            node.metadata.update(reduced_metadata)
            # Store document ID in metadata
            if document.doc_id:
                node.metadata["source_doc_id"] = document.doc_id

            # Find heading context for this node
            if heading_chain:
                # Match node text back to original document to find position
                node_start_in_original = document.text.find(node.text[:50]) if len(node.text) >= 50 else document.text.find(node.text)
                if node_start_in_original >= 0:
                    current_heading, heading_chain_list = get_heading_chain_for_position(
                        heading_chain, document.text[:node_start_in_original].count("\n")
                    )
                    if current_heading:
                        node.metadata["current_heading"] = current_heading
                    if heading_chain_list:
                        node.metadata["heading_chain"] = heading_chain_list

            # Find images in this chunk
            chunk_images = find_images_in_chunk(node.text, image_map)
            node.metadata["chunk_images"] = chunk_images
            node.metadata["has_images"] = len(chunk_images) > 0
            # Store image paths directly for easy access
            if chunk_images:
                node.metadata["image_paths"] = [
                    image_map.get(img_id, {}).get("path", "")
                    for img_id in chunk_images
                    if image_map.get(img_id, {}).get("path")
                ]
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

    def update_chunk_size(self, chunk_size: int) -> None:
        """Update chunk size and recreate splitter.

        Args:
            chunk_size: New chunk size
        """
        self.chunk_size = chunk_size
        self._splitter = self._create_splitter()

    def update_overlap(self, chunk_overlap: int) -> None:
        """Update chunk overlap and recreate splitter.

        Args:
            chunk_overlap: New overlap value
        """
        self.chunk_overlap = chunk_overlap
        self._splitter = self._create_splitter()


class ChineseTextSplitter:
    """Text splitter optimized for Chinese text.

    Handles Chinese-specific sentence boundaries and punctuation.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """Initialize Chinese text splitter.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap in characters
            **kwargs: Additional arguments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into Chinese sentences.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        import re
        # Split on Chinese punctuation
        pattern = r'([。！？；\n]+)'
        parts = re.split(pattern, text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
            if sentence.strip():
                sentences.append(sentence.strip())

        # Handle remaining text
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        return sentences

    def split_text(self, text: str) -> List[TextNode]:
        """Split Chinese text into nodes.

        Args:
            text: Chinese text

        Returns:
            List of TextNode objects with _char_start metadata for heading context.
        """
        HARD_LIMIT = 4000

        sentences = self._split_sentences(text)
        nodes = []
        current_chunk = ""
        # Cumulative character offset in the original text
        # Represents the starting position of the current_chunk in original text
        offset = 0

        def flush_chunk(chunk: str, flush_offset: int) -> None:
            """Helper to yield a chunk with its starting position."""
            if chunk.strip():
                nodes.append(
                    TextNode(text=chunk.strip(), metadata={"_char_start": flush_offset})
                )

        for sentence in sentences:
            sentence_len = len(sentence)

            # Handle very long sentences (exceed chunk_size or hard limit)
            if sentence_len > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk.strip():
                    flush_chunk(current_chunk, offset - len(current_chunk))
                    current_chunk = ""

                # Split long sentence into smaller pieces
                split_size = min(self.chunk_size, HARD_LIMIT)
                piece_offset = offset
                for i in range(0, sentence_len, split_size - self.chunk_overlap):
                    chunk_piece = sentence[i:i + split_size]
                    if chunk_piece.strip():
                        flush_chunk(chunk_piece, piece_offset + i)
                offset += sentence_len
                continue

            # Check if adding sentence would exceed chunk_size
            if len(current_chunk) + sentence_len > self.chunk_size:
                if current_chunk:
                    flush_chunk(current_chunk, offset - len(current_chunk))
                # Add overlap: keep last chunk_overlap chars of current_chunk
                overlap = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap:] + sentence
                offset += sentence_len  # advance by new sentence
            else:
                current_chunk += sentence
                offset += sentence_len

        # Add remaining chunk
        if current_chunk.strip():
            if len(current_chunk) > HARD_LIMIT:
                piece_offset = offset - len(current_chunk)
                for i in range(0, len(current_chunk), HARD_LIMIT - self.chunk_overlap):
                    chunk_piece = current_chunk[i:i + HARD_LIMIT]
                    if chunk_piece.strip():
                        flush_chunk(chunk_piece, piece_offset + i)
            else:
                flush_chunk(current_chunk, offset - len(current_chunk))

        return nodes

    def split_document(self, document: Document) -> List[TextNode]:
        """Split a Chinese Document into nodes.

        Args:
            document: Document with Chinese text

        Returns:
            List of TextNode objects
        """
        # Extract heading chain from original text
        heading_chain = extract_heading_chain(document.text)

        nodes = self.split_text(document.text)
        # Get image_map from document
        image_map = document.metadata.get("image_map", {})
        # Copy base metadata, excluding large image_map
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if k != "image_map"  # Skip the potentially large image_map
        }
        # Update metadata with image propagation and heading context
        for node in nodes:
            node.metadata.update(base_metadata)
            # Store document ID in metadata instead
            if document.doc_id:
                node.metadata["source_doc_id"] = document.doc_id

            # Find heading context for this node using actual character position
            if heading_chain:
                char_start = node.metadata.get("_char_start", 0)
                # Convert character offset to line count in original document
                line_count = document.text[:char_start].count("\n") if char_start > 0 else 0
                current_heading, heading_chain_list = get_heading_chain_for_position(
                    heading_chain, line_count
                )
                if current_heading:
                    node.metadata["current_heading"] = current_heading
                if heading_chain_list:
                    node.metadata["heading_chain"] = heading_chain_list

            # Remove internal tracking key
            node.metadata.pop("_char_start", None)

            # Find images in this chunk
            chunk_images = find_images_in_chunk(node.text, image_map)
            node.metadata["chunk_images"] = chunk_images
            node.metadata["has_images"] = len(chunk_images) > 0
            # Store image paths directly for easy access
            if chunk_images:
                node.metadata["image_paths"] = [
                    image_map.get(img_id, {}).get("path", "")
                    for img_id in chunk_images
                    if image_map.get(img_id, {}).get("path")
                ]
        return nodes

    def split_documents(self, documents: List[Document]) -> List[TextNode]:
        """Split multiple Chinese Documents into nodes.

        Args:
            documents: List of Documents with Chinese text

        Returns:
            List of TextNode objects
        """
        all_nodes = []
        for doc in documents:
            nodes = self.split_document(doc)
            all_nodes.extend(nodes)
        return all_nodes