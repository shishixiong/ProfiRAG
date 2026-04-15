"""Text splitters for chunking documents"""

import re
from typing import List, Optional, Any, Dict
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
)
from llama_index.core.schema import TextNode, Document


# Pattern for markdown image references
IMAGE_REFERENCE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


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

        # Add image-related metadata to each node
        for node in nodes:
            node.metadata.update(reduced_metadata)
            # Store document ID in metadata
            if document.doc_id:
                node.metadata["source_doc_id"] = document.doc_id
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
            List of TextNode objects
        """
        # Hard limit for embedding API safety (most APIs have ~8k token limit)
        # Chinese text: roughly 1 char = 1-2 tokens, so 4000 chars is safe
        HARD_LIMIT = 4000

        sentences = self._split_sentences(text)
        nodes = []
        current_chunk = ""

        for sentence in sentences:
            # Handle very long sentences (exceed chunk_size or hard limit)
            if len(sentence) > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk.strip():
                    nodes.append(TextNode(text=current_chunk.strip()))
                    current_chunk = ""

                # Split long sentence into smaller pieces
                # Use hard_limit for safety with embedding APIs
                split_size = min(self.chunk_size, HARD_LIMIT)
                for i in range(0, len(sentence), split_size - self.chunk_overlap):
                    chunk_piece = sentence[i:i + split_size]
                    if chunk_piece.strip():
                        nodes.append(TextNode(text=chunk_piece.strip()))
                continue

            # Check if adding sentence would exceed chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    nodes.append(TextNode(text=current_chunk.strip()))
                # Add overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence
            else:
                current_chunk += sentence

        # Add remaining chunk
        if current_chunk.strip():
            # Final safety check - split if exceeds hard limit
            if len(current_chunk) > HARD_LIMIT:
                for i in range(0, len(current_chunk), HARD_LIMIT - self.chunk_overlap):
                    chunk_piece = current_chunk[i:i + HARD_LIMIT]
                    if chunk_piece.strip():
                        nodes.append(TextNode(text=chunk_piece.strip()))
            else:
                nodes.append(TextNode(text=current_chunk.strip()))

        return nodes

    def split_document(self, document: Document) -> List[TextNode]:
        """Split a Chinese Document into nodes.

        Args:
            document: Document with Chinese text

        Returns:
            List of TextNode objects
        """
        nodes = self.split_text(document.text)
        # Get image_map from document
        image_map = document.metadata.get("image_map", {})
        # Copy base metadata, excluding large image_map
        base_metadata = {
            k: v for k, v in document.metadata.items()
            if k != "image_map"  # Skip the potentially large image_map
        }
        # Update metadata with image propagation
        for node in nodes:
            node.metadata.update(base_metadata)
            # Store document ID in metadata instead
            if document.doc_id:
                node.metadata["source_doc_id"] = document.doc_id
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