"""Complete ingestion pipeline for processing documents"""

from typing import List, Optional, Dict, Any
from llama_index.core import Document
from llama_index.core.schema import TextNode

from .loaders import DocumentLoader
from .splitters import TextSplitter, ChineseTextSplitter


class IngestionPipeline:
    """Complete ingestion pipeline for document processing.

    Combines loading, splitting, and optional preprocessing.
    """

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        splitter: Optional[TextSplitter] = None,
        splitter_type: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embed_model: Optional[Any] = None,
        language: str = "en",
        **kwargs
    ):
        """Initialize ingestion pipeline.

        Args:
            loader: Document loader instance (creates default if None)
            splitter: Text splitter instance (creates based on params if None)
            splitter_type: Splitter type ("sentence", "token", "semantic", "chinese")
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            embed_model: Embedding model for semantic splitter
            language: Document language ("en" or "zh")
            **kwargs: Additional arguments
        """
        self.loader = loader or DocumentLoader()

        if splitter:
            self.splitter = splitter
        elif language == "zh" and splitter_type == "sentence":
            self.splitter = ChineseTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.splitter = TextSplitter(
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_model=embed_model,
            )

        self.language = language
        self.kwargs = kwargs

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True,
        **kwargs
    ) -> List[TextNode]:
        """Ingest all documents from a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        documents = self.loader.load_directory(directory, recursive=recursive)
        return self.splitter.split_documents(documents)

    def ingest_file(
        self,
        file_path: str,
        **kwargs
    ) -> List[TextNode]:
        """Ingest a single file.

        Args:
            file_path: File path
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        documents = self.loader.load_file(file_path)
        return self.splitter.split_documents(documents)

    def ingest_files(
        self,
        file_paths: List[str],
        **kwargs
    ) -> List[TextNode]:
        """Ingest multiple files.

        Args:
            file_paths: List of file paths
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        documents = self.loader.load_files(file_paths)
        return self.splitter.split_documents(documents)

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[TextNode]:
        """Ingest raw text.

        Args:
            text: Text content
            metadata: Optional metadata
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        document = self.loader.load_text(text, metadata=metadata)
        return self.splitter.split_document(document)

    def ingest_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[TextNode]:
        """Ingest multiple text strings.

        Args:
            texts: List of text contents
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        documents = self.loader.load_texts(texts, metadatas=metadatas)
        return self.splitter.split_documents(documents)

    def ingest_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[TextNode]:
        """Ingest pre-loaded documents.

        Args:
            documents: List of Document objects
            **kwargs: Additional arguments

        Returns:
            List of TextNode objects
        """
        return self.splitter.split_documents(documents)

    def update_splitter_params(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """Update splitter parameters.

        Args:
            chunk_size: New chunk size
            chunk_overlap: New overlap
        """
        if chunk_size:
            self.splitter.update_chunk_size(chunk_size)
        if chunk_overlap:
            self.splitter.update_overlap(chunk_overlap)