"""Document loaders for various file types"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader


class DocumentLoader:
    """Unified document loader supporting multiple file types.

    Supports: PDF, TXT, MD, DOCX, HTML, JSON, CSV
    """

    SUPPORTED_EXTENSIONS = [
        ".pdf", ".txt", ".md", ".docx", ".html", ".htm",
        ".json", ".csv", ".xlsx", ".pptx"
    ]

    def __init__(
        self,
        encoding: str = "utf-8",
        extract_metadata: bool = True,
        **kwargs
    ):
        """Initialize document loader.

        Args:
            encoding: File encoding for text files
            extract_metadata: Extract metadata from files
            **kwargs: Additional arguments
        """
        self.encoding = encoding
        self.extract_metadata = extract_metadata
        self.kwargs = kwargs

    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """Load all documents from a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            exclude: File patterns to exclude
            **kwargs: Additional arguments

        Returns:
            List of Document objects
        """
        reader = SimpleDirectoryReader(
            input_dir=directory,
            recursive=recursive,
            exclude=exclude or [],
            encoding=self.encoding,
            **kwargs
        )
        return reader.load_data()

    def load_file(
        self,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """Load a single file.

        Args:
            file_path: File path
            **kwargs: Additional arguments

        Returns:
            List of Document objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = SimpleDirectoryReader(
            input_files=[file_path],
            encoding=self.encoding,
            **kwargs
        )
        return reader.load_data()

    def load_files(
        self,
        file_paths: List[str],
        **kwargs
    ) -> List[Document]:
        """Load multiple files.

        Args:
            file_paths: List of file paths
            **kwargs: Additional arguments

        Returns:
            List of Document objects
        """
        valid_files = []
        for fp in file_paths:
            if Path(fp).exists():
                valid_files.append(fp)

        if not valid_files:
            return []

        reader = SimpleDirectoryReader(
            input_files=valid_files,
            encoding=self.encoding,
            **kwargs
        )
        return reader.load_data()

    def load_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Create a Document from raw text.

        Args:
            text: Text content
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document object
        """
        return Document(
            text=text,
            metadata=metadata or {},
            doc_id=doc_id,
        )

    def load_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Create Documents from multiple texts.

        Args:
            texts: List of text contents
            metadatas: Optional list of metadata dicts

        Returns:
            List of Document objects
        """
        metadatas = metadatas or [{} for _ in texts]
        return [
            Document(text=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if file type is supported.

        Args:
            file_path: File path

        Returns:
            True if supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in DocumentLoader.SUPPORTED_EXTENSIONS