"""Document loaders for various file types with pymupdf4llm PDF support"""

import os
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader


class PDFLoader:
    """PDF document loader using pymupdf4llm for Markdown conversion.

    Converts PDF to Markdown preserving structure:
    - Headers, lists, tables
    - Document hierarchy
    - Images (optional)

    Args:
        use_pymupdf4llm: Use pymupdf4llm for PDF processing (default True)
        write_images: Extract and save images from PDF
        image_path: Directory to save extracted images (default: temp dir)
        pages: Specific pages to extract (None = all pages)
        as_llama_index_docs: Directly return LlamaIndex Documents
    """

    def __init__(
        self,
        use_pymupdf4llm: bool = True,
        write_images: bool = False,
        image_path: Optional[str] = None,
        pages: Optional[List[int]] = None,
        as_llama_index_docs: bool = True,
        **kwargs
    ):
        """Initialize PDF loader.

        Args:
            use_pymupdf4llm: Use pymupdf4llm for PDF processing
            write_images: Extract and save images
            image_path: Path to save images (creates temp dir if None)
            pages: Page numbers to extract (None = all)
            as_llama_index_docs: Return LlamaIndex Documents directly
            **kwargs: Additional arguments
        """
        self.use_pymupdf4llm = use_pymupdf4llm
        self.write_images = write_images
        self.image_path = image_path
        self.pages = pages
        self.as_llama_index_docs = as_llama_index_docs
        self.kwargs = kwargs

        # Check if pymupdf4llm is available
        self._pymupdf4llm_available = self._check_pymupdf4llm()

    def _check_pymupdf4llm(self) -> bool:
        """Check if pymupdf4llm is installed."""
        try:
            import pymupdf4llm
            return True
        except ImportError:
            return False

    def load_pdf(
        self,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """Load a PDF file and convert to Markdown.

        Args:
            file_path: Path to PDF file
            **kwargs: Additional arguments

        Returns:
            List of Document objects with Markdown content
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not self._pymupdf4llm_available:
            # Fallback to LlamaIndex default reader
            return self._load_pdf_fallback(file_path)

        import pymupdf4llm

        # Setup image path if needed
        image_dir = None
        if self.write_images:
            if self.image_path:
                image_dir = Path(self.image_path)
                image_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Create temp directory for images
                image_dir = Path(tempfile.mkdtemp(prefix="pdf_images_"))

        try:
            # Convert PDF to Markdown/Documents
            if self.as_llama_index_docs:
                # Direct LlamaIndex Document output
                docs = pymupdf4llm.to_markdown(
                    doc=file_path,
                    pages=self.pages,
                    write_images=self.write_images,
                    image_path=str(image_dir) if image_dir else None,
                    as_llama_index_docs=True,
                )

                # Add metadata
                for doc in docs:
                    doc.metadata["source_file"] = str(path.name)
                    doc.metadata["source_path"] = str(path)
                    doc.metadata["loader"] = "pymupdf4llm"
                    if image_dir:
                        doc.metadata["image_path"] = str(image_dir)

                return docs
            else:
                # Get Markdown text
                md_text = pymupdf4llm.to_markdown(
                    doc=file_path,
                    pages=self.pages,
                    write_images=self.write_images,
                    image_path=str(image_dir) if image_dir else None,
                )

                # Create single Document
                doc = Document(
                    text=md_text,
                    metadata={
                        "source_file": str(path.name),
                        "source_path": str(path),
                        "loader": "pymupdf4llm",
                        "image_path": str(image_dir) if image_dir else None,
                    },
                )
                return [doc]

        except Exception as e:
            print(f"Warning: pymupdf4llm failed for {file_path}: {e}")
            print("Falling back to default PDF reader...")
            return self._load_pdf_fallback(file_path)

    def _load_pdf_fallback(
        self,
        file_path: str,
    ) -> List[Document]:
        """Fallback to LlamaIndex default PDF reader.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects
        """
        reader = SimpleDirectoryReader(
            input_files=[file_path],
        )
        return reader.load_data()

    def load_pdf_directory(
        self,
        directory: str,
        recursive: bool = True,
        **kwargs
    ) -> List[Document]:
        """Load all PDFs from a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            **kwargs: Additional arguments

        Returns:
            List of Document objects
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all PDF files
        pattern = "*.pdf" if not recursive else "**/*.pdf"
        pdf_files = list(dir_path.glob(pattern))

        if not pdf_files:
            return []

        documents = []
        for pdf_file in pdf_files:
            docs = self.load_pdf(str(pdf_file), **kwargs)
            documents.extend(docs)

        return documents


class DocumentLoader:
    """Unified document loader supporting multiple file types.

    Supports: PDF (via pymupdf4llm), TXT, MD, DOCX, HTML, JSON, CSV

    Args:
        encoding: File encoding for text files
        extract_metadata: Extract metadata from files
        use_pymupdf4llm: Use pymupdf4llm for PDF processing
        pdf_write_images: Extract images from PDFs
        pdf_image_path: Directory to save extracted PDF images
        pdf_pages: Specific PDF pages to extract
    """

    SUPPORTED_EXTENSIONS = [
        ".pdf", ".txt", ".md", ".docx", ".html", ".htm",
        ".json", ".csv", ".xlsx", ".pptx"
    ]

    def __init__(
        self,
        encoding: str = "utf-8",
        extract_metadata: bool = True,
        use_pymupdf4llm: bool = True,
        pdf_write_images: bool = False,
        pdf_image_path: Optional[str] = None,
        pdf_pages: Optional[List[int]] = None,
        **kwargs
    ):
        """Initialize document loader.

        Args:
            encoding: File encoding for text files
            extract_metadata: Extract metadata from files
            use_pymupdf4llm: Use pymupdf4llm for PDF processing
            pdf_write_images: Extract and save images from PDFs
            pdf_image_path: Directory for PDF images
            pdf_pages: Specific PDF pages to extract
            **kwargs: Additional arguments
        """
        self.encoding = encoding
        self.extract_metadata = extract_metadata
        self.use_pymupdf4llm = use_pymupdf4llm
        self.pdf_write_images = pdf_write_images
        self.pdf_image_path = pdf_image_path
        self.pdf_pages = pdf_pages
        self.kwargs = kwargs

        # Initialize PDF loader
        self._pdf_loader = PDFLoader(
            use_pymupdf4llm=use_pymupdf4llm,
            write_images=pdf_write_images,
            image_path=pdf_image_path,
            pages=pdf_pages,
        )

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
        dir_path = Path(directory)

        # Separate PDF files and other files
        pattern = "*" if not recursive else "**/*"
        all_files = list(dir_path.glob(pattern))

        pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
        other_files = [
            f for f in all_files
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and f.suffix.lower() != ".pdf"
            and f.is_file()
        ]

        # Apply exclude patterns
        if exclude:
            import fnmatch
            other_files = [
                f for f in other_files
                if not any(fnmatch.fnmatch(str(f), ex) for ex in exclude)
            ]
            pdf_files = [
                f for f in pdf_files
                if not any(fnmatch.fnmatch(str(f), ex) for ex in exclude)
            ]

        documents = []

        # Load PDFs with pymupdf4llm
        if pdf_files:
            for pdf_file in pdf_files:
                docs = self._pdf_loader.load_pdf(str(pdf_file))
                documents.extend(docs)

        # Load other files with SimpleDirectoryReader
        if other_files:
            reader = SimpleDirectoryReader(
                input_files=[str(f) for f in other_files],
                encoding=self.encoding,
            )
            other_docs = reader.load_data()
            documents.extend(other_docs)

        return documents

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

        # Use PDF loader for PDF files
        if path.suffix.lower() == ".pdf":
            return self._pdf_loader.load_pdf(file_path)

        # Use SimpleDirectoryReader for other files
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

        # Separate PDF and non-PDF files
        pdf_files = [f for f in valid_files if Path(f).suffix.lower() == ".pdf"]
        other_files = [
            f for f in valid_files
            if Path(f).suffix.lower() != ".pdf"
        ]

        documents = []

        # Load PDFs
        if pdf_files:
            for pdf_file in pdf_files:
                docs = self._pdf_loader.load_pdf(pdf_file)
                documents.extend(docs)

        # Load other files
        if other_files:
            reader = SimpleDirectoryReader(
                input_files=other_files,
                encoding=self.encoding,
                **kwargs
            )
            other_docs = reader.load_data()
            documents.extend(other_docs)

        return documents

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

    def pdf_to_markdown_file(
        self,
        pdf_path: str,
        output_md_path: str,
        **kwargs
    ) -> str:
        """Convert PDF to Markdown file and save it.

        Args:
            pdf_path: Path to PDF file
            output_md_path: Path to save Markdown file
            **kwargs: Additional arguments for pymupdf4llm

        Returns:
            Path to the saved Markdown file
        """
        import pymupdf4llm

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Convert to Markdown
        md_text = pymupdf4llm.to_markdown(
            doc=pdf_path,
            pages=self.pdf_pages,
            write_images=self.pdf_write_images,
            image_path=self.pdf_image_path,
        )

        # Save to file
        output_path = Path(output_md_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_text, encoding="utf-8")

        return str(output_path)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: Optional[str] = None,
    write_images: bool = False,
    image_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
) -> Union[str, List[Document]]:
    """Convenience function to convert PDF to Markdown.

    Args:
        pdf_path: Path to PDF file
        output_path: Path to save Markdown file (optional)
        write_images: Extract images
        image_path: Directory for images
        pages: Specific pages to extract

    Returns:
        Markdown text or path to saved file
    """
    loader = PDFLoader(
        write_images=write_images,
        image_path=image_path,
        pages=pages,
        as_llama_index_docs=False,
    )

    docs = loader.load_pdf(pdf_path)
    md_text = docs[0].text if docs else ""

    if output_path:
        Path(output_path).write_text(md_text, encoding="utf-8")
        return output_path

    return md_text