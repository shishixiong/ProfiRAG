"""Document loaders for various file types with pymupdf4llm PDF support"""

import os
import re
import tempfile
import shutil
from collections import Counter
from typing import List, Optional, Dict, Any, Union, Set
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader


def detect_header_footer_patterns(
    text: str,
    min_occurrences: int = 3,
    min_line_length: int = 5,
    max_line_length: int = 100,
) -> Set[str]:
    """Auto-detect header/footer patterns from repeating lines.

    Args:
        text: Full document text
        min_occurrences: Minimum occurrences to consider as header/footer
        min_line_length: Minimum line length to consider
        max_line_length: Maximum line length to consider (headers are usually short)

    Returns:
        Set of detected header/footer patterns
    """
    # Pattern for markdown table separator rows (e.g., |---|---|)
    # Matches lines containing only |, -, :, and whitespace
    TABLE_SEPARATOR_PATTERN = re.compile(r'^[\|\s\-:]+$')

    lines = text.split("\n")
    # Filter lines by length and clean whitespace
    # Exclude table separator rows (critical for markdown table formatting)
    candidate_lines = [
        line.strip() for line in lines
        if min_line_length <= len(line.strip()) <= max_line_length
        and not TABLE_SEPARATOR_PATTERN.match(line.strip())  # Skip table separators
    ]

    # Count occurrences
    line_counts = Counter(candidate_lines)

    # Find lines that repeat frequently (likely headers/footers)
    patterns = {
        line for line, count in line_counts.items()
        if count >= min_occurrences
    }

    return patterns


def filter_header_footer(
    text: str,
    patterns: Optional[Set[str]] = None,
    auto_detect: bool = True,
    min_occurrences: int = 3,
    custom_patterns: Optional[List[str]] = None,
) -> str:
    """Remove header/footer content from text.

    Args:
        text: Original text
        patterns: Pre-detected patterns to remove
        auto_detect: Auto-detect repeating lines as headers/footers
        min_occurrences: Minimum occurrences for auto-detection
        custom_patterns: Additional custom regex patterns to remove

    Returns:
        Filtered text without headers/footers
    """
    # Pattern for markdown table separator rows (e.g., |---|---|)
    # Matches lines containing only |, -, :, and whitespace
    TABLE_SEPARATOR_PATTERN = re.compile(r'^[\|\s\-:]+$')

    all_patterns: Set[str] = patterns or set()

    # Auto-detect patterns
    if auto_detect:
        detected = detect_header_footer_patterns(
            text, min_occurrences=min_occurrences
        )
        all_patterns.update(detected)

    # Add custom patterns
    if custom_patterns:
        all_patterns.update(custom_patterns)

    if not all_patterns:
        return text

    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        stripped = line.strip()

        # Always preserve table separator rows (essential for markdown tables)
        if TABLE_SEPARATOR_PATTERN.match(stripped):
            filtered_lines.append(line)
            continue

        # Skip if line matches any header/footer pattern
        if stripped in all_patterns:
            continue
        # Also check if line starts/ends with common header/footer markers
        skip = False
        for pattern in all_patterns:
            if stripped.startswith(pattern) or stripped.endswith(pattern):
                skip = True
                break
        if not skip:
            filtered_lines.append(line)

    # Join and clean up excessive newlines
    result = "\n".join(filtered_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)  # Max 2 consecutive newlines

    return result.strip()


# Pattern for detecting numbered headings in markdown
# Matches various formats:
# - ## **1** heading text (number wrapped in **)
# - ## **1.1 gsql** heading text (partial bold)
# - ## 1 heading text (plain)
# - ## 1.1.1 heading text (plain)
HEADING_NUMBER_PATTERN = re.compile(
    r'^(#+)\s*'  # Markdown heading prefix (##, ###, etc.)
    r'(?:\*{2})?'  # Optional opening bold marker **
    r'(\d+(?:\.\d+)*)'  # Section number: 1, 1.1, 1.1.1, etc.
    r'(?:\*{2})?\s+'  # Optional closing bold ** followed by space
)


def fix_heading_levels(text: str) -> str:
    """Fix markdown heading levels based on section number format.

    When PDFs are converted to markdown, heading levels may be incorrect
    because pymupdf4llm uses font size detection which can be unreliable
    when headings are composed of multiple spans with different sizes.

    This function corrects heading levels based on the section number:
    - "1" → level 1 (h1)
    - "1.1" → level 2 (h2)
    - "1.1.1" → level 3 (h3)
    - "1.1.1.1" → level 4 (h4)
    - etc.

    Args:
        text: Markdown text with potentially incorrect heading levels

    Returns:
        Markdown text with corrected heading levels
    """
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        match = HEADING_NUMBER_PATTERN.match(line)
        if match:
            existing_prefix = match.group(1)  # e.g., ##
            number = match.group(2)  # e.g., 1.1.1

            # Calculate level from number format
            # 1 → level 1, 1.1 → level 2, 1.1.1 → level 3
            level = number.count(".") + 1

            # Cap at maximum markdown heading level (6)
            level = min(level, 6)

            # Get the rest of the line after the matched portion
            rest = line[match.end():]

            # Clean up remaining bold markers in heading text
            # Remove trailing ** that might be left from partial bold formatting
            rest = re.sub(r'\*{2}(?:\s|$)', '', rest)  # Remove ** followed by space or end
            rest = re.sub(r'^\*{2}\s*', '', rest)  # Remove leading ** at start of rest

            # Build new line with correct heading level
            new_line = "#" * level + " " + number + " " + rest.strip()
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


class PDFLoader:
    """PDF document loader using pymupdf4llm for Markdown conversion.

    Converts PDF to Markdown preserving structure:
    - Headers, lists, tables
    - Document hierarchy
    - Images (optional)

    Optionally filters out repeating header/footer content and fixes
    heading levels based on section number format.

    Args:
        use_pymupdf4llm: Use pymupdf4llm for PDF processing (default True)
        write_images: Extract and save images from PDF
        image_path: Directory to save extracted images (default: temp dir)
        pages: Specific pages to extract (None = all pages)
        as_llama_index_docs: Directly return LlamaIndex Documents
        exclude_header_footer: Filter out repeating header/footer content
        header_footer_patterns: Custom regex patterns for header/footer removal
        header_footer_auto_detect: Auto-detect header/footer patterns
        header_footer_min_occurrences: Min occurrences for auto-detection
        fix_heading_levels: Fix heading levels based on section numbers (e.g., 1.1.1)
    """

    def __init__(
        self,
        use_pymupdf4llm: bool = True,
        write_images: bool = False,
        image_path: Optional[str] = None,
        pages: Optional[List[int]] = None,
        as_llama_index_docs: bool = True,
        exclude_header_footer: bool = False,
        header_footer_patterns: Optional[List[str]] = None,
        header_footer_auto_detect: bool = True,
        header_footer_min_occurrences: int = 3,
        fix_heading_levels: bool = True,
        **kwargs
    ):
        """Initialize PDF loader.

        Args:
            use_pymupdf4llm: Use pymupdf4llm for PDF processing
            write_images: Extract and save images
            image_path: Path to save images (creates temp dir if None)
            pages: Page numbers to extract (None = all)
            as_llama_index_docs: Return LlamaIndex Documents directly
            exclude_header_footer: Filter header/footer content
            header_footer_patterns: Custom patterns for removal
            header_footer_auto_detect: Auto-detect patterns
            header_footer_min_occurrences: Min occurrences for detection
            fix_heading_levels: Fix heading levels based on section numbers
            **kwargs: Additional arguments
        """
        self.use_pymupdf4llm = use_pymupdf4llm
        self.write_images = write_images
        self.image_path = image_path
        self.pages = pages
        self.as_llama_index_docs = as_llama_index_docs
        self.exclude_header_footer = exclude_header_footer
        self.header_footer_patterns = header_footer_patterns
        self.header_footer_auto_detect = header_footer_auto_detect
        self.header_footer_min_occurrences = header_footer_min_occurrences
        self.fix_heading_levels = fix_heading_levels
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

                # Add metadata and apply header/footer filtering
                if self.exclude_header_footer:
                    for doc in docs:
                        original_text = doc.text
                        filtered_text = filter_header_footer(
                            original_text,
                            auto_detect=self.header_footer_auto_detect,
                            min_occurrences=self.header_footer_min_occurrences,
                            custom_patterns=self.header_footer_patterns,
                        )
                        doc.text = filtered_text

                # Fix heading levels based on section numbers
                if self.fix_heading_levels:
                    for doc in docs:
                        doc.text = fix_heading_levels(doc.text)

                for doc in docs:
                    doc.metadata["source_file"] = str(path.name)
                    doc.metadata["source_path"] = str(path)
                    doc.metadata["loader"] = "pymupdf4llm"
                    doc.metadata["header_footer_filtered"] = self.exclude_header_footer
                    doc.metadata["heading_levels_fixed"] = self.fix_heading_levels
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

                # Apply header/footer filtering if enabled
                if self.exclude_header_footer:
                    md_text = filter_header_footer(
                        md_text,
                        auto_detect=self.header_footer_auto_detect,
                        min_occurrences=self.header_footer_min_occurrences,
                        custom_patterns=self.header_footer_patterns,
                    )

                # Fix heading levels based on section numbers
                if self.fix_heading_levels:
                    md_text = fix_heading_levels(md_text)

                # Create single Document
                doc = Document(
                    text=md_text,
                    metadata={
                        "source_file": str(path.name),
                        "source_path": str(path),
                        "loader": "pymupdf4llm",
                        "header_footer_filtered": self.exclude_header_footer,
                        "heading_levels_fixed": self.fix_heading_levels,
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
        exclude_header_footer: Filter out repeating header/footer content
        header_footer_patterns: Custom patterns for header/footer removal
        header_footer_auto_detect: Auto-detect header/footer patterns
        header_footer_min_occurrences: Min occurrences for auto-detection
        fix_heading_levels: Fix heading levels based on section numbers (e.g., 1.1.1)
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
        exclude_header_footer: bool = False,
        header_footer_patterns: Optional[List[str]] = None,
        header_footer_auto_detect: bool = True,
        header_footer_min_occurrences: int = 3,
        fix_heading_levels: bool = True,
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
            exclude_header_footer: Filter header/footer content
            header_footer_patterns: Custom patterns for removal
            header_footer_auto_detect: Auto-detect patterns
            header_footer_min_occurrences: Min occurrences for detection
            fix_heading_levels: Fix heading levels based on section numbers
            **kwargs: Additional arguments
        """
        self.encoding = encoding
        self.extract_metadata = extract_metadata
        self.use_pymupdf4llm = use_pymupdf4llm
        self.pdf_write_images = pdf_write_images
        self.pdf_image_path = pdf_image_path
        self.pdf_pages = pdf_pages
        self.exclude_header_footer = exclude_header_footer
        self.header_footer_patterns = header_footer_patterns
        self.header_footer_auto_detect = header_footer_auto_detect
        self.header_footer_min_occurrences = header_footer_min_occurrences
        self.fix_heading_levels = fix_heading_levels
        self.kwargs = kwargs

        # Initialize PDF loader
        self._pdf_loader = PDFLoader(
            use_pymupdf4llm=use_pymupdf4llm,
            write_images=pdf_write_images,
            image_path=pdf_image_path,
            pages=pdf_pages,
            exclude_header_footer=exclude_header_footer,
            header_footer_patterns=header_footer_patterns,
            header_footer_auto_detect=header_footer_auto_detect,
            header_footer_min_occurrences=header_footer_min_occurrences,
            fix_heading_levels=fix_heading_levels,
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
        exclude_header_footer: Optional[bool] = None,
        fix_headings: Optional[bool] = None,
        **kwargs
    ) -> str:
        """Convert PDF to Markdown file and save it.

        Args:
            pdf_path: Path to PDF file
            output_md_path: Path to save Markdown file
            exclude_header_footer: Filter header/footer (uses instance default if None)
            fix_headings: Fix heading levels (uses instance default if None)
            **kwargs: Additional arguments for pymupdf4llm

        Returns:
            Path to the saved Markdown file
        """
        import pymupdf4llm

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Use instance defaults if not specified
        do_filter = exclude_header_footer if exclude_header_footer is not None else self.exclude_header_footer
        do_fix_headings = fix_headings if fix_headings is not None else self.fix_heading_levels

        # Convert to Markdown
        md_text = pymupdf4llm.to_markdown(
            doc=pdf_path,
            pages=self.pdf_pages,
            write_images=self.pdf_write_images,
            image_path=self.pdf_image_path,
        )

        # Apply header/footer filtering if enabled
        if do_filter:
            md_text = filter_header_footer(
                md_text,
                auto_detect=self.header_footer_auto_detect,
                min_occurrences=self.header_footer_min_occurrences,
                custom_patterns=self.header_footer_patterns,
            )

        # Fix heading levels based on section numbers
        if do_fix_headings:
            md_text = fix_heading_levels(md_text)

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
    exclude_header_footer: bool = False,
    header_footer_patterns: Optional[List[str]] = None,
    header_footer_auto_detect: bool = True,
    header_footer_min_occurrences: int = 3,
    fix_heading_levels: bool = True,
) -> Union[str, List[Document]]:
    """Convenience function to convert PDF to Markdown.

    Args:
        pdf_path: Path to PDF file
        output_path: Path to save Markdown file (optional)
        write_images: Extract images
        image_path: Directory for images
        pages: Specific pages to extract
        exclude_header_footer: Filter header/footer content
        header_footer_patterns: Custom patterns for removal
        header_footer_auto_detect: Auto-detect patterns
        header_footer_min_occurrences: Min occurrences for detection
        fix_heading_levels: Fix heading levels based on section numbers

    Returns:
        Markdown text or path to saved file
    """
    loader = PDFLoader(
        write_images=write_images,
        image_path=image_path,
        pages=pages,
        as_llama_index_docs=False,
        exclude_header_footer=exclude_header_footer,
        header_footer_patterns=header_footer_patterns,
        header_footer_auto_detect=header_footer_auto_detect,
        header_footer_min_occurrences=header_footer_min_occurrences,
        fix_heading_levels=fix_heading_levels,
    )

    docs = loader.load_pdf(pdf_path)
    md_text = docs[0].text if docs else ""

    if output_path:
        Path(output_path).write_text(md_text, encoding="utf-8")
        return output_path

    return md_text