#!/usr/bin/env python3
"""
ProfiRAG Document Chunking Script

This script chunks documents and writes the chunks to a specified output directory.
Does NOT perform embedding or vector store operations.

Usage:
    python scripts/chunk_documents.py --input ./documents --output ./chunks
    python scripts/chunk_documents.py --input ./documents --output ./chunks --splitter sentence
    python scripts/chunk_documents.py --input ./documents --output ./chunks --splitter chinese --chunk-size 512
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter
from llama_index.core.schema import TextNode


def chunk_documents(
    input_dir: str,
    output_dir: str,
    splitter_type: str = "sentence",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    language: str = "auto",
    recursive: bool = True,
    output_format: str = "txt",
    show_progress: bool = True,
) -> dict:
    """Chunk documents and write to output directory.

    Args:
        input_dir: Path to input documents directory
        output_dir: Path to output directory for chunks
        splitter_type: Type of splitter ("sentence", "token", "semantic", "chinese")
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        language: Document language ("auto", "en", "zh")
        recursive: Search subdirectories
        output_format: Output format ("txt", "json", "jsonl")
        show_progress: Show progress information

    Returns:
        Dictionary with chunking statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load documents
    if show_progress:
        print(f"Loading documents from {input_dir}...")

    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_directory(input_dir, recursive=recursive)

    if not documents:
        print(f"No documents found in {input_dir}")
        return {"documents_loaded": 0, "chunks_created": 0}

    if show_progress:
        print(f"  - Found {len(documents)} documents")

    # Detect language if auto
    if language == "auto":
        # Simple detection: check if splitter_type is "chinese" or detect from content
        if splitter_type == "chinese":
            language = "zh"
        else:
            # Sample first document to detect
            sample_text = documents[0].text[:500]
            chinese_chars = sum(1 for c in sample_text if '\u4e00' <= c <= '\u9fff')
            language = "zh" if chinese_chars > len(sample_text) * 0.3 else "en"
            if show_progress:
                print(f"  - Detected language: {language}")

    # Create splitter
    if show_progress:
        print(f"Creating splitter: {splitter_type} (chunk_size={chunk_size}, overlap={chunk_overlap})")

    if splitter_type == "chinese" or (language == "zh" and splitter_type == "sentence"):
        splitter = ChineseTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = TextSplitter(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Split documents
    if show_progress:
        print(f"Splitting documents...")

    all_chunks: List[TextNode] = []
    for i, doc in enumerate(documents):
        if splitter_type == "chinese" or isinstance(splitter, ChineseTextSplitter):
            chunks = splitter.split_document(doc)
        else:
            chunks = splitter.split_document(doc)

        # Add source metadata
        source_file = doc.metadata.get("file_path", doc.metadata.get("file_name", f"doc_{i}"))
        for j, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = source_file
            chunk.metadata["chunk_index"] = j
            chunk.metadata["total_chunks_in_doc"] = len(chunks)

        all_chunks.extend(chunks)

        if show_progress:
            print(f"  - {source_file}: {len(chunks)} chunks")

    # Write chunks to output directory
    if show_progress:
        print(f"Writing {len(all_chunks)} chunks to {output_dir}...")

    if output_format == "txt":
        write_chunks_txt(all_chunks, output_path, documents)
    elif output_format == "json":
        write_chunks_json(all_chunks, output_path, documents)
    elif output_format == "jsonl":
        write_chunks_jsonl(all_chunks, output_path)

    # Write summary
    summary = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "documents_loaded": len(documents),
        "chunks_created": len(all_chunks),
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "language": language,
        "output_format": output_format,
        "avg_chunks_per_doc": len(all_chunks) / len(documents) if documents else 0,
    }

    summary_path = output_path / "chunking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if show_progress:
        print(f"\nChunking complete!")
        print(f"  - Documents: {len(documents)}")
        print(f"  - Total chunks: {len(all_chunks)}")
        print(f"  - Avg chunks/doc: {summary['avg_chunks_per_doc']:.1f}")
        print(f"  - Output: {output_dir}")
        print(f"  - Summary: {summary_path}")

    return summary


def write_chunks_txt(chunks: List[TextNode], output_dir: Path, documents: List) -> None:
    """Write chunks as individual .txt files.

    Organizes by source document, with each chunk as a separate file.
    """
    # Group chunks by source file
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        if source not in chunks_by_source:
            chunks_by_source[source] = []
        chunks_by_source[source].append(chunk)

    # Write each source's chunks to a subdirectory
    for source, source_chunks in chunks_by_source.items():
        # Clean source name for directory
        source_name = Path(source).stem if source != "unknown" else "unknown"
        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(source_chunks):
            chunk_file = source_dir / f"chunk_{i:04d}.txt"

            # Write header with metadata, then content
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(f"=== Chunk {i} ===\n")
                f.write(f"Source: {source}\n")
                f.write(f"Total chunks in doc: {chunk.metadata.get('total_chunks_in_doc', 'N/A')}\n")
                f.write(f"---\n\n")
                f.write(chunk.text)


def write_chunks_json(chunks: List[TextNode], output_dir: Path, documents: List) -> None:
    """Write chunks as a single JSON file with all chunks.

    Structure:
    {
        "documents": [
            {
                "source": "file.pdf",
                "chunks": [
                    {"index": 0, "text": "...", "metadata": {...}},
                    ...
                ]
            }
        ]
    }
    """
    # Group by source
    chunks_by_source = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        if source not in chunks_by_source:
            chunks_by_source[source] = {"source": source, "chunks": []}
        chunks_by_source[source]["chunks"].append({
            "index": chunk.metadata.get("chunk_index", 0),
            "text": chunk.text,
            "metadata": chunk.metadata,
        })

    output_data = {
        "documents": list(chunks_by_source.values()),
        "total_chunks": len(chunks),
    }

    output_file = output_dir / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def write_chunks_jsonl(chunks: List[TextNode], output_dir: Path) -> None:
    """Write chunks as JSONL (one JSON object per line).

    Each line is a complete chunk object with metadata.
    Useful for downstream processing pipelines.
    """
    output_file = output_dir / "chunks.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_data = {
                "id": chunk.node_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk documents and write to output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input documents directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory for chunks",
    )
    parser.add_argument(
        "--splitter", "-s",
        type=str,
        choices=["sentence", "token", "semantic", "chinese"],
        default="sentence",
        help="Splitter type (default: sentence)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap (default: 50)",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["auto", "en", "zh"],
        default="auto",
        help="Document language (default: auto-detect)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["txt", "json", "jsonl"],
        default="txt",
        help="Output format: txt (individual files), json (single file), jsonl (default: txt)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Search subdirectories (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not search subdirectories",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    show_progress = not args.quiet

    try:
        result = chunk_documents(
            input_dir=args.input,
            output_dir=args.output,
            splitter_type=args.splitter,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            language=args.language,
            recursive=args.recursive,
            output_format=args.format,
            show_progress=show_progress,
        )

        if not show_progress:
            # Output JSON for quiet mode
            print(json.dumps(result))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())