#!/usr/bin/env python3
"""
ProfiRAG Document Ingestion Script

This script ingests documents from a directory into the vector store
using the configuration from .env file.

Usage:
    # Basic usage (uses splitter config from .env)
    python scripts/ingest_documents.py --documents ./documents

    # Use Chinese splitter for Chinese documents
    python scripts/ingest_documents.py --documents ./documents --splitter chinese

    # Override chunk size and overlap
    python scripts/ingest_documents.py --documents ./documents --splitter sentence --chunk-size 1024 --chunk-overlap 100

    # Ingest a single file
    python scripts/ingest_documents.py --file ./documents/example.pdf

    # Use AST splitter for code files
    python scripts/ingest_documents.py --documents ./code --splitter ast --ast-language python

    # Vector-only mode (no BM25)
    python scripts/ingest_documents.py --file ./documents/example.pdf --mode vector

    # Hybrid mode (default)
    python scripts/ingest_documents.py --file ./documents/example.pdf --mode hybrid

Index modes:
    - vector: Dense vector index only, best for semantic search
    - hybrid: Both BM25 and vector indexes (default), best for mixed queries

Splitter types:
    - sentence: Split by sentences (default)
    - token: Split by token count
    - semantic: Split by semantic similarity (requires embedding)
    - chinese: Optimized for Chinese text
    - ast: AST-based splitter for code files (Python, Java, C++, Go)
    - markdown: Structured splitter for Markdown (preserves headers, code blocks, tables)
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.config.settings import load_config, RAGConfig
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def apply_index_mode(config: RAGConfig, mode: str) -> RAGConfig:
    """Apply index mode settings to configuration.

    Args:
        config: RAGConfig instance
        mode: Index mode - "vector" or "hybrid"

    Returns:
        Modified RAGConfig with index mode applied
    """
    config.storage.config["index_mode"] = mode
    return config


def ingest_directory(
    documents_dir: str,
    env_file: str = ".env",
    recursive: bool = True,
    show_progress: bool = True,
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    ast_language: str = None,
    mode: str = "hybrid",
) -> dict:
    """Ingest documents from a directory into the RAG pipeline.

    Args:
        documents_dir: Path to documents directory
        env_file: Path to .env configuration file
        recursive: Whether to search subdirectories
        show_progress: Show progress information
        splitter_type: Override splitter type (sentence, token, semantic, chinese, ast)
        chunk_size: Override chunk size
        chunk_overlap: Override chunk overlap
        ast_language: Language for AST splitter (python, java, cpp, go)
        mode: Index mode - "vector" or "hybrid"

    Returns:
        Dictionary with ingestion statistics
    """
    # Load configuration
    config = load_config(env_file)

    # Apply index mode settings
    config = apply_index_mode(config, mode)

    if show_progress:
        print(f"Loading configuration from {env_file}...")

    # Override chunking settings if provided
    if splitter_type:
        config.chunking.splitter_type = splitter_type
    if chunk_size:
        config.chunking.chunk_size = chunk_size
    if chunk_overlap:
        config.chunking.chunk_overlap = chunk_overlap
    if ast_language:
        config.chunking.ast_language = ast_language

    # Initialize pipeline
    if show_progress:
        print(f"Initializing RAG pipeline...")
        print(f"  - Embedding model: {config.embedding.model}")
        print(f"  - LLM model: {config.llm.model}")
        print(f"  - Storage type: {config.storage.type}")
        print(f"  - Splitter: {config.chunking.splitter_type}")
        print(f"  - Chunk size: {config.chunking.chunk_size}")
        print(f"  - Chunk overlap: {config.chunking.chunk_overlap}")

    pipeline = RAGPipeline(config)

    # Load documents
    if show_progress:
        print(f"Loading documents from {documents_dir}...")

    loader = DocumentLoader(fix_heading_levels=True, encoding="utf-8")
    documents = loader.load_directory(
        documents_dir,
        recursive=recursive,
    )

    if not documents:
        print(f"No documents found in {documents_dir}")
        return {"documents_loaded": 0, "documents_ingested": 0}

    if show_progress:
        print(f"  - Found {len(documents)} documents")

    # Ingest documents
    if show_progress:
        print(f"Ingesting documents into vector store...")

    start_time = time.time()
    doc_ids = pipeline.ingest_documents(documents)
    elapsed = time.time() - start_time

    if show_progress:
        print(f"  - Ingested {len(doc_ids)} documents in {elapsed:.2f} seconds")

    # Get final stats
    stats = pipeline.get_stats()

    if show_progress:
        print(f"\nIngestion complete!")
        print(f"  - Vector store count: {stats['vector_store']['count']}")

    return {
        "documents_loaded": len(documents),
        "documents_ingested": len(doc_ids),
        "elapsed_seconds": elapsed,
        "vector_store_count": stats["vector_store"]["count"],
    }


def ingest_file(
    file_path: str,
    env_file: str = ".env",
    show_progress: bool = True,
    mode: str = "hybrid",
) -> dict:
    """Ingest a single file into the RAG pipeline.

    Args:
        file_path: Path to the file
        env_file: Path to .env configuration file
        show_progress: Show progress information
        mode: Index mode - "vector" or "hybrid"

    Returns:
        Dictionary with ingestion statistics
    """
    # Load configuration
    config = load_config(env_file)

    # Apply index mode settings
    config = apply_index_mode(config, mode)

    if show_progress:
        print(f"Loading configuration from {env_file}...")

    # Initialize pipeline
    if show_progress:
        print(f"Initializing RAG pipeline...")

    pipeline = RAGPipeline(config)

    # Load document
    if show_progress:
        print(f"Loading file: {file_path}...")

    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_file(file_path)

    if not documents:
        print(f"Could not load file: {file_path}")
        return {"documents_loaded": 0, "documents_ingested": 0}

    # Ingest document
    if show_progress:
        print(f"Ingesting document...")
        start_time = time.time()

    doc_ids = pipeline.ingest_documents(documents)

    elapsed = time.time() - start_time
    if show_progress:
        print(f"  - Ingested in {elapsed:.2f} seconds")

    # Get final stats
    stats = pipeline.get_stats()

    if show_progress:
        print(f"\nIngestion complete!")
        print(f"  - Vector store count: {stats['vector_store']['count']}")

    return {
        "documents_loaded": len(documents),
        "documents_ingested": len(doc_ids),
        "elapsed_seconds": elapsed,
        "vector_store_count": stats["vector_store"]["count"],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into ProfiRAG vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--documents",
        "-d",
        type=str,
        default="./documents",
        help="Path to documents directory (default: ./documents)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to a single file to ingest",
    )
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="Search subdirectories (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not search subdirectories",
    )
    parser.add_argument(
        "--splitter",
        "-s",
        type=str,
        choices=["sentence", "token", "semantic", "chinese", "ast", "markdown"],
        default=None,
        help="Splitter type (default: from .env or 'sentence')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size (default: from .env or 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap (default: from .env or 50)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--ast-language",
        type=str,
        choices=["python", "java", "cpp", "go"],
        default=None,
        help="Language for AST splitter (default: from .env or 'python')",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Index mode: vector (semantic only), hybrid (both BM25 and vector, default)",
    )

    args = parser.parse_args()

    show_progress = not args.quiet

    try:
        if args.file:
            # Ingest single file
            result = ingest_file(
                file_path=args.file,
                env_file=args.env,
                show_progress=show_progress,
                mode=args.mode,
            )
        else:
            # Ingest directory
            result = ingest_directory(
                documents_dir=args.documents,
                env_file=args.env,
                recursive=args.recursive,
                splitter_type=args.splitter,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                ast_language=args.ast_language,
                show_progress=show_progress,
                mode=args.mode,
            )

        if not show_progress:
            # Output JSON for quiet mode
            import json
            print(json.dumps(result))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())