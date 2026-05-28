#!/usr/bin/env python3
"""ProfiRAG Document Ingestion CLI.

Usage:
    profirag-ingest --documents ./documents
    profirag-ingest --documents ./documents --splitter chinese
    profirag-ingest --file ./documents/example.pdf
    profirag-ingest --documents ./code --splitter ast --ast-language python
    profirag-ingest --file ./documents/example.pdf --mode vector
    profirag-ingest --file ./documents/example.pdf --mode hybrid

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
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from profirag.config.settings import load_config, RAGConfig
from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter
from profirag.pipeline.rag_pipeline import RAGPipeline


def _resolve_env_path(env_file: str) -> Path:
    p = Path(env_file)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def apply_index_mode(config: RAGConfig, mode: str) -> RAGConfig:
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
    env_path = _resolve_env_path(env_file)
    config = load_config(str(env_path))
    config = apply_index_mode(config, mode)

    if show_progress:
        print(f"Loading configuration from {env_path}...")

    if splitter_type:
        config.chunking.splitter_type = splitter_type
    if chunk_size:
        config.chunking.chunk_size = chunk_size
    if chunk_overlap:
        config.chunking.chunk_overlap = chunk_overlap
    if ast_language:
        config.chunking.ast_language = ast_language

    if show_progress:
        print(f"Initializing RAG pipeline...")
        print(f"  - Embedding model: {config.embedding.model}")
        print(f"  - LLM model: {config.llm.model}")
        print(f"  - Storage type: {config.storage.type}")
        print(f"  - Splitter: {config.chunking.splitter_type}")
        print(f"  - Chunk size: {config.chunking.chunk_size}")
        print(f"  - Chunk overlap: {config.chunking.chunk_overlap}")

    pipeline = RAGPipeline(config)

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

    if show_progress:
        print(f"Ingesting documents into vector store...")

    start_time = time.time()
    doc_ids = pipeline.ingest_documents(documents)
    elapsed = time.time() - start_time

    if show_progress:
        print(f"  - Ingested {len(doc_ids)} documents in {elapsed:.2f} seconds")

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
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    ast_language: str = None,
    mode: str = "hybrid",
) -> dict:
    env_path = _resolve_env_path(env_file)
    config = load_config(str(env_path))
    config = apply_index_mode(config, mode)

    if show_progress:
        print(f"Loading configuration from {env_path}...")

    if splitter_type:
        config.chunking.splitter_type = splitter_type
    if chunk_size:
        config.chunking.chunk_size = chunk_size
    if chunk_overlap:
        config.chunking.chunk_overlap = chunk_overlap
    if ast_language:
        config.chunking.ast_language = ast_language

    if show_progress:
        print(f"Initializing RAG pipeline...")
        print(f"  - Embedding model: {config.embedding.model}")
        print(f"  - LLM model: {config.llm.model}")
        print(f"  - Storage type: {config.storage.type}")
        print(f"  - Splitter: {config.chunking.splitter_type}")
        print(f"  - Chunk size: {config.chunking.chunk_size}")
        print(f"  - Chunk overlap: {config.chunking.chunk_overlap}")

    pipeline = RAGPipeline(config)

    if show_progress:
        print(f"Loading file: {file_path}...")

    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_file(file_path)

    if not documents:
        print(f"Could not load file: {file_path}")
        return {"documents_loaded": 0, "documents_ingested": 0}

    if show_progress:
        print(f"Ingesting document...")
        start_time = time.time()

    doc_ids = pipeline.ingest_documents(documents)

    elapsed = time.time() - start_time
    if show_progress:
        print(f"  - Ingested in {elapsed:.2f} seconds")

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
    parser = argparse.ArgumentParser(
        description="Ingest documents into ProfiRAG vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--documents", "-d",
        type=str,
        default="./documents",
        help="Path to documents directory (default: ./documents)",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a single file to ingest",
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env)",
    )
    parser.add_argument(
        "--recursive", "-r",
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
        "--splitter", "-s",
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
        "--quiet", "-q",
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
        "--mode", "-m",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Index mode: vector (semantic only), hybrid (both BM25 and vector, default)",
    )

    args = parser.parse_args()
    show_progress = not args.quiet

    try:
        if args.file:
            result = ingest_file(
                file_path=args.file,
                env_file=args.env,
                show_progress=show_progress,
                splitter_type=args.splitter,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                ast_language=args.ast_language,
                mode=args.mode,
            )
        else:
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
            print(json.dumps(result))

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during ingestion: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
