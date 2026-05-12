#!/usr/bin/env python3
"""
ProfiRAG Wiki Ingestion Script

This script ingests wiki content from URLs into the vector store
using the configuration from .env file.

Usage:
    # Basic usage
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123"

    # Process multiple wikis from a file
    python scripts/ingest_wiki.py --wiki-list wiki_urls.txt

    # Use markdown splitter for better structure preservation
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123" --splitter markdown

    # Override chunk settings
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123" --chunk-size 1024 --chunk-overlap 100

    # Vector-only mode (no BM25)
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123" --mode vector

    # Add custom metadata
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123" --domain finance --group accounting
    python scripts/ingest_wiki.py --wiki-url "https://wiki.example.com/pages/123" --metadata "key1=value1" --metadata "key2=value2"

Splitter types:
    - sentence: Split by sentences (default)
    - token: Split by token count
    - semantic: Split by semantic similarity (requires embedding)
    - chinese: Optimized for Chinese text
    - markdown: Structured splitter for Markdown (preserves headers, code blocks, tables)

Index modes:
    - vector: Dense vector index only, best for semantic search
    - hybrid: Both BM25 and vector indexes (default), best for mixed queries
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.config.settings import load_config, RAGConfig
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter, MarkdownSplitter
from profirag.wiki.fetch_wiki_content import fetch_wiki_content

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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


def ingest_single_wiki(
    wiki_url: str,
    env_file: str = ".env",
    show_progress: bool = True,
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    mode: str = "hybrid",
    metadata: dict = None,
) -> dict:
    """Ingest a single wiki URL into the RAG pipeline.

    Args:
        wiki_url: Wiki URL to fetch and ingest
        env_file: Path to .env configuration file
        show_progress: Show progress information
        splitter_type: Override splitter type (sentence, token, semantic, chinese, markdown)
        chunk_size: Override chunk size
        chunk_overlap: Override chunk overlap
        mode: Index mode - "vector" or "hybrid"
        metadata: Additional metadata to attach to the document

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

    # Fetch wiki content
    if show_progress:
        print(f"Fetching wiki content from {wiki_url}...")

    wiki_data = fetch_wiki_content(wiki_url)

    if wiki_data is None:
        print(f"Failed to fetch wiki content from {wiki_url}")
        return {
            "wiki_url": wiki_url,
            "documents_loaded": 0,
            "documents_ingested": 0,
            "error": "Failed to fetch wiki content"
        }

    if show_progress:
        print(f"  - Title: {wiki_data.get('title')}")
        print(f"  - Document type: {wiki_data.get('document_type')}")
        print(f"  - Author: {wiki_data.get('author')}")
        print(f"  - Content length: {len(wiki_data.get('content', ''))} characters")

    # Create document from wiki content
    loader = DocumentLoader()
    base_metadata = {
        "source": wiki_url,
        "title": wiki_data.get("title", ""),
        "document_type": wiki_data.get("document_type", ""),
        "author": wiki_data.get("author", ""),
        "view_count": wiki_data.get("viewCount", 0),
        "last_update_time": wiki_data.get("lastUpdateTime", ""),
        "loader": "wiki"
    }
    # Merge custom metadata
    if metadata:
        base_metadata.update(metadata)
    document = loader.load_text(wiki_data.get("content", ""), metadata=base_metadata)

    # Ingest document
    if show_progress:
        print(f"Ingesting document into vector store...")

    start_time = time.time()
    ingest_result = pipeline.ingest_documents([document])
    doc_id = ingest_result["document_ids"][0]
    elapsed = time.time() - start_time

    if show_progress:
        print(f"  - Ingested in {elapsed:.2f} seconds")

    # Get final stats
    stats = pipeline.get_stats()

    if show_progress:
        print(f"\nIngestion complete!")
        print(f"  - Document ID: {doc_id}")
        print(f"  - Vector store count: {stats['vector_store']['count']}")

    return {
        "wiki_url": wiki_url,
        "title": wiki_data.get("title"),
        "documents_loaded": 1,
        "documents_ingested": 1,
        "document_id": doc_id,
        "elapsed_seconds": elapsed,
        "vector_store_count": stats["vector_store"]["count"],
    }


def ingest_wiki_list(
    wiki_list_file: str,
    env_file: str = ".env",
    show_progress: bool = True,
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    mode: str = "hybrid",
    metadata: dict = None,
) -> dict:
    """Ingest multiple wikis from a list file.

    Args:
        wiki_list_file: Path to file containing wiki URLs (one per line)
        env_file: Path to .env configuration file
        show_progress: Show progress information
        splitter_type: Override splitter type
        chunk_size: Override chunk size
        chunk_overlap: Override chunk overlap
        mode: Index mode - "vector" or "hybrid"
        metadata: Additional metadata to attach to all documents

    Returns:
        Dictionary with aggregated ingestion statistics
    """
    list_path = Path(wiki_list_file)
    if not list_path.exists():
        print(f"Wiki list file not found: {wiki_list_file}")
        return {"total_wikis": 0, "successful": 0, "failed": 0}

    # Read wiki URLs
    with open(list_path, 'r', encoding='utf-8') as f:
        wiki_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not wiki_urls:
        print(f"No wiki URLs found in {wiki_list_file}")
        return {"total_wikis": 0, "successful": 0, "failed": 0}

    if show_progress:
        print(f"Found {len(wiki_urls)} wiki URLs to process")

    # Process each wiki
    results = []
    for i, url in enumerate(wiki_urls, 1):
        if show_progress:
            print(f"\n[{i}/{len(wiki_urls)}] Processing: {url}")
        try:
            result = ingest_single_wiki(
                wiki_url=url,
                env_file=env_file,
                show_progress=False,  # Suppress individual progress
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                mode=mode,
                metadata=metadata,
            )
            results.append(result)

            if result.get("error"):
                if show_progress:
                    print(f"  - Failed: {result.get('error')}")
            else:
                if show_progress:
                    print(f"  - Success: {result.get('title')}")
        except Exception as ex:
            print(ex)

    # Aggregate results
    successful = sum(1 for r in results if not r.get("error"))
    failed = len(results) - successful

    if show_progress:
        print(f"\nSummary:")
        print(f"  - Total wikis: {len(wiki_urls)}")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {failed}")

    return {
        "total_wikis": len(wiki_urls),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest wiki content into ProfiRAG vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--wiki-url",
        "-u",
        type=str,
        default="https://wiki.huawei.com/domains/98/wiki/4773/WIKI202508127849624",
        help="Wiki URL to ingest",
    )
    parser.add_argument(
        "--wiki-list",
        "-l",
        type=str,
        default=None,
        help="File containing list of wiki URLs (one per line)",
    )
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env)",
    )
    parser.add_argument(
        "--splitter",
        "-s",
        type=str,
        choices=["sentence", "token", "semantic", "chinese", "markdown"],
        default="sentence",
        help="Splitter type (default: from .env or 'sentence')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size (default: from .env or 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap (default: from .env or 50)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Index mode: vector (semantic only), hybrid (both BM25 and vector, default)",
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default=None,
        help="Domain metadata value (e.g., finance, hr, it)",
    )
    parser.add_argument(
        "--group",
        "-g",
        type=str,
        default=None,
        help="Group metadata value (e.g., accounting, team-lead, devops)",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        action="append",
        help="Additional metadata as key=value pairs (can be used multiple times)",
    )

    args = parser.parse_args()

    show_progress = not args.quiet

    # Build metadata dictionary from arguments
    metadata = {}
    if args.domain:
        metadata["domain"] = args.domain
    if args.group:
        metadata["group"] = args.group
    if args.metadata:
        for item in args.metadata:
            if "=" not in item:
                print(f"Error: Invalid metadata format '{item}'. Expected format: key=value")
                return 1
            key, value = item.split("=", 1)
            metadata[key.strip()] = value.strip()

    # Check if either wiki-url or wiki-list is provided
    if not args.wiki_url and not args.wiki_list:
        print("Error: Either --wiki-url or --wiki-list must be specified")
        parser.print_help()
        return 1

    try:
        if args.wiki_list:
            # Process multiple wikis from a file
            result = ingest_wiki_list(
                wiki_list_file=args.wiki_list,
                env_file=args.env,
                splitter_type=args.splitter,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                show_progress=show_progress,
                mode=args.mode,
                metadata=metadata,
            )
        else:
            # Process single wiki
            result = ingest_single_wiki(
                wiki_url=args.wiki_url,
                env_file=args.env,
                splitter_type=args.splitter,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                show_progress=show_progress,
                mode=args.mode,
                metadata=metadata,
            )

        if not show_progress:
            # Output JSON for quiet mode
            import json
            print(json.dumps(result))

        return 0

    except Exception as e:
        print(f"Error during wiki ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())