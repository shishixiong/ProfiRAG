#!/usr/bin/env python3
"""
ProfiRAG Retrieval Evaluation Flow

This script performs a complete retrieval evaluation:
1. Load documents from markdown directory
2. Chunk documents and ingest into vector store
3. Generate evaluation dataset using real node IDs
4. Run retrieval evaluation

Usage:
    # Full evaluation pipeline
    uv run python scripts/eval_retrieval_flow.py

    # Specify options
    uv run python scripts/eval_retrieval_flow.py --num-samples 20 --splitter chinese --top-k 5
"""

import argparse
import sys
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter
from profirag.evaluation.dataset import EvalDataset, EvalItem
from profirag.evaluation.retrieval import RetrievalEvaluator


def extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text for query generation."""
    import re
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "这", "那", "是", "有", "和", "的", "了", "在", "不", "也", "就", "都",
        "可以", "会", "要", "能", "一个", "这个", "那个", "什么", "怎么", "如何",
        "工具", "使用", "功能", "方法", "进行", "操作", "连接", "配置", "设置",
    }
    words = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", text.lower())
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:max_keywords]]


def generate_query_from_text(text: str, style: str = "question") -> str:
    """Generate query from text content."""
    keywords = extract_keywords_from_text(text, max_keywords=5)
    if not keywords:
        return text[:100].replace("\n", " ").strip() + "?"

    if style == "keyword":
        return " ".join(keywords[:3])

    if len(keywords) >= 2:
        return f"{keywords[0]}与{keywords[1]}的关系是什么?"
    else:
        return f"{keywords[0]}是什么?"


def run_retrieval_evaluation(
    documents_dir: str,
    num_samples: int = 15,
    splitter_type: str = "chinese",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    top_k: int = 10,
    metrics: List[str] = ["hit_rate", "mrr", "precision", "recall"],
    env_file: str = ".env",
    output_path: str = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run complete retrieval evaluation flow.

    Args:
        documents_dir: Path to documents directory
        num_samples: Number of evaluation samples
        splitter_type: Splitter type (sentence, chinese, token)
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        top_k: Number of documents to retrieve
        metrics: Retrieval metrics to compute
        env_file: Path to .env file
        output_path: Path to save results
        show_progress: Show progress output

    Returns:
        Dictionary with evaluation results
    """
    # 1. Load configuration
    if show_progress:
        print("=" * 60)
        print("ProfiRAG Retrieval Evaluation Flow")
        print("=" * 60)
        print(f"\n[1] Loading configuration from {env_file}...")

    config = load_config(env_file)

    # Override chunking config
    config.chunking.splitter_type = splitter_type
    config.chunking.chunk_size = chunk_size
    config.chunking.chunk_overlap = chunk_overlap

    # 2. Load documents
    if show_progress:
        print(f"\n[2] Loading documents from {documents_dir}...")

    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_directory(documents_dir, recursive=True)

    if not documents:
        print(f"Error: No documents found in {documents_dir}")
        return {"error": "No documents found"}

    if show_progress:
        print(f"  - Loaded {len(documents)} documents")

    # 3. Create splitter and chunk documents
    if show_progress:
        print(f"\n[3] Chunking documents...")
        print(f"  - Splitter: {splitter_type}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Chunk overlap: {chunk_overlap}")

    if splitter_type == "chinese":
        splitter = ChineseTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        splitter = TextSplitter(splitter_type=splitter_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    nodes = splitter.split_documents(documents)

    if show_progress:
        print(f"  - Generated {len(nodes)} chunks")
        lengths = [len(n.text) for n in nodes]
        print(f"  - Avg chunk length: {sum(lengths)/len(lengths):.1f}")

    # 4. Initialize pipeline and ingest nodes
    if show_progress:
        print(f"\n[4] Initializing vector store and ingesting chunks...")

    pipeline = RAGPipeline(config)

    # Clear existing data if needed
    # pipeline.clear()

    # Ingest nodes
    node_ids = pipeline.ingest_nodes(nodes, update_bm25=True)

    if show_progress:
        stats = pipeline.get_stats()
        print(f"  - Vector store count: {stats['vector_store']['count']}")
        print(f"  - BM25 index count: {stats['bm25_index']['count']}")

    # 5. Generate evaluation dataset using real node IDs
    if show_progress:
        print(f"\n[5] Generating evaluation dataset...")
        print(f"  - Sampling {num_samples} chunks")

    # Sample nodes for evaluation
    sample_size = min(num_samples, len(nodes))
    sampled_nodes = random.sample(nodes, sample_size)

    eval_items = []
    for node in sampled_nodes:
        query = generate_query_from_text(node.text, style="question")
        eval_items.append(EvalItem(
            query=query,
            expected_ids=[node.node_id],
            expected_texts=[node.text[:200]],
        ))

    dataset = EvalDataset(items=eval_items)

    if show_progress:
        print(f"  - Generated {len(dataset)} evaluation items")
        print(f"\n  Sample queries:")
        for i, item in enumerate(dataset.items[:3]):
            print(f"    {i+1}. {item.query}")
            print(f"       Expected node: {item.expected_ids[0][:20]}...")

    # Save dataset
    dataset_file = "eval_dataset_temp.json"
    dataset.save(dataset_file)

    # 6. Run retrieval evaluation
    if show_progress:
        print(f"\n[6] Running retrieval evaluation...")
        print(f"  - Metrics: {metrics}")
        print(f"  - Top-K: {top_k}")

    # Create retriever from pipeline
    from llama_index.core import VectorIndexRetriever
    retriever = VectorIndexRetriever(
        index=pipeline._index,
        similarity_top_k=top_k,
    )

    evaluator = RetrievalEvaluator(
        retriever=retriever,
        metrics=metrics,
    )

    results = evaluator.evaluate_dataset(dataset, show_progress=show_progress)

    # Compute summary metrics
    summary = evaluator.get_metrics_summary(results)

    if show_progress:
        print(f"\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"\nTotal queries: {len(results)}")
        print(f"\nMetrics Summary:")
        for metric, value in summary.items():
            print(f"  - {metric}: {value:.4f}")

        # Show detailed results for first few queries
        print(f"\nDetailed Results (first 5):")
        for i, result in enumerate(results[:5]):
            print(f"\n  Query {i+1}: {dataset.items[i].query[:50]}...")
            for metric in metrics:
                val = result.metric_vals_dict.get(metric, 0.0)
                print(f"    {metric}: {val:.4f}")

    # Save results
    output_data = {
        "config": {
            "documents_dir": documents_dir,
            "splitter_type": splitter_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_samples": num_samples,
            "top_k": top_k,
            "metrics": metrics,
        },
        "summary": summary,
        "total_queries": len(results),
        "vector_store_count": pipeline.get_stats()['vector_store']['count'],
        "details": [
            {
                "query": dataset.items[i].query,
                "expected_id": dataset.items[i].expected_ids[0],
                **{m: r.metric_vals_dict.get(m, 0.0) for m in metrics}
            }
            for i, r in enumerate(results)
        ],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        if show_progress:
            print(f"\nResults saved to: {output_path}")

    # Cleanup temp file
    Path(dataset_file).unlink(missing_ok=True)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run complete retrieval evaluation flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--documents", "-d",
        type=str,
        default="./markdown",
        help="Documents directory (default: ./markdown)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=15,
        help="Number of evaluation samples (default: 15)",
    )
    parser.add_argument(
        "--splitter", "-s",
        type=str,
        choices=["sentence", "token", "chinese"],
        default="chinese",
        help="Splitter type (default: chinese)",
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
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)",
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        default="hit_rate,mrr,precision,recall",
        help="Comma-separated metrics (default: hit_rate,mrr,precision,recall)",
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="retrieval_eval_results.json",
        help="Output results file (default: retrieval_eval_results.json)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    metrics = args.metrics.split(",")

    try:
        result = run_retrieval_evaluation(
            documents_dir=args.documents,
            num_samples=args.num_samples,
            splitter_type=args.splitter,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            metrics=metrics,
            env_file=args.env,
            output_path=args.output,
            show_progress=not args.quiet,
        )
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())