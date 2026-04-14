#!/usr/bin/env python3
"""
ProfiRAG RAG Evaluation Script

Evaluate RAG system performance using retrieval and response metrics.

Usage:
    # Run evaluation with existing dataset
    python scripts/evaluate_rag.py --dataset ./eval_data.json

    # Generate dataset from documents directory
    python scripts/evaluate_rag.py --generate-from-docs ./documents --output ./eval_data.json

    # Generate dataset from existing pipeline (requires .env)
    python scripts/evaluate_rag.py --generate-from-pipeline --output ./eval_data.json

    # Create hardcoded sample dataset
    python scripts/evaluate_rag.py --create-sample --output ./eval_data.json

    # List available metrics
    python scripts/evaluate_rag.py --list-metrics

Example eval_data.json format:
{
    "items": [
        {
            "query": "What is RAG?",
            "expected_ids": ["doc_id_1", "doc_id_2"],
            "expected_texts": ["..."],
            "reference_answer": "RAG is..."
        }
    ]
}
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.evaluation import (
    EvalDataset,
    RAGEvalRunner,
    RetrievalEvaluator,
    ResponseEvaluator,
    create_sample_dataset,
    create_dataset_from_documents,
    create_dataset_from_pipeline,
)


def run_evaluation(
    dataset_path: str,
    env_file: str = ".env",
    retrieval_metrics: list = None,
    response_metrics: list = None,
    top_k: int = 10,
    output_path: str = None,
    show_progress: bool = True,
) -> dict:
    """Run RAG evaluation.

    Args:
        dataset_path: Path to evaluation dataset JSON
        env_file: Path to .env configuration
        retrieval_metrics: List of retrieval metrics
        response_metrics: List of response evaluators
        top_k: Number of documents to retrieve
        output_path: Path to save results JSON
        show_progress: Show progress output

    Returns:
        Dictionary with evaluation results
    """
    # Default metrics
    retrieval_metrics = retrieval_metrics or ["hit_rate", "mrr", "precision", "recall"]
    response_metrics = response_metrics or ["faithfulness", "relevancy"]

    # Load configuration
    if show_progress:
        print(f"Loading configuration from {env_file}...")

    config = load_config(env_file)

    # Initialize pipeline
    if show_progress:
        print(f"Initializing RAG pipeline...")
        print(f"  - Embedding model: {config.embedding.model}")
        print(f"  - LLM model: {config.llm.model}")
        print(f"  - Storage type: {config.storage.type}")

    pipeline = RAGPipeline(config)

    # Load evaluation dataset
    if show_progress:
        print(f"Loading evaluation dataset from {dataset_path}...")

    dataset = EvalDataset.from_json(dataset_path)
    if show_progress:
        print(f"  - Found {len(dataset)} queries")

    # Initialize evaluation runner
    if show_progress:
        print(f"Setting up evaluators...")
        print(f"  - Retrieval metrics: {retrieval_metrics}")
        print(f"  - Response metrics: {response_metrics}")

    runner = RAGEvalRunner(
        pipeline=pipeline,
        llm=pipeline._llm,
        retrieval_metrics=retrieval_metrics,
        response_metrics=response_metrics,
        top_k=top_k,
    )

    # Run evaluation
    if show_progress:
        print(f"\nRunning evaluation...")

    results = runner.run_evaluation(dataset, show_progress=show_progress)

    # Print summary
    if show_progress:
        print("\n" + results.get_summary_text())

    # Save results
    if output_path:
        results.save(output_path)
        if show_progress:
            print(f"\nResults saved to {output_path}")

    return results.model_dump()


def generate_from_documents(
    documents_dir: str,
    output_path: str,
    num_samples: int = 10,
    query_style: str = "question",
    chunk_size: int = 512,
    show_progress: bool = True,
) -> None:
    """Generate evaluation dataset from documents directory.

    Args:
        documents_dir: Path to documents
        output_path: Output JSON path
        num_samples: Number of samples to generate
        query_style: Query style (question, keyword, summary)
        chunk_size: Chunk size
        show_progress: Show progress
    """
    if show_progress:
        print(f"Generating evaluation dataset from documents...")
        print(f"  - Documents directory: {documents_dir}")
        print(f"  - Number of samples: {num_samples}")
        print(f"  - Query style: {query_style}")
        print(f"  - Chunk size: {chunk_size}")

    dataset = create_dataset_from_documents(
        documents_dir=documents_dir,
        output_path=output_path,
        num_samples=num_samples,
        query_style=query_style,
        chunk_size=chunk_size,
    )

    if show_progress:
        print(f"\nDataset generated with {len(dataset)} items")
        print(f"Saved to: {output_path}")
        print(f"\nSample queries:")
        for i, item in enumerate(dataset.items[:3]):
            print(f"  {i+1}. {item.query}")
            print(f"     Expected IDs: {item.expected_ids}")


def generate_from_pipeline(
    env_file: str,
    output_path: str,
    num_samples: int = 10,
    query_style: str = "question",
    generate_answers: bool = False,
    show_progress: bool = True,
) -> None:
    """Generate evaluation dataset from existing RAG pipeline.

    Args:
        env_file: Path to .env configuration
        output_path: Output JSON path
        num_samples: Number of samples
        query_style: Query style
        generate_answers: Generate reference answers using LLM
        show_progress: Show progress
    """
    if show_progress:
        print(f"Loading configuration from {env_file}...")

    config = load_config(env_file)

    if show_progress:
        print(f"Initializing RAG pipeline...")
        print(f"  - Storage type: {config.storage.type}")

    pipeline = RAGPipeline(config)

    if show_progress:
        print(f"Generating evaluation dataset from vector store...")
        print(f"  - Number of samples: {num_samples}")
        print(f"  - Query style: {query_style}")
        print(f"  - Generate answers: {generate_answers}")
        print(f"  - Vector store count: {pipeline.get_stats()['vector_store']['count']}")

    dataset = create_dataset_from_pipeline(
        pipeline=pipeline,
        output_path=output_path,
        num_samples=num_samples,
        query_style=query_style,
        generate_answers=generate_answers,
    )

    if show_progress:
        print(f"\nDataset generated with {len(dataset)} items")
        print(f"Saved to: {output_path}")
        print(f"\nSample queries:")
        for i, item in enumerate(dataset.items[:3]):
            print(f"  {i+1}. {item.query}")
            print(f"     Expected IDs: {item.expected_ids}")


def create_sample(output_path: str = "eval_data.json") -> None:
    """Create hardcoded sample dataset.

    Args:
        output_path: Output path
    """
    dataset = create_sample_dataset()
    dataset.save(output_path)
    print(f"Sample dataset created at {output_path}")
    print(f"Contains {len(dataset)} sample queries")
    print("\nNote: This is a hardcoded sample. Use --generate-from-docs or --generate-from-pipeline")
    print("      to generate a dataset from actual documents.")


def list_available_metrics() -> None:
    """Print all available evaluation metrics."""
    print("=== Available Retrieval Metrics ===")
    for metric in RetrievalEvaluator.get_available_metrics():
        descriptions = {
            "hit_rate": "Whether any relevant document was retrieved (0 or 1)",
            "mrr": "Mean Reciprocal Rank - position of first relevant document",
            "precision": "Fraction of retrieved documents that are relevant",
            "recall": "Fraction of relevant documents that were retrieved",
            "ndcg": "Normalized Discounted Cumulative Gain",
            "ap": "Average Precision",
        }
        print(f"  - {metric}: {descriptions.get(metric, 'No description')}")

    print("\n=== Available Response Evaluators ===")
    for evaluator in ResponseEvaluator.get_available_evaluators():
        descriptions = {
            "faithfulness": "Whether response is grounded in retrieved context",
            "relevancy": "Whether response is relevant to the query",
            "correctness": "Whether response matches reference answer (requires reference)",
            "answer_relevancy": "How well response addresses the query",
            "context_relevancy": "How relevant the context is to the query",
        }
        print(f"  - {evaluator}: {descriptions.get(evaluator, 'No description')}")

    print("\n=== Query Styles for Dataset Generation ===")
    print("  - question: Generate questions like 'What is X?'")
    print("  - keyword: Use keywords extracted from text")
    print("  - summary: Generate 'Tell me about X' style queries")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset source options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="eval_data.json",
        help="Path to existing evaluation dataset JSON file",
    )
    parser.add_argument(
        "--generate-from-docs",
        type=str,
        metavar="DIR",
        help="Generate dataset from documents directory",
    )
    parser.add_argument(
        "--generate-from-pipeline",
        action="store_true",
        help="Generate dataset from existing RAG pipeline (requires .env)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create hardcoded sample dataset",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save results/dataset JSON",
    )

    # Generation options
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)",
    )
    parser.add_argument(
        "--query-style",
        type=str,
        choices=["question", "keyword", "summary"],
        default="question",
        help="Query generation style (default: question)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for document processing (default: 512)",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Generate reference answers using LLM (slow, costs tokens)",
    )

    # Evaluation options
    parser.add_argument(
        "--env", "-e",
        type=str,
        default=".env",
        help="Path to .env configuration file (default: .env)",
    )
    parser.add_argument(
        "--retrieval-metrics", "-r",
        type=str,
        default="hit_rate,mrr,precision,recall",
        help="Comma-separated retrieval metrics (default: hit_rate,mrr,precision,recall)",
    )
    parser.add_argument(
        "--response-metrics", "-s",
        type=str,
        default="faithfulness,relevancy",
        help="Comma-separated response evaluators (default: faithfulness,relevancy)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)",
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available evaluation metrics",
    )

    args = parser.parse_args()

    if args.list_metrics:
        list_available_metrics()
        return 0

    show_progress = not args.quiet

    # Generate from documents
    if args.generate_from_docs:
        output = args.output or "eval_data.json"
        generate_from_documents(
            documents_dir=args.generate_from_docs,
            output_path=output,
            num_samples=args.num_samples,
            query_style=args.query_style,
            chunk_size=args.chunk_size,
            show_progress=show_progress,
        )
        return 0

    # Generate from pipeline
    if args.generate_from_pipeline:
        output = args.output or "eval_data.json"
        generate_from_pipeline(
            env_file=args.env,
            output_path=output,
            num_samples=args.num_samples,
            query_style=args.query_style,
            generate_answers=args.generate_answers,
            show_progress=show_progress,
        )
        return 0

    # Create sample
    if args.create_sample:
        output = args.output or args.dataset
        create_sample(output)
        return 0

    # Run evaluation
    retrieval_metrics = args.retrieval_metrics.split(",")
    response_metrics = args.response_metrics.split(",")

    try:
        result = run_evaluation(
            dataset_path=args.dataset,
            env_file=args.env,
            retrieval_metrics=retrieval_metrics,
            response_metrics=response_metrics,
            top_k=args.top_k,
            output_path=args.output,
            show_progress=show_progress,
        )

        if not show_progress and args.output:
            print(f"Results saved to {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())