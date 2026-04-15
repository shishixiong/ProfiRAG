#!/usr/bin/env python3
"""
ProfiRAG Chunking Evaluation Script

Compare different chunking strategies and evaluate their effectiveness.

Usage:
    # Statistics only (no embedding/retrieval needed)
    python scripts/evaluate_chunking.py \
        --documents ./markdown \
        --stats-only \
        --configs sentence:512:50,chinese:512:50,token:512:50

    # With quality evaluation (requires LLM)
    python scripts/evaluate_chunking.py \
        --documents ./markdown \
        --quality-eval \
        --configs sentence:512:50,chinese:512:50

    # Full evaluation including retrieval impact (requires embedding + dataset)
    python scripts/evaluate_chunking.py \
        --documents ./markdown \
        --eval-dataset ./eval_data.json \
        --configs sentence:512:50,sentence:1024:100,chinese:512:50 \
        --output ./chunking_results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.ingestion.loaders import DocumentLoader
from profirag.evaluation.chunking import (
    ChunkingEvaluator,
    ChunkingCompareResults,
    parse_config_string,
)
from profirag.evaluation.dataset import EvalDataset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare chunking strategies"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="./markdown",
        help="Directory containing documents to chunk",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="sentence:512:50",
        help="Comma-separated config strings (format: splitter:size:overlap)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute statistics (no quality or retrieval evaluation)",
    )
    parser.add_argument(
        "--quality-eval",
        action="store_true",
        help="Enable LLM-based quality evaluation",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Evaluation dataset for retrieval impact testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./chunking_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--quality-sample-size",
        type=int,
        default=10,
        help="Number of chunks to sample for quality evaluation",
    )

    args = parser.parse_args()

    # Parse configurations
    config_strings = args.configs.split(",")
    configs = [parse_config_string(c) for c in config_strings]

    print(f"=== Chunking Evaluation ===")
    print(f"Documents: {args.documents}")
    print(f"Configurations: {config_strings}")
    print()

    # Load documents
    print("Loading documents...")
    loader = DocumentLoader(encoding="utf-8")
    doc_path = Path(args.documents)

    if not doc_path.exists():
        print(f"Error: Document directory not found: {args.documents}")
        sys.exit(1)

    documents = loader.load_directory(args.documents, recursive=True)
    if not documents:
        print(f"No documents found in {args.documents}")
        sys.exit(0)

    print(f"  Loaded {len(documents)} documents")
    print()

    # Initialize evaluator
    evaluator = ChunkingEvaluator(
        use_quality_eval=args.quality_eval and not args.stats_only,
        quality_sample_size=args.quality_sample_size,
    )

    # Set up LLM for quality evaluation if requested
    if args.quality_eval and not args.stats_only:
        try:
            from openai import OpenAI as OpenAIClient
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
            if api_key:
                # Create a simple wrapper for LLM calls
                client = OpenAIClient(api_key=api_key, base_url=base_url)

                class SimpleLLMWrapper:
                    """Simple LLM wrapper for quality evaluation."""

                    def complete(self, prompt: str) -> str:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1000,  # Need enough tokens for reasoning models
                        )
                        from llama_index.core.llms import CompletionResponse
                        return CompletionResponse(text=response.choices[0].message.content)

                evaluator.llm = SimpleLLMWrapper()
                print(f"  LLM initialized for quality evaluation (model={model})")
            else:
                print("  Warning: OPENAI_API_KEY not set, quality evaluation disabled")
                evaluator.use_quality_eval = False
        except Exception as e:
            print(f"  Warning: Could not initialize LLM: {e}")
            evaluator.use_quality_eval = False

    # Run evaluation
    print("Evaluating configurations...")
    results = evaluator.compare_configs(documents, configs)

    # Print results
    if args.format == "text":
        print(results.get_summary_text())
    else:
        print(json.dumps(results.model_dump(), indent=2, ensure_ascii=False))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(str(output_path))
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n=== Comparison Table ===")
    print(f"{'Config':<25} {'Chunks':<10} {'Avg Len':<10} {'Std Dev':<10}")
    print("-" * 55)
    for config_name, data in results.comparison_table.items():
        print(f"{config_name:<25} {data['total_chunks']:<10} {data['avg_length']:<10.1f} {data['std_length']:<10.1f}")


if __name__ == "__main__":
    main()