#!/usr/bin/env python3
"""
Generate evaluation dataset from documents.

This script generates an EvalDataset JSON file without requiring
embedding or vector store. The dataset can be used later for
retrieval evaluation once the vector store is populated.

Usage:
    uv run python scripts/generate_eval_dataset.py --documents ./markdown --output eval_data.json
"""

import argparse
import random
import json
from pathlib import Path
from typing import List

sys_path_insert = str(Path(__file__).parent.parent / "src")
import sys
sys.path.insert(0, sys_path_insert)

from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import TextSplitter, ChineseTextSplitter
from profirag.evaluation.dataset import EvalDataset, EvalItem


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text."""
    import re
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "这", "那", "是", "有", "和", "的", "了", "在", "不", "也", "就", "都",
        "可以", "会", "要", "能", "一个", "这个", "那个", "什么", "怎么",
        "工具", "使用", "功能", "方法", "进行", "操作", "连接", "配置",
    }
    words = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", text.lower())
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:max_keywords]]


def generate_query(text: str) -> str:
    """Generate a query from text."""
    keywords = extract_keywords(text)
    if not keywords:
        return "相关内容是什么?"

    if len(keywords) >= 2:
        return f"{keywords[0]}与{keywords[1]}有什么关系?"
    return f"{keywords[0]}是什么?"


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation dataset")
    parser.add_argument("--documents", "-d", default="./markdown", help="Documents directory")
    parser.add_argument("--output", "-o", default="eval_data.json", help="Output JSON file")
    parser.add_argument("--num-samples", "-n", type=int, default=20, help="Number of samples")
    parser.add_argument("--splitter", "-s", default="chinese", help="Splitter type")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")

    args = parser.parse_args()

    print(f"Loading documents from {args.documents}...")
    loader = DocumentLoader(encoding="utf-8")
    documents = loader.load_directory(args.documents, recursive=True)

    if not documents:
        print("No documents found!")
        return 1

    print(f"Loaded {len(documents)} documents")

    print(f"Chunking with {args.splitter} splitter...")
    if args.splitter == "chinese":
        splitter = ChineseTextSplitter(args.chunk_size, args.chunk_overlap)
    else:
        splitter = TextSplitter(args.splitter, args.chunk_size, args.chunk_overlap)

    nodes = splitter.split_documents(documents)
    print(f"Generated {len(nodes)} chunks")

    print(f"Sampling {args.num_samples} nodes for evaluation...")
    sample_size = min(args.num_samples, len(nodes))
    sampled_nodes = random.sample(nodes, sample_size)

    items = []
    for node in sampled_nodes:
        query = generate_query(node.text)
        items.append(EvalItem(
            query=query,
            expected_ids=[node.node_id],
            expected_texts=[node.text[:300]],
        ))

    dataset = EvalDataset(items=items)

    # Save dataset
    dataset.save(args.output)
    print(f"\nSaved {len(dataset)} evaluation items to {args.output}")

    # Print sample queries
    print("\nSample queries:")
    for i, item in enumerate(items[:5]):
        print(f"  {i+1}. {item.query}")
        print(f"     Node ID: {item.expected_ids[0]}")

    print("\nNote: Node IDs are generated during chunking. For retrieval evaluation,")
    print("      these IDs must match the IDs in your vector store. Use the same")
    print("      splitter configuration when ingesting into the vector store.")

    return 0


if __name__ == "__main__":
    sys.exit(main())