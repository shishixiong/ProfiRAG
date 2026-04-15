#!/usr/bin/env python3
"""
ProfiRAG Response Generation Evaluation Flow

This script evaluates the answer generation quality of the RAG system:
1. Generate queries from existing vector store nodes (using LLM or keywords)
2. Retrieve context from vector store
3. Generate answers using LLM
4. Evaluate answer quality (faithfulness, relevancy, etc.)

Features:
- Sample caching: Only generates missing samples if cache exists
- Progress tracking: Shows how many new samples need to be generated

Usage:
    # Use LLM for query generation + answer generation + evaluation
    uv run python scripts/eval_response_flow.py --num-samples 20 --llm-queries

    # Continue from cache (if 5 samples exist, only generates 15 more)
    uv run python scripts/eval_response_flow.py --num-samples 20 --llm-queries --cache eval_samples_cache.json

    # Force regenerate all samples
    uv run python scripts/eval_response_flow.py --num-samples 10 --no-cache

    # Specific evaluators
    uv run python scripts/eval_response_flow.py --evaluators faithfulness,relevancy --num-samples 15
"""

import argparse
import sys
import os
import random
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.evaluation.dataset import EvalDataset, EvalItem
from profirag.evaluation.response import ResponseEvaluator
from profirag.retrieval import BM25Index
from llama_index.core.schema import TextNode


def get_cache_key(use_llm_queries: bool, top_k: int, llm_model: str) -> str:
    """Generate a cache key based on configuration."""
    config_str = f"llm_queries={use_llm_queries}|top_k={top_k}|model={llm_model}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_cache(cache_path: str, cache_key: str) -> Dict[str, Any]:
    """Load cached samples from file.

    Args:
        cache_path: Path to cache file
        cache_key: Cache key to identify the configuration

    Returns:
        Dictionary with cache data
    """
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return {"samples": [], "cache_key": cache_key}

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # Check if cache key matches current configuration
        if cache_data.get("cache_key") != cache_key:
            print(f"  Cache key mismatch, will regenerate all samples")
            return {"samples": [], "cache_key": cache_key}

        return cache_data
    except Exception as e:
        print(f"  Warning: Could not load cache: {e}")
        return {"samples": [], "cache_key": cache_key}


def save_cache(cache_path: str, cache_data: Dict[str, Any]) -> None:
    """Save samples to cache file.

    Args:
        cache_path: Path to cache file
        cache_data: Cache data dictionary
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text for query generation."""
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


def generate_query_from_text(text: str, style: str = "question") -> str:
    """Generate query from text content (simple keyword-based)."""
    keywords = extract_keywords_from_text(text, max_keywords=5)
    if not keywords:
        return text[:100].replace("\n", " ").strip() + "?"
    if len(keywords) >= 2:
        return f"{keywords[0]}与{keywords[1]}的关系是什么?"
    return f"{keywords[0]}是什么?"


def create_llm_client(env_file: str) -> tuple:
    """Create LLM client from environment config."""
    from openai import OpenAI

    load_dotenv(Path(env_file))
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4-turbo")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def generate_query_with_llm(llm_client: Any, model: str, text: str, max_retries: int = 3) -> str:
    """Generate a high-quality query from text using LLM with retry."""
    prompt = f"""基于以下文本内容，生成一个具体的、有针对性的问题。

要求：
1. 问题应该能够通过这段文本内容回答
2. 问题要具体，不要过于宽泛
3. 问题应该涉及文本中的关键概念或操作步骤
4. 使用中文提问

文本内容：
{text}

请直接输出一个问题，不要有任何解释或其他内容："""

    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            query = response.choices[0].message.content.strip()
            query = query.strip('"').strip("'").strip("「").strip("」")
            return query
        except Exception as e:
            error_str = str(e)
            if "529" in error_str or "500" in error_str or "overload" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2.0 * (attempt + 1)
                    print(f"    Query generation error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            print(f"  Warning: LLM query generation failed: {e}")
            return generate_query_from_text(text)


def generate_answer_with_llm(llm_client: Any, model: str, query: str, context: str, max_retries: int = 3) -> str:
    """Generate an answer using LLM based on retrieved context with retry."""
    prompt = f"""请基于以下参考信息回答问题。如果参考信息中没有相关内容，请说明无法回答。

参考信息：
{context}

问题：{query}

请用中文简洁回答，不要引用原文，直接给出答案："""

    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "529" in error_str or "500" in error_str or "overload" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 2.0 * (attempt + 1)
                    print(f"    Answer generation error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            print(f"  Warning: LLM answer generation failed: {e}")
            return "无法生成答案"


def load_nodes_from_vector_store(vector_store: Any, limit: int = 100) -> List[TextNode]:
    """Load nodes from vector store without embedding."""
    nodes = []
    try:
        if hasattr(vector_store, '_client') and hasattr(vector_store._client, 'scroll'):
            import json as json_lib
            results, _ = vector_store._client.scroll(
                collection_name=vector_store.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                node_content = point.payload.get("_node_content", "{}")
                try:
                    content_data = json_lib.loads(node_content)
                    text = content_data.get("text", "")
                except:
                    text = ""
                metadata = point.payload.get("metadata", {})
                node_id = str(point.id)
                if text:
                    nodes.append(TextNode(
                        id_=node_id,
                        text=text,
                        metadata=metadata,
                    ))
    except Exception as e:
        print(f"  Warning: Could not scroll vector store: {e}")
    return nodes


def run_response_evaluation(
    num_samples: int = 15,
    top_k: int = 5,
    evaluators: List[str] = ["faithfulness", "relevancy"],
    env_file: str = ".env",
    output_path: str = None,
    show_progress: bool = True,
    use_llm_queries: bool = False,
    use_llm_answers: bool = True,
    cache_path: Optional[str] = None,
    no_cache: bool = False,
) -> Dict[str, Any]:
    """Run response generation evaluation flow with caching support.

    Args:
        num_samples: Number of evaluation samples
        top_k: Number of documents to retrieve for context
        evaluators: List of response evaluators
        env_file: Path to .env file
        output_path: Path to save results
        show_progress: Show progress output
        use_llm_queries: Use LLM to generate queries
        use_llm_answers: Use LLM to generate answers (default True, always use LLM)
        cache_path: Path to cache file (default: eval_samples_cache.json)
        no_cache: Force regenerate all samples without caching

    Returns:
        Dictionary with evaluation results
    """
    if show_progress:
        print("=" * 60)
        print("ProfiRAG Response Generation Evaluation")
        print("=" * 60)

    # Set default cache path
    if not no_cache and cache_path is None:
        cache_path = "eval_samples_cache.json"
    elif no_cache:
        cache_path = None

    # 1. Load configuration and initialize pipeline
    if show_progress:
        print(f"\n[1] Loading configuration from {env_file}...")

    config = load_config(env_file)
    pipeline = RAGPipeline(config)
    llm_client, llm_model = create_llm_client(env_file)

    stats = pipeline.get_stats()
    vector_count = stats['vector_store']['count']

    if vector_count == 0:
        print("Error: Vector store is empty. Please ingest documents first.")
        return {"error": "Vector store is empty"}

    if show_progress:
        print(f"  - Vector store count: {vector_count}")
        print(f"  - LLM model: {llm_model}")

    # Generate cache key
    cache_key = get_cache_key(use_llm_queries, top_k, llm_model)

    # 2. Load existing cache
    existing_samples = []
    if cache_path and not no_cache:
        if show_progress:
            print(f"\n[2] Checking cache ({cache_path})...")

        cache_data = load_cache(cache_path, cache_key)
        existing_samples = cache_data.get("samples", [])

        if show_progress:
            print(f"  - Cached samples: {len(existing_samples)}")
            print(f"  - Cache key: {cache_key}")

    # Calculate how many new samples needed
    cached_count = len(existing_samples)
    new_samples_needed = max(0, num_samples - cached_count)

    if show_progress:
        print(f"\n  Samples status:")
        print(f"    - Required: {num_samples}")
        print(f"    - Cached: {cached_count}")
        print(f"    - Need to generate: {new_samples_needed}")

    # 3. Load nodes from vector store (only if need to generate new samples)
    nodes = []
    bm25 = None
    if new_samples_needed > 0:
        if show_progress:
            print(f"\n[3] Loading nodes from vector store...")

        nodes = load_nodes_from_vector_store(pipeline._vector_store, limit=num_samples + 50)

        if not nodes:
            print("Error: Could not load any nodes from vector store")
            return {"error": "No nodes loaded"}

        if show_progress:
            print(f"  - Loaded {len(nodes)} nodes")

        # Build BM25 index for retrieval
        bm25 = BM25Index(tokenizer="jieba", language="zh")
        bm25.add_nodes(nodes)
    else:
        if show_progress:
            print(f"\n[3] All samples cached, skipping node loading")

    # 4. Generate new samples
    eval_data = existing_samples.copy()  # Start with cached samples

    if new_samples_needed > 0:
        if show_progress:
            print(f"\n[4] Generating {new_samples_needed} new samples...")
            if use_llm_queries:
                print(f"  - Using LLM for query generation")

        # Filter out node_ids already used in cached samples
        used_node_ids = set(s.get("node_id") for s in existing_samples)
        available_nodes = [n for n in nodes if n.node_id not in used_node_ids]

        if len(available_nodes) < new_samples_needed:
            print(f"  Warning: Only {len(available_nodes)} nodes available, reducing new samples")
            new_samples_needed = len(available_nodes)

        # Sample nodes for new samples
        new_sampled_nodes = random.sample(available_nodes, new_samples_needed)

        for i, node in enumerate(new_sampled_nodes):
            # Generate query
            if use_llm_queries:
                query = generate_query_with_llm(llm_client, llm_model, node.text)
            else:
                query = generate_query_from_text(node.text)

            # Retrieve context using BM25
            retrieved = bm25.retrieve(query, top_k=top_k)
            contexts = [n.node.text for n in retrieved]

            # Generate answer
            context_text = "\n\n".join(contexts[:3])  # Use top 3 contexts
            answer = generate_answer_with_llm(llm_client, llm_model, query, context_text)

            new_sample = {
                "query": query,
                "contexts": contexts,
                "answer": answer,
                "expected_text": node.text,
                "node_id": node.node_id,
            }
            eval_data.append(new_sample)

            # Save cache after each sample (incremental caching)
            if cache_path:
                cache_data = {
                    "cache_key": cache_key,
                    "config": {
                        "use_llm_queries": use_llm_queries,
                        "top_k": top_k,
                        "llm_model": llm_model,
                    },
                    "samples": eval_data,
                }
                save_cache(cache_path, cache_data)

            if show_progress and (i + 1) % 3 == 0:
                print(f"    Generated {i + 1}/{new_samples_needed} new samples (total: {len(eval_data)})")

        if show_progress:
            print(f"  - Completed: {len(eval_data)} total samples")
    else:
        if show_progress:
            print(f"\n[4] No new samples needed, using cached data")

    if show_progress:
        print(f"\n  Sample queries and answers:")
        for i, data in enumerate(eval_data[:3]):
            print(f"    {i+1}. Query: {data['query'][:50]}...")
            print(f"       Answer: {data['answer'][:80]}...")

    # 5. Run response evaluation
    if show_progress:
        print(f"\n[5] Running response evaluation...")
        print(f"  - Evaluators: {evaluators}")
        print(f"  - Total samples: {len(eval_data)}")

    # Create simple LLM wrapper for llama_index evaluators
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.core.llms import CompletionResponse, LLMMetadata
    from openai import OpenAI as OpenAIClient

    class MiniMaxLLM(CustomLLM):
        """Custom LLM wrapper for MiniMax/DashScope API."""

        client: Any = None
        model_actual: str = ""

        def __init__(self, client: OpenAIClient, model: str, **kwargs):
            super().__init__(client=client, model_actual=model, **kwargs)

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=4096,
                num_output=512,
                model_name="gpt-3.5-turbo",
            )

        def complete(self, prompt: str, **kwargs) -> CompletionResponse:
            """Complete with retry logic for transient API errors."""
            max_retries = 3
            retry_delay = 2.0  # seconds

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_actual,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=kwargs.get("max_tokens", 512),
                    )
                    return CompletionResponse(text=response.choices[0].message.content)
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Check if it's a retryable error (5xx, overload)
                        error_str = str(e)
                        if "529" in error_str or "500" in error_str or "overload" in error_str.lower():
                            wait_time = retry_delay * (attempt + 1)
                            print(f"    API error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    raise  # Re-raise if not retryable or max retries reached

        def stream_complete(self, prompt: str, **kwargs):
            response = self.complete(prompt, **kwargs)
            yield response

    eval_llm = MiniMaxLLM(llm_client, llm_model)

    response_evaluator = ResponseEvaluator(
        llm=eval_llm,
        evaluators=evaluators,
    )

    # Evaluate each response
    eval_results = []
    for data in eval_data:
        result = response_evaluator.evaluate(
            query=data["query"],
            response=data["answer"],
            contexts=data["contexts"],
        )
        eval_results.append(result)

    # 6. Compute summary
    summary = {}
    for eval_name in evaluators:
        scores = []
        passing_count = 0
        for result in eval_results:
            if eval_name in result:
                r = result[eval_name]
                if r.score is not None:
                    scores.append(r.score)
                if r.passing:
                    passing_count += 1

        summary[eval_name] = {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "passing_rate": passing_count / len(eval_results) if eval_results else 0.0,
            "count": len(scores),
        }

    if show_progress:
        print(f"\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"\nTotal samples: {len(eval_results)}")
        print(f"\nMetrics Summary:")
        for eval_name, stats in summary.items():
            print(f"  - {eval_name}:")
            print(f"      Mean score: {stats['mean_score']:.4f}")
            print(f"      Passing rate: {stats['passing_rate']:.4f}")

        # Show detailed results
        print(f"\nDetailed Results (first 5):")
        for i, (data, result) in enumerate(zip(eval_data[:5], eval_results[:5])):
            print(f"\n  Sample {i+1}:")
            print(f"    Query: {data['query'][:60]}...")
            print(f"    Answer: {data['answer'][:100]}...")
            for eval_name in evaluators:
                if eval_name in result:
                    r = result[eval_name]
                    score = r.score if r.score else 0.0
                    passing = "PASS" if r.passing else "FAIL"
                    print(f"    {eval_name}: {score:.4f} ({passing})")

    # 7. Save results
    output_data = {
        "config": {
            "num_samples": num_samples,
            "cached_samples": cached_count,
            "new_samples": new_samples_needed,
            "top_k": top_k,
            "evaluators": evaluators,
            "use_llm_queries": use_llm_queries,
            "llm_model": llm_model,
            "cache_path": cache_path,
        },
        "summary": summary,
        "total_samples": len(eval_results),
        "details": [
            {
                "query": d["query"],
                "answer": d["answer"],
                "contexts_count": len(d["contexts"]),
                **{
                    f"{eval_name}_score": result.get(eval_name).score if eval_name in result and result[eval_name].score else 0.0
                    for eval_name in evaluators
                },
                **{
                    f"{eval_name}_passing": result.get(eval_name).passing if eval_name in result else False
                    for eval_name in evaluators
                },
            }
            for d, result in zip(eval_data, eval_results)
        ],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        if show_progress:
            print(f"\nResults saved to: {output_path}")

    if cache_path and show_progress:
        print(f"Cache saved to: {cache_path} ({len(eval_data)} samples)")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run response generation evaluation with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=20,
        help="Number of evaluation samples (default: 20)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve for context (default: 5)",
    )
    parser.add_argument(
        "--evaluators", "-e",
        type=str,
        default="faithfulness,relevancy",
        help="Comma-separated evaluators (default: faithfulness,relevancy)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="response_eval_results.json",
        help="Output results file (default: response_eval_results.json)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--llm-queries",
        action="store_true",
        help="Use LLM to generate evaluation queries (better quality)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="eval_samples_cache.json",
        help="Cache file path (default: eval_samples_cache.json)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force regenerate all samples without caching",
    )

    args = parser.parse_args()

    evaluators = args.evaluators.split(",")

    try:
        result = run_response_evaluation(
            num_samples=args.num_samples,
            top_k=args.top_k,
            evaluators=evaluators,
            env_file=args.env,
            output_path=args.output,
            show_progress=not args.quiet,
            use_llm_queries=args.llm_queries,
            cache_path=args.cache if not args.no_cache else None,
            no_cache=args.no_cache,
        )
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())