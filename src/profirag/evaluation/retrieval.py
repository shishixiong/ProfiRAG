"""Retrieval evaluation module"""

import asyncio
from typing import List, Dict, Any, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation.retrieval.base import (
    RetrievalEvalResult,
    RetrievalEvalMode,
)
from llama_index.core.evaluation.retrieval.metrics import (
    HitRate,
    MRR,
    Precision,
    Recall,
    NDCG,
    AveragePrecision,
)

from .dataset import EvalDataset


# Available retrieval metrics
AVAILABLE_RETRIEVAL_METRICS = {
    "hit_rate": HitRate,
    "mrr": MRR,
    "precision": Precision,
    "recall": Recall,
    "ndcg": NDCG,
    "ap": AveragePrecision,  # Average Precision
}


class RetrievalEvaluator:
    """Retrieval evaluator for measuring retrieval quality.

    Supports multiple metrics:
    - hit_rate: Whether any relevant doc was retrieved
    - mrr: Mean Reciprocal Rank (position of first relevant doc)
    - precision: Fraction of retrieved docs that are relevant
    - recall: Fraction of relevant docs that were retrieved
    - ndcg: Normalized Discounted Cumulative Gain
    - ap: Average Precision

    Args:
        retriever: LlamaIndex retriever to evaluate
        metrics: List of metric names to compute
        node_postprocessors: Optional post-processors to apply after retrieval
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        metrics: List[str] = ["hit_rate", "mrr", "precision", "recall"],
        node_postprocessors: Optional[List[Any]] = None,
    ):
        """Initialize retrieval evaluator.

        Args:
            retriever: The retriever to evaluate
            metrics: List of metric names (default: hit_rate, mrr, precision, recall)
            node_postprocessors: Optional post-processors
        """
        # Validate metrics
        for metric in metrics:
            if metric not in AVAILABLE_RETRIEVAL_METRICS:
                raise ValueError(
                    f"Unknown metric: {metric}. "
                    f"Available: {list(AVAILABLE_RETRIEVAL_METRICS.keys())}"
                )

        self.metrics = metrics
        self.retriever = retriever

        # Create LlamaIndex evaluator
        metric_instances = [AVAILABLE_RETRIEVAL_METRICS[m]() for m in metrics]
        self._evaluator = RetrieverEvaluator(
            metrics=metric_instances,
            retriever=retriever,
            node_postprocessors=node_postprocessors,
        )

    def evaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
    ) -> RetrievalEvalResult:
        """Evaluate retrieval for a single query.

        Args:
            query: Query string
            expected_ids: List of expected relevant document/node IDs
            expected_texts: Optional expected text content

        Returns:
            RetrievalEvalResult with computed metrics
        """
        return self._evaluator.evaluate(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
            mode=RetrievalEvalMode.TEXT,
        )

    async def aevaluate(
        self,
        query: str,
        expected_ids: List[str],
        expected_texts: Optional[List[str]] = None,
    ) -> RetrievalEvalResult:
        """Async evaluate retrieval for a single query.

        Args:
            query: Query string
            expected_ids: List of expected relevant document/node IDs
            expected_texts: Optional expected text content

        Returns:
            RetrievalEvalResult with computed metrics
        """
        return await self._evaluator.aevaluate(
            query=query,
            expected_ids=expected_ids,
            expected_texts=expected_texts,
            mode=RetrievalEvalMode.TEXT,
        )

    def evaluate_batch(
        self,
        queries: List[str],
        expected_ids_list: List[List[str]],
        expected_texts_list: Optional[List[List[str]]] = None,
        workers: int = 2,
        show_progress: bool = False,
    ) -> List[RetrievalEvalResult]:
        """Evaluate retrieval for multiple queries.

        Args:
            queries: List of query strings
            expected_ids_list: List of expected IDs for each query
            expected_texts_list: Optional expected texts for each query
            workers: Number of parallel workers
            show_progress: Show progress bar

        Returns:
            List of RetrievalEvalResult objects
        """
        # Build dataset for batch evaluation
        from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset

        queries_dict = {str(i): q for i, q in enumerate(queries)}
        relevant_docs = {str(i): ids for i, ids in enumerate(expected_ids_list)}
        corpus = {}  # Not needed for retrieval eval

        dataset = EmbeddingQAFinetuneDataset(
            queries=queries_dict,
            corpus=corpus,
            relevant_docs=relevant_docs,
        )

        # Run async evaluation
        return asyncio.run(
            self._evaluator.aevaluate_dataset(
                dataset=dataset,
                workers=workers,
                show_progress=show_progress,
            )
        )

    def evaluate_dataset(
        self,
        dataset: EvalDataset,
        workers: int = 2,
        show_progress: bool = False,
    ) -> List[RetrievalEvalResult]:
        """Evaluate retrieval using EvalDataset.

        Args:
            dataset: EvalDataset containing queries and expected IDs
            workers: Number of parallel workers
            show_progress: Show progress bar

        Returns:
            List of RetrievalEvalResult objects
        """
        return self.evaluate_batch(
            queries=dataset.get_queries(),
            expected_ids_list=dataset.get_expected_ids(),
            workers=workers,
            show_progress=show_progress,
        )

    def get_metrics_summary(
        self,
        results: List[RetrievalEvalResult],
    ) -> Dict[str, float]:
        """Compute average metrics from results.

        Args:
            results: List of RetrievalEvalResult objects

        Returns:
            Dictionary of average metric values
        """
        if not results:
            return {}

        summary = {}
        for metric in self.metrics:
            values = [r.metric_vals_dict.get(metric, 0.0) for r in results]
            summary[metric] = sum(values) / len(values)

        return summary

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metric names."""
        return list(AVAILABLE_RETRIEVAL_METRICS.keys())


def get_retrieval_results_df(
    results: List[RetrievalEvalResult],
) -> Any:
    """Convert retrieval results to pandas DataFrame.

    Args:
        results: List of RetrievalEvalResult objects

    Returns:
        pandas DataFrame with query and metrics columns
    """
    from llama_index.core.evaluation import get_retrieval_results_df as _get_df
    return _get_df(results)