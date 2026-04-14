"""RAG evaluation runner - combines retrieval and response evaluation"""

import time
from typing import List, Dict, Any, Optional

from pydantic import BaseModel

from llama_index.core.llms.llm import LLM
from llama_index.core.schema import NodeWithScore

from ..pipeline.rag_pipeline import RAGPipeline
from .retrieval import RetrievalEvaluator, RetrievalEvalResult
from .response import ResponseEvaluator, EvaluationResult
from .dataset import EvalDataset, EvalItem


class EvalResultItem(BaseModel):
    """Result for a single evaluation item.

    Attributes:
        query: Original query
        response: Generated response
        retrieval_metrics: Dictionary of retrieval metric values
        response_metrics: Dictionary of response metric values
        source_count: Number of sources retrieved
        elapsed_time: Time taken for query
    """

    query: str
    response: str
    retrieval_metrics: Dict[str, float]
    response_metrics: Dict[str, Any]
    source_count: int
    elapsed_time: float


class RAGEvalResults(BaseModel):
    """Complete evaluation results.

    Attributes:
        items: List of EvalResultItem
        retrieval_summary: Average retrieval metrics
        response_summary: Average response metrics
        total_time: Total evaluation time
    """

    items: List[EvalResultItem]
    retrieval_summary: Dict[str, float]
    response_summary: Dict[str, Dict[str, float]]
    total_time: float

    def save(self, path: str) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path
        """
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def get_summary_text(self) -> str:
        """Get summary as formatted text.

        Returns:
            Formatted summary string
        """
        lines = ["=== RAG Evaluation Results ===", ""]

        # Retrieval summary
        lines.append("## Retrieval Metrics")
        for metric, value in self.retrieval_summary.items():
            lines.append(f"  - {metric}: {value:.3f}")
        lines.append("")

        # Response summary
        lines.append("## Response Metrics")
        for evaluator, stats in self.response_summary.items():
            lines.append(f"  - {evaluator}:")
            lines.append(f"      mean: {stats.get('mean', 0):.3f}")
            lines.append(f"      passing_rate: {stats.get('passing_rate', 0):.3f}")
        lines.append("")

        lines.append(f"Total queries: {len(self.items)}")
        lines.append(f"Total time: {self.total_time:.2f}s")

        return "\n".join(lines)


class RAGEvalRunner:
    """RAG evaluation runner combining retrieval and response evaluation.

    Provides end-to-end evaluation of RAG pipeline:
    1. Run queries through the pipeline
    2. Evaluate retrieval (hit_rate, mrr, precision, recall, etc.)
    3. Evaluate response (faithfulness, relevancy, correctness, etc.)
    4. Aggregate results and compute summaries

    Args:
        pipeline: RAGPipeline instance to evaluate
        llm: LLM for response evaluation
        retrieval_metrics: List of retrieval metric names
        response_metrics: List of response evaluator names
        top_k: Number of documents to retrieve
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        llm: Optional[LLM] = None,
        retrieval_metrics: List[str] = ["hit_rate", "mrr", "precision", "recall"],
        response_metrics: List[str] = ["faithfulness", "relevancy"],
        top_k: int = 10,
    ):
        """Initialize RAG evaluation runner.

        Args:
            pipeline: RAGPipeline to evaluate
            llm: LLM for response evaluation (defaults to pipeline's LLM)
            retrieval_metrics: Retrieval metrics to compute
            response_metrics: Response evaluators to use
            top_k: Number of results to retrieve
        """
        self.pipeline = pipeline
        self.llm = llm or pipeline._llm
        self.top_k = top_k

        # Get retriever from pipeline
        self.retriever = pipeline._hybrid_retriever._vector_retriever

        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator(
            retriever=self.retriever,
            metrics=retrieval_metrics,
        )

        self.response_evaluator = ResponseEvaluator(
            llm=self.llm,
            evaluators=response_metrics,
        )

    def run_single(
        self,
        item: EvalItem,
    ) -> EvalResultItem:
        """Run evaluation for a single item.

        Args:
            item: EvalItem with query and expected IDs

        Returns:
            EvalResultItem with all metrics
        """
        start_time = time.time()

        # 1. Run query through pipeline
        result = self.pipeline.query(item.query, top_k=self.top_k)

        elapsed = time.time() - start_time

        # Extract response and contexts
        response = result["response"]
        source_nodes: List[NodeWithScore] = result["source_nodes"]
        contexts = [node.node.text for node in source_nodes]

        # 2. Evaluate retrieval
        retrieval_eval = self.retrieval_evaluator.evaluate(
            query=item.query,
            expected_ids=item.expected_ids,
            expected_texts=item.expected_texts,
        )

        # 3. Evaluate response
        response_eval = self.response_evaluator.evaluate(
            query=item.query,
            response=response,
            contexts=contexts,
            reference=item.reference_answer,
        )

        # Format response metrics
        response_metrics = {}
        for name, eval_result in response_eval.items():
            response_metrics[name] = {
                "score": eval_result.score,
                "passing": eval_result.passing,
                "feedback": eval_result.feedback,
            }

        return EvalResultItem(
            query=item.query,
            response=response,
            retrieval_metrics=retrieval_eval.metric_vals_dict,
            response_metrics=response_metrics,
            source_count=len(source_nodes),
            elapsed_time=elapsed,
        )

    def run_evaluation(
        self,
        dataset: EvalDataset,
        show_progress: bool = True,
    ) -> RAGEvalResults:
        """Run full evaluation on dataset.

        Args:
            dataset: EvalDataset with queries and expected IDs
            show_progress: Show progress during evaluation

        Returns:
            RAGEvalResults with all metrics and summaries
        """
        start_time = time.time()
        results = []

        for i, item in enumerate(dataset):
            if show_progress:
                print(f"Evaluating query {i+1}/{len(dataset)}: {item.query[:50]}...")

            result_item = self.run_single(item)
            results.append(result_item)

        total_time = time.time() - start_time

        # Compute summaries
        retrieval_summary = self.retrieval_evaluator.get_metrics_summary(
            [self._create_retrieval_result(r) for r in results]
        )

        # Build response results for summary
        response_results = {}
        for name in self.response_evaluator.evaluator_names:
            eval_results = []
            for r in results:
                if name in r.response_metrics:
                    # Create mock EvaluationResult for summary calculation
                    eval_result = EvaluationResult(
                        query=r.query,
                        score=r.response_metrics[name].get("score"),
                        passing=r.response_metrics[name].get("passing"),
                    )
                    eval_results.append(eval_result)
            response_results[name] = eval_results

        response_summary = self.response_evaluator.get_metrics_summary(response_results)

        return RAGEvalResults(
            items=results,
            retrieval_summary=retrieval_summary,
            response_summary=response_summary,
            total_time=total_time,
        )

    def _create_retrieval_result(self, item: EvalResultItem) -> RetrievalEvalResult:
        """Create RetrievalEvalResult from EvalResultItem for summary calculation."""
        from llama_index.core.evaluation.retrieval.metrics_base import RetrievalMetricResult

        metric_dict = {
            k: RetrievalMetricResult(score=v)
            for k, v in item.retrieval_metrics.items()
        }

        return RetrievalEvalResult(
            query=item.query,
            expected_ids=[],  # Not needed for summary
            retrieved_ids=[],  # Not needed for summary
            retrieved_texts=[],  # Not needed for summary
            metric_dict=metric_dict,
        )

    def quick_eval(
        self,
        queries: List[str],
        expected_ids_list: List[List[str]],
        references: Optional[List[str]] = None,
    ) -> RAGEvalResults:
        """Quick evaluation with minimal setup.

        Args:
            queries: List of queries
            expected_ids_list: Expected IDs for each query
            references: Optional reference answers

        Returns:
            RAGEvalResults
        """
        items = []
        for i, query in enumerate(queries):
            item = EvalItem(
                query=query,
                expected_ids=expected_ids_list[i],
                reference_answer=references[i] if references else None,
            )
            items.append(item)

        dataset = EvalDataset(items=items)
        return self.run_evaluation(dataset)

    def get_available_metrics(self) -> Dict[str, List[str]]:
        """Get all available metrics for retrieval and response.

        Returns:
            Dictionary with retrieval and response metric lists
        """
        return {
            "retrieval": RetrievalEvaluator.get_available_metrics(),
            "response": ResponseEvaluator.get_available_evaluators(),
        }