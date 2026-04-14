"""Response evaluation module"""

from typing import List, Dict, Any, Optional, Sequence

from llama_index.core.llms.llm import LLM
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    BatchEvalRunner,
    EvaluationResult,
)


# Available response evaluators
AVAILABLE_RESPONSE_EVALUATORS = {
    "faithfulness": FaithfulnessEvaluator,
    "relevancy": RelevancyEvaluator,
    "correctness": CorrectnessEvaluator,
    "answer_relevancy": AnswerRelevancyEvaluator,
    "context_relevancy": ContextRelevancyEvaluator,
}


class ResponseEvaluator:
    """Response evaluator for measuring answer quality.

    Supports multiple evaluation types:
    - faithfulness: Whether the answer is grounded in retrieved context
    - relevancy: Whether the answer is relevant to the query
    - correctness: Whether the answer matches a reference answer
    - answer_relevancy: How well the answer addresses the query
    - context_relevancy: How relevant the context is to the query

    Args:
        llm: LLM instance for evaluation (used for LLM-based evaluators)
        evaluators: List of evaluator names to use
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        evaluators: List[str] = ["faithfulness", "relevancy"],
    ):
        """Initialize response evaluator.

        Args:
            llm: LLM for evaluation (required for most evaluators)
            evaluators: List of evaluator names
        """
        # Validate evaluators
        for eval_name in evaluators:
            if eval_name not in AVAILABLE_RESPONSE_EVALUATORS:
                raise ValueError(
                    f"Unknown evaluator: {eval_name}. "
                    f"Available: {list(AVAILABLE_RESPONSE_EVALUATORS.keys())}"
                )

        self.llm = llm
        self.evaluator_names = evaluators

        # Create evaluator instances
        self.evaluators: Dict[str, Any] = {}
        for name in evaluators:
            self.evaluators[name] = AVAILABLE_RESPONSE_EVALUATORS[name](llm=llm)

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate a single response.

        Args:
            query: Query string
            response: Generated response string
            contexts: List of context strings (retrieved documents)
            reference: Optional reference answer for correctness evaluation

        Returns:
            Dictionary of evaluator name -> EvaluationResult
        """
        results = {}

        for name, evaluator in self.evaluators.items():
            if name == "correctness" and reference is None:
                # Skip correctness if no reference provided
                continue

            eval_kwargs = {}
            if name == "correctness":
                eval_kwargs["reference"] = reference

            result = evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                **eval_kwargs,
            )
            results[name] = result

        return results

    async def aevaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, EvaluationResult]:
        """Async evaluate a single response.

        Args:
            query: Query string
            response: Generated response string
            contexts: List of context strings
            reference: Optional reference answer

        Returns:
            Dictionary of evaluator name -> EvaluationResult
        """
        results = {}

        for name, evaluator in self.evaluators.items():
            if name == "correctness" and reference is None:
                continue

            eval_kwargs = {}
            if name == "correctness":
                eval_kwargs["reference"] = reference

            result = await evaluator.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
                **eval_kwargs,
            )
            results[name] = result

        return results

    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
        workers: int = 2,
        show_progress: bool = False,
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate multiple responses in batch.

        Args:
            queries: List of query strings
            responses: List of response strings
            contexts_list: List of context lists for each query
            references: Optional list of reference answers
            workers: Number of parallel workers
            show_progress: Show progress bar

        Returns:
            Dictionary of evaluator name -> List of EvaluationResult
        """
        # Use BatchEvalRunner for parallel evaluation
        runner = BatchEvalRunner(
            evaluators=self.evaluators,
            workers=workers,
            show_progress=show_progress,
        )

        # Prepare eval_kwargs for correctness evaluator
        eval_kwargs = {}
        if "correctness" in self.evaluator_names and references:
            eval_kwargs["correctness"] = {"reference": references}

        return runner.evaluate_response_strs(
            queries=queries,
            response_strs=responses,
            contexts_list=contexts_list,
            **eval_kwargs,
        )

    async def aevaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        references: Optional[List[str]] = None,
        workers: int = 2,
        show_progress: bool = False,
    ) -> Dict[str, List[EvaluationResult]]:
        """Async evaluate multiple responses in batch.

        Args:
            queries: List of query strings
            responses: List of response strings
            contexts_list: List of context lists
            references: Optional reference answers
            workers: Number of parallel workers
            show_progress: Show progress bar

        Returns:
            Dictionary of evaluator name -> List of EvaluationResult
        """
        runner = BatchEvalRunner(
            evaluators=self.evaluators,
            workers=workers,
            show_progress=show_progress,
        )

        eval_kwargs = {}
        if "correctness" in self.evaluator_names and references:
            eval_kwargs["correctness"] = {"reference": references}

        return await runner.aevaluate_response_strs(
            queries=queries,
            response_strs=responses,
            contexts_list=contexts_list,
            **eval_kwargs,
        )

    def get_metrics_summary(
        self,
        results: Dict[str, List[EvaluationResult]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics from batch results.

        Args:
            results: Dictionary of evaluator name -> List of EvaluationResult

        Returns:
            Dictionary with per-evaluator statistics:
            {
                "faithfulness": {"mean": 0.85, "passing_rate": 0.90},
                "relevancy": {"mean": 0.72, "passing_rate": 0.75},
            }
        """
        summary = {}

        for name, eval_results in results.items():
            if not eval_results:
                continue

            scores = [r.score for r in eval_results if r.score is not None]
            passing = [r.passing for r in eval_results if r.passing is not None]

            summary[name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "passing_rate": sum(passing) / len(passing) if passing else 0.0,
                "count": len(eval_results),
            }

        return summary

    @classmethod
    def get_available_evaluators(cls) -> List[str]:
        """Get list of available evaluator names."""
        return list(AVAILABLE_RESPONSE_EVALUATORS.keys())


def format_evaluation_result(result: EvaluationResult) -> Dict[str, Any]:
    """Format EvaluationResult for display/export.

    Args:
        result: EvaluationResult object

    Returns:
        Dictionary with score, passing, feedback
    """
    return {
        "score": result.score,
        "passing": result.passing,
        "feedback": result.feedback,
        "invalid_result": result.invalid_result,
        "invalid_reason": result.invalid_reason,
    }