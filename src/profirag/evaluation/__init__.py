"""ProfiRAG Evaluation Module

Provides evaluation capabilities for RAG systems:
- Retrieval evaluation: hit_rate, mrr, precision, recall, ndcg, ap
- Response evaluation: faithfulness, relevancy, correctness, answer_relevancy, context_relevancy
- Chunking evaluation: statistics, quality, retrieval impact comparison

Usage:
    from profirag.evaluation import (
        RetrievalEvaluator,
        ResponseEvaluator,
        RAGEvalRunner,
        EvalDataset,
        EvalItem,
        ChunkingEvaluator,
    )

    # Create evaluation dataset
    dataset = EvalDataset.from_json("./eval_data.json")

    # Run evaluation
    runner = RAGEvalRunner(pipeline, llm)
    results = runner.run_evaluation(dataset)

    # Get summary
    print(results.get_summary_text())

    # Generate dataset from documents
    from profirag.evaluation import create_dataset_from_documents
    dataset = create_dataset_from_documents("./documents", "./eval_data.json")

    # Generate dataset from existing pipeline
    from profirag.evaluation import create_dataset_from_pipeline
    dataset = create_dataset_from_pipeline(pipeline, "./eval_data.json")

    # Evaluate chunking strategies
    from profirag.evaluation import ChunkingEvaluator, parse_config_string
    evaluator = ChunkingEvaluator()
    configs = ["sentence:512:50", "chinese:512:50", "sentence:1024:100"]
    results = evaluator.compare_configs(documents, [parse_config_string(c) for c in configs])
"""

from .dataset import (
    EvalDataset,
    EvalItem,
    create_sample_dataset,
    create_dataset_from_nodes,
    create_dataset_from_documents,
    create_dataset_from_pipeline,
    generate_query_from_text,
    extract_keywords_from_text,
)
from .retrieval import (
    RetrievalEvaluator,
    RetrievalEvalResult,
    AVAILABLE_RETRIEVAL_METRICS,
    get_retrieval_results_df,
)
from .response import (
    ResponseEvaluator,
    EvaluationResult,
    AVAILABLE_RESPONSE_EVALUATORS,
    format_evaluation_result,
)
from .runner import (
    RAGEvalRunner,
    RAGEvalResults,
    EvalResultItem,
)
from .chunking import (
    ChunkingEvaluator,
    ChunkingEvalResult,
    ChunkingCompareResults,
    ChunkStatistics,
    ChunkQualityResult,
    parse_config_string,
)


__all__ = [
    # Dataset
    "EvalDataset",
    "EvalItem",
    "create_sample_dataset",
    "create_dataset_from_nodes",
    "create_dataset_from_documents",
    "create_dataset_from_pipeline",
    "generate_query_from_text",
    "extract_keywords_from_text",
    # Retrieval
    "RetrievalEvaluator",
    "RetrievalEvalResult",
    "AVAILABLE_RETRIEVAL_METRICS",
    "get_retrieval_results_df",
    # Response
    "ResponseEvaluator",
    "EvaluationResult",
    "AVAILABLE_RESPONSE_EVALUATORS",
    "format_evaluation_result",
    # Runner
    "RAGEvalRunner",
    "RAGEvalResults",
    "EvalResultItem",
    # Chunking
    "ChunkingEvaluator",
    "ChunkingEvalResult",
    "ChunkingCompareResults",
    "ChunkStatistics",
    "ChunkQualityResult",
    "parse_config_string",
]