"""Chunking evaluation for comparing different splitting strategies"""

import random
import statistics
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from llama_index.core.schema import TextNode, Document
from llama_index.core.llms.llm import LLM

from ..ingestion.splitters import TextSplitter, ChineseTextSplitter


class ChunkStatistics(BaseModel):
    """Statistics about chunked documents.

    Attributes:
        total_chunks: Total number of chunks
        total_documents: Number of source documents
        total_characters: Total characters across all chunks
        avg_chunk_length: Average chunk length in characters
        min_chunk_length: Minimum chunk length
        max_chunk_length: Maximum chunk length
        std_chunk_length: Standard deviation of chunk lengths
        median_chunk_length: Median chunk length
        length_distribution: Distribution of chunk lengths by size buckets
        chunks_per_doc_avg: Average chunks per document
    """

    total_chunks: int
    total_documents: int
    total_characters: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    std_chunk_length: float
    median_chunk_length: float
    length_distribution: Dict[str, int]
    chunks_per_doc_avg: float


class ChunkQualityResult(BaseModel):
    """Quality assessment of chunks.

    Attributes:
        semantic_completeness: Score for semantic completeness (0-1)
        boundary_quality: Score for boundary quality (0-1)
        info_density: Score for information density (0-1)
        samples_evaluated: Number of samples evaluated
        issues_found: List of identified issues
    """

    semantic_completeness: float
    boundary_quality: float
    info_density: float
    samples_evaluated: int
    issues_found: List[str]


class ChunkingEvalResult(BaseModel):
    """Complete evaluation result for a chunking configuration.

    Attributes:
        splitter_type: Type of splitter used
        chunk_size: Configured chunk size
        chunk_overlap: Configured overlap
        statistics: Chunk statistics
        quality: Optional quality assessment
        retrieval_metrics: Optional retrieval impact metrics
    """

    splitter_type: str
    chunk_size: int
    chunk_overlap: int
    statistics: ChunkStatistics
    quality: Optional[ChunkQualityResult] = None
    retrieval_metrics: Optional[Dict[str, float]] = None


class ChunkingCompareResults(BaseModel):
    """Results comparing multiple chunking configurations.

    Attributes:
        results: List of ChunkingEvalResult for each configuration
        best_config: Best configuration based on retrieval metrics
        comparison_table: Summary comparison data
    """

    results: List[ChunkingEvalResult]
    best_config: Optional[str] = None
    comparison_table: Dict[str, Dict[str, Any]] = {}

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def get_summary_text(self) -> str:
        """Get summary as formatted text."""
        lines = ["=== Chunking Evaluation Results ===", ""]

        for result in self.results:
            config_name = f"{result.splitter_type}:{result.chunk_size}:{result.chunk_overlap}"
            lines.append(f"## Configuration: {config_name}")
            lines.append(f"  Total chunks: {result.statistics.total_chunks}")
            lines.append(f"  Avg length: {result.statistics.avg_chunk_length:.1f}")
            lines.append(f"  Min/Max length: {result.statistics.min_chunk_length}/{result.statistics.max_chunk_length}")
            lines.append(f"  Std deviation: {result.statistics.std_chunk_length:.1f}")

            if result.quality:
                lines.append(f"  Semantic completeness: {result.quality.semantic_completeness:.2f}")
                lines.append(f"  Boundary quality: {result.quality.boundary_quality:.2f}")

            if result.retrieval_metrics:
                lines.append(f"  Retrieval metrics:")
                for metric, value in result.retrieval_metrics.items():
                    lines.append(f"    - {metric}: {value:.3f}")

            lines.append("")

        if self.best_config:
            lines.append(f"Best configuration: {self.best_config}")

        return "\n".join(lines)


class ChunkingEvaluator:
    """Evaluator for comparing chunking strategies.

    Provides evaluation at three levels:
    1. Statistics: Basic chunk metrics
    2. Quality: LLM-based quality assessment
    3. Retrieval: Impact on retrieval performance
    """

    # Quality evaluation prompts
    SEMANTIC_PROMPT = """评估以下文本块的语义完整性。

评分标准：
- 1.0: 完整的语义单元，意思表达完整
- 0.7: 基本完整，但缺少部分上下文
- 0.5: 部分完整，需要更多上下文才能理解
- 0.3: 不完整，语义被切断
- 0.0: 严重不完整，无法理解

文本块内容：
{chunk_text}

请只返回一个数字评分（0-1之间的数值），不要包含其他内容。"""

    BOUNDARY_PROMPT = """评估以下文本块的边界质量。

评分标准：
- 1.0: 边界非常合理，在自然断点处切分
- 0.7: 边界基本合理
- 0.5: 边界一般，有轻微问题
- 0.3: 边界不太合理，在句子中间切断
- 0.0: 边界不合理，破坏了重要结构

检查要点：
- 是否在句子中间切断？
- 表格是否完整？
- 代码块是否完整？
- 重要信息是否被拆分？

文本块内容：
{chunk_text}

请只返回一个数字评分（0-1之间的数值），不要包含其他内容。"""

    def __init__(
        self,
        use_quality_eval: bool = False,
        llm: Optional[LLM] = None,
        quality_sample_size: int = 10,
    ):
        """Initialize chunking evaluator.

        Args:
            use_quality_eval: Whether to use LLM for quality evaluation
            llm: LLM instance for quality evaluation
            quality_sample_size: Number of chunks to sample for quality eval
        """
        self.use_quality_eval = use_quality_eval
        self.llm = llm
        self.quality_sample_size = quality_sample_size

    def evaluate_statistics(self, chunks: List[TextNode]) -> ChunkStatistics:
        """Calculate chunk statistics.

        Args:
            chunks: List of TextNode chunks

        Returns:
            ChunkStatistics object
        """
        if not chunks:
            return ChunkStatistics(
                total_chunks=0,
                total_documents=0,
                total_characters=0,
                avg_chunk_length=0,
                min_chunk_length=0,
                max_chunk_length=0,
                std_chunk_length=0,
                median_chunk_length=0,
                length_distribution={},
                chunks_per_doc_avg=0,
            )

        lengths = [len(chunk.text) for chunk in chunks]

        # Calculate distribution by size buckets
        buckets = {
            "<100": 0,
            "100-200": 0,
            "200-500": 0,
            "500-1000": 0,
            "1000-2000": 0,
            ">2000": 0,
        }
        for length in lengths:
            if length < 100:
                buckets["<100"] += 1
            elif length < 200:
                buckets["100-200"] += 1
            elif length < 500:
                buckets["200-500"] += 1
            elif length < 1000:
                buckets["500-1000"] += 1
            elif length < 2000:
                buckets["1000-2000"] += 1
            else:
                buckets[">2000"] += 1

        # Count unique documents
        doc_ids = set(chunk.ref_doc_id for chunk in chunks if chunk.ref_doc_id)
        total_docs = len(doc_ids) if doc_ids else 1

        return ChunkStatistics(
            total_chunks=len(chunks),
            total_documents=total_docs,
            total_characters=sum(lengths),
            avg_chunk_length=statistics.mean(lengths),
            min_chunk_length=min(lengths),
            max_chunk_length=max(lengths),
            std_chunk_length=statistics.stdev(lengths) if len(lengths) > 1 else 0,
            median_chunk_length=statistics.median(lengths),
            length_distribution=buckets,
            chunks_per_doc_avg=len(chunks) / total_docs if total_docs > 0 else 0,
        )

    def evaluate_quality(
        self,
        chunks: List[TextNode],
        sample_size: Optional[int] = None,
    ) -> ChunkQualityResult:
        """Evaluate chunk quality using LLM.

        Args:
            chunks: List of TextNode chunks
            sample_size: Number of samples to evaluate

        Returns:
            ChunkQualityResult with quality scores
        """
        if not self.llm:
            raise ValueError("LLM required for quality evaluation")

        sample_size = sample_size or self.quality_sample_size

        # Sample chunks for evaluation
        if len(chunks) <= sample_size:
            samples = chunks
        else:
            # Sample chunks with varied lengths
            sorted_chunks = sorted(chunks, key=lambda c: len(c.text))
            step = len(sorted_chunks) // sample_size
            samples = [sorted_chunks[i * step] for i in range(sample_size)]

        semantic_scores = []
        boundary_scores = []
        issues = []

        for chunk in samples:
            # Evaluate semantic completeness
            semantic_response = self.llm.complete(
                self.SEMANTIC_PROMPT.format(chunk_text=chunk.text[:500])
            )
            semantic_score = 0.5  # Default value
            try:
                response_text = semantic_response.text.strip()
                # Extract last number from response (handles reasoning models)
                import re
                numbers = re.findall(r'[0-9]*\.?[0-9]+', response_text)
                if numbers:
                    semantic_score = float(numbers[-1])
                    semantic_score = max(0, min(1, semantic_score))
            except (ValueError, AttributeError):
                semantic_score = 0.5
            semantic_scores.append(semantic_score)

            # Evaluate boundary quality
            boundary_response = self.llm.complete(
                self.BOUNDARY_PROMPT.format(chunk_text=chunk.text[:500])
            )
            boundary_score = 0.5  # Default value
            try:
                response_text = boundary_response.text.strip()
                import re
                numbers = re.findall(r'[0-9]*\.?[0-9]+', response_text)
                if numbers:
                    boundary_score = float(numbers[-1])
                    boundary_score = max(0, min(1, boundary_score))
            except (ValueError, AttributeError):
                boundary_score = 0.5
            boundary_scores.append(boundary_score)

            # Track issues for low-scoring chunks
            if semantic_score < 0.5:
                issues.append(f"Low semantic completeness in chunk: {chunk.text[:50]}...")
            if boundary_score < 0.5:
                issues.append(f"Boundary issue in chunk: {chunk.text[:50]}...")

        # Calculate info density (ratio of non-whitespace characters)
        total_chars = sum(len(c.text) for c in samples)
        non_whitespace = sum(len(c.text.replace(" ", "").replace("\n", "")) for c in samples)
        info_density = non_whitespace / total_chars if total_chars > 0 else 0

        return ChunkQualityResult(
            semantic_completeness=statistics.mean(semantic_scores) if semantic_scores else 0,
            boundary_quality=statistics.mean(boundary_scores) if boundary_scores else 0,
            info_density=info_density,
            samples_evaluated=len(samples),
            issues_found=issues[:10],  # Limit to 10 issues
        )

    def evaluate_splitter_config(
        self,
        documents: List[Document],
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        embed_model: Optional[Any] = None,
    ) -> ChunkingEvalResult:
        """Evaluate a specific splitter configuration.

        Args:
            documents: Documents to chunk
            splitter_type: Type of splitter ("sentence", "token", "semantic", "chinese")
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            embed_model: Embedding model (required for semantic splitter)

        Returns:
            ChunkingEvalResult with evaluation results
        """
        # Create splitter
        if splitter_type == "chinese":
            splitter = ChineseTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            splitter = TextSplitter(
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_model=embed_model if splitter_type == "semantic" else None,
            )

        # Split documents
        chunks = splitter.split_documents(documents)

        # Calculate statistics
        stats = self.evaluate_statistics(chunks)

        # Evaluate quality if enabled
        quality = None
        if self.use_quality_eval and self.llm:
            quality = self.evaluate_quality(chunks)

        return ChunkingEvalResult(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            statistics=stats,
            quality=quality,
        )

    def compare_configs(
        self,
        documents: List[Document],
        configs: List[Dict[str, Any]],
        embed_model: Optional[Any] = None,
    ) -> ChunkingCompareResults:
        """Compare multiple splitter configurations.

        Args:
            documents: Documents to chunk
            configs: List of configurations, each with splitter_type, chunk_size, chunk_overlap
            embed_model: Embedding model for semantic splitter

        Returns:
            ChunkingCompareResults comparing all configurations
        """
        results = []

        for config in configs:
            result = self.evaluate_splitter_config(
                documents=documents,
                splitter_type=config.get("splitter_type", "sentence"),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                embed_model=embed_model,
            )
            results.append(result)

        # Build comparison table
        comparison_table = {}
        for result in results:
            config_name = f"{result.splitter_type}:{result.chunk_size}:{result.chunk_overlap}"
            comparison_table[config_name] = {
                "total_chunks": result.statistics.total_chunks,
                "avg_length": result.statistics.avg_chunk_length,
                "std_length": result.statistics.std_chunk_length,
            }
            if result.quality:
                comparison_table[config_name]["semantic_score"] = result.quality.semantic_completeness
                comparison_table[config_name]["boundary_score"] = result.quality.boundary_quality

        # Find best config based on retrieval metrics if available
        best_config = None
        configs_with_retrieval = [r for r in results if r.retrieval_metrics]
        if configs_with_retrieval:
            # Sort by hit_rate
            best = max(configs_with_retrieval, key=lambda r: r.retrieval_metrics.get("hit_rate", 0))
            best_config = f"{best.splitter_type}:{best.chunk_size}:{best.chunk_overlap}"

        return ChunkingCompareResults(
            results=results,
            best_config=best_config,
            comparison_table=comparison_table,
        )


def parse_config_string(config_str: str) -> Dict[str, Any]:
    """Parse a configuration string like 'sentence:512:50'.

    Args:
        config_str: Configuration string

    Returns:
        Dictionary with splitter_type, chunk_size, chunk_overlap
    """
    parts = config_str.split(":")
    if len(parts) >= 1:
        splitter_type = parts[0]
    else:
        splitter_type = "sentence"

    if len(parts) >= 2:
        try:
            chunk_size = int(parts[1])
        except ValueError:
            chunk_size = 512
    else:
        chunk_size = 512

    if len(parts) >= 3:
        try:
            chunk_overlap = int(parts[2])
        except ValueError:
            chunk_overlap = 50
    else:
        chunk_overlap = 50

    return {
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }