"""Document Cleaner - Main module for cleaning issue/ticket documents."""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_index.core import Document
from llama_index.core.llms import LLM

from ..config.settings import CustomOpenAILLM
from .cleaner_config import (
    CleanedDocument,
    CleanerConfig,
    DocumentMetadata,
    QualityCheckResult,
)
from .rule_extractor import RuleExtractor
from .llm_extractor import LLMExtractor
from .quality_checker import QualityChecker
from .loaders import DocumentLoader


logger = logging.getLogger(__name__)


class DocumentCleaner:
    """文档清理器 - 工单/问题单文档结构化处理

    处理流程:
    1. 规则提取层: 提取确定性信息(错误码、日志、环境、服务组件)
    2. LLM结构提取: 从文档中识别三要素
    3. 质量门禁: 检查完整性、一致性
    4. 生成结构化输出

    Example:
        >>> cleaner = DocumentCleaner.from_env()
        >>> result = cleaner.clean(document)
        >>> if result:
        >>>     result.save_to_file("output/structured_issue.md")
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        config: Optional[CleanerConfig] = None,
    ):
        """Initialize document cleaner.

        Args:
            llm: LLM instance for semantic extraction (supports OpenAI-compatible APIs)
            config: Cleaner configuration
        """
        self.config = config or CleanerConfig()
        self._llm = llm or self._create_default_llm()

        # Initialize components
        self._rule_extractor = RuleExtractor(self.config)
        self._llm_extractor = LLMExtractor(self._llm, self.config)
        self._quality_checker = QualityChecker(self._llm, self.config)

        # Statistics
        self._stats = {
            "total_processed": 0,
            "passed": 0,
            "rejected": 0,
            "errors": 0,
        }

    def _create_default_llm(self) -> LLM:
        """Create default LLM from environment (支持OpenAI兼容API)."""
        # Try to get config from RAGConfig if available
        try:
            from ..config.settings import RAGConfig
            rag_config = RAGConfig.from_env()
            llm_kwargs = {
                "model": rag_config.llm.model,
                "api_key": rag_config.llm.api_key,
                "temperature": rag_config.llm.temperature,
                "context_window": 128000,
                "is_chat_model": True,
            }
            if rag_config.llm.max_tokens:
                llm_kwargs["max_tokens"] = rag_config.llm.max_tokens
            if rag_config.llm.base_url:
                llm_kwargs["api_base"] = rag_config.llm.base_url
            return CustomOpenAILLM(**llm_kwargs)
        except Exception:
            # Fallback to default with CleanerConfig
            llm_kwargs = {
                "model": self.config.llm_model,
                "temperature": self.config.llm_temperature,
                "context_window": 128000,
                "is_chat_model": True,
            }
            if self.config.llm_max_tokens:
                llm_kwargs["max_tokens"] = self.config.llm_max_tokens
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
            if self.config.llm_base_url:
                llm_kwargs["api_base"] = self.config.llm_base_url
            return CustomOpenAILLM(**llm_kwargs)

    def clean(self, document: Document) -> Optional[CleanedDocument]:
        """Clean a single document.

        Args:
            document: Document to clean

        Returns:
            CleanedDocument if quality check passed, None if rejected
        """
        self._stats["total_processed"] += 1

        try:
            # Step 1: 规则提取 (快速、低成本)
            logger.debug("Step 1: Rule extraction...")
            rule_result = self._rule_extractor.extract(document.text)

            # Step 2: LLM结构提取 (语义理解)
            logger.debug("Step 2: LLM structure extraction...")
            structure = self._llm_extractor.extract_structure(
                document.text, hints=rule_result
            )

            # Step 3: 质量检查
            logger.debug("Step 3: Quality check...")
            quality = self._quality_checker.check(document.text, structure)

            if self._quality_checker.should_reject(quality):
                self._stats["rejected"] += 1
                logger.info(
                    f"Document rejected: {quality.rejection_reason} "
                    f"(source: {document.metadata.get('source_file', 'unknown')})"
                )
                return None

            # Step 4: 组装输出
            logger.debug("Step 4: Assembling output...")
            self._stats["passed"] += 1

            metadata = DocumentMetadata(
                error_codes=rule_result.error_codes,
                log_patterns=rule_result.log_patterns,
                environment=rule_result.environment,
                service_components=rule_result.service_components,
                keywords=rule_result.keywords,
                confidence_score=structure.confidence_score,
            )

            # Try to extract original title from document
            original_title = self._extract_title(document.text)

            return CleanedDocument(
                source_file=document.metadata.get("source_file", ""),
                original_title=original_title,
                problem=structure.problem,
                cause=structure.cause,
                solution=structure.solution,
                metadata=metadata,
                quality=quality,
                original_text=document.text if self.config.include_original_text else None,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Document cleaning failed: {e}")
            return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from document text."""
        import re
        # Look for markdown heading
        match = re.match(r'^#\s+(.+)$', text)
        if match:
            return match.group(1).strip()
        # Look for first significant line
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                return line[:50]  # First 50 chars as title
        return None

    def clean_documents(
        self,
        documents: List[Document]
    ) -> List[CleanedDocument]:
        """Clean multiple documents.

        Args:
            documents: List of documents to clean

        Returns:
            List of cleaned documents (only those that passed quality check)
        """
        results = []
        for doc in documents:
            result = self.clean(doc)
            if result:
                results.append(result)
        return results

    def clean_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
    ) -> List[CleanedDocument]:
        """Clean all documents in a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            file_types: File types to process (default: .md, .txt)

        Returns:
            List of cleaned documents
        """
        file_types = file_types or ['.md', '.txt']
        loader = DocumentLoader()

        # Load documents
        documents = loader.load_directory(
            directory,
            recursive=recursive,
        )

        # Filter by file type
        filtered = [
            doc for doc in documents
            if Path(doc.metadata.get("source_file", "")).suffix.lower() in file_types
        ]

        logger.info(f"Loaded {len(filtered)} documents from {directory}")
        return self.clean_documents(filtered)

    def save_results(
        self,
        results: List[CleanedDocument],
        output_dir: str,
        filename_prefix: Optional[str] = None,
    ) -> List[str]:
        """Save cleaned documents to output directory.

        Args:
            results: List of cleaned documents
            output_dir: Output directory path
            filename_prefix: Optional prefix for filenames

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for result in results:
            # Generate filename
            source_name = Path(result.source_file).stem
            if filename_prefix:
                filename = f"{filename_prefix}_{source_name}_cleaned.md"
            else:
                filename = f"{source_name}_cleaned.md"

            filepath = output_path / filename
            saved_path = result.save_to_file(str(filepath))
            saved_paths.append(saved_path)

        return saved_paths

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_processed": 0,
            "passed": 0,
            "rejected": 0,
            "errors": 0,
        }

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "DocumentCleaner":
        """Create DocumentCleaner from environment configuration.

        Args:
            env_file: Path to .env file (default: ".env" in current directory)

        Returns:
            DocumentCleaner instance
        """
        try:
            from ..config.settings import RAGConfig
            config = RAGConfig.from_env(env_file)

            llm_kwargs = {
                "model": config.llm.model,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "context_window": 128000,
                "is_chat_model": True,
            }
            if config.llm.max_tokens:
                llm_kwargs["max_tokens"] = config.llm.max_tokens
            if config.llm.base_url:
                llm_kwargs["api_base"] = config.llm.base_url

            llm = CustomOpenAILLM(**llm_kwargs)

            cleaner_config = CleanerConfig(
                llm_model=config.llm.model,
                llm_api_key=config.llm.api_key,
                llm_base_url=config.llm.base_url,
                llm_temperature=config.llm.temperature,
                llm_max_tokens=config.llm.max_tokens,
            )

            return cls(llm=llm, config=cleaner_config)

        except Exception as e:
            logger.warning(f"Failed to load config from env: {e}, using defaults")
            return cls()


def main():
    """CLI entry point for document cleaner."""
    parser = argparse.ArgumentParser(
        description="Clean and structure issue/ticket documents for RAG"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory or file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env config file (default: .env)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process subdirectories"
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=[".md", ".txt"],
        help="File types to process (default: .md .txt)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show processing statistics after completion"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create cleaner
    cleaner = DocumentCleaner.from_env(args.env_file)

    # Process
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        loader = DocumentLoader()
        documents = loader.load_file(str(input_path))
        results = cleaner.clean_documents(documents)
    else:
        # Directory
        results = cleaner.clean_directory(
            str(input_path),
            recursive=args.recursive,
            file_types=args.file_types,
        )

    # Save results
    saved_paths = cleaner.save_results(results, args.output)

    # Output summary
    print(f"Processed: {len(results)} documents passed quality check")
    print(f"Saved to: {args.output}")

    if args.stats:
        stats = cleaner.get_stats()
        print(f"\nStatistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Passed: {stats['passed']}")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()