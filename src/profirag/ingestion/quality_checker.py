"""Quality checker for document validation."""

import logging
from typing import Optional, Set

from llama_index.core.llms import LLM

from .cleaner_config import (
    QualityCheckResult,
    StructureResult,
    CleanerConfig,
)
from .llm_extractor import LLMExtractor


logger = logging.getLogger(__name__)


class QualityChecker:
    """质量门禁检查器"""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        config: Optional[CleanerConfig] = None,
    ):
        self.config = config or CleanerConfig()
        self._llm_extractor = LLMExtractor(llm, config) if llm else None

    def check(
        self,
        original_text: str,
        structure: StructureResult
    ) -> QualityCheckResult:
        """执行质量检查"""

        issues: list = []

        # Level 1: 规则检查 (快速)
        # 1.1 检查文档长度有效性
        length_ok = self._check_document_length(original_text)
        if not length_ok:
            issues.append("文档内容过短，信息不足")

        # 1.2 检查是否有明确的解决方案
        has_solution = self._check_solution_exists(structure)
        if self.config.require_solution_steps and not has_solution:
            issues.append("缺少明确的解决方案")

        # 1.3 检查三要素是否有基本内容
        has_basic_content = self._check_basic_content(structure)
        if not has_basic_content:
            issues.append("三要素内容不完整")

        # Level 2: LLM语义检查 (深度) - 如果配置了LLM
        completeness_result = {"completeness_score": 0.7, "missing_elements": []}
        contradiction_result = {"has_contradictions": False, "description": "", "match_score": 0.7}

        if self._llm_extractor:
            # 2.1 检查三要素完整性
            completeness_result = self._llm_extractor.check_completeness(structure)
            if completeness_result.get("completeness_score", 0) < self.config.min_completeness_score:
                missing = completeness_result.get("missing_elements", [])
                issues.append(f"完整性不足: {', '.join(missing)}")

            # 2.2 检查信息一致性（矛盾检测）
            contradiction_result = self._llm_extractor.check_contradictions(structure)
            if contradiction_result.get("has_contradictions", False):
                desc = contradiction_result.get("description", "存在矛盾")
                issues.append(f"存在矛盾: {desc}")

        # 计算完整性分数
        completeness_score = completeness_result.get("completeness_score", 0.5)

        # 最终判断
        passed = len(issues) == 0 and completeness_score >= self.config.min_completeness_score

        return QualityCheckResult(
            passed=passed,
            has_solution=has_solution,
            no_contradictions=not contradiction_result.get("has_contradictions", False),
            completeness_score=completeness_score,
            issues=issues,
            rejection_reason=issues[0] if issues else None,
        )

    def _check_document_length(self, text: str) -> bool:
        """检查文档长度"""
        return len(text) >= self.config.min_document_length

    def _check_solution_exists(self, structure: StructureResult) -> bool:
        """规则检查：是否有可操作的解决方案"""
        solution = structure.solution

        # 必须有步骤或命令
        if len(solution.steps) > 0:
            # 检查步骤是否包含可操作内容
            steps_text = " ".join(solution.steps)
            if self._contains_actionable_content(steps_text):
                return True

        # 检查命令
        if len(solution.commands) > 0:
            return True

        # 检查排查步骤
        if len(solution.troubleshooting_steps) > 0:
            for step in solution.troubleshooting_steps:
                if step.command:
                    return True

        return False

    def _contains_actionable_content(self, text: str) -> bool:
        """检查是否包含可操作内容"""
        keywords = self.config.solution_keywords
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def _check_basic_content(self, structure: StructureResult) -> bool:
        """检查三要素是否有基本内容"""
        has_problem = bool(structure.problem.description)
        has_cause = bool(structure.cause.root_cause)
        has_solution = bool(structure.solution.steps) or bool(structure.solution.commands)

        # 至少需要有问题描述和解决方案
        return has_problem and has_solution

    def quick_check(self, text: str) -> bool:
        """快速质量检查（仅规则层）"""
        return len(text) >= self.config.min_document_length

    def should_reject(self, quality: QualityCheckResult) -> bool:
        """判断是否应该拒绝文档"""
        return not quality.passed

    def get_rejection_message(self, quality: QualityCheckResult) -> str:
        """获取拒绝原因描述"""
        if quality.rejection_reason:
            return quality.rejection_reason
        if not quality.has_solution:
            return "文档缺少可操作的解决方案"
        if not quality.no_contradictions:
            return "文档存在信息矛盾"
        if quality.completeness_score < self.config.min_completeness_score:
            return f"文档完整性不足(分数: {quality.completeness_score:.2f})"
        return "文档质量不合格"