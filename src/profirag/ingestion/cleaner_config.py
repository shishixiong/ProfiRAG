"""Configuration and data models for document cleaner."""

import re
from typing import List, Optional, Dict, Any, Set, Literal
from pydantic import BaseModel, Field
from pathlib import Path


# ============================================================================
# Data Models for Structured Document Output
# ============================================================================

class TroubleshootingStep(BaseModel):
    """排查步骤"""
    description: str = Field(default="", description="步骤描述")
    command: Optional[str] = Field(default=None, description="执行的命令")
    result: Optional[str] = Field(default=None, description="排查结果")


class ProblemElement(BaseModel):
    """问题现象"""
    description: str = Field(default="", description="问题现象的简洁描述")
    symptoms: List[str] = Field(default_factory=list, description="症状列表")
    affected_components: List[str] = Field(default_factory=list, description="受影响的组件")


class CauseAnalysis(BaseModel):
    """原因分析"""
    root_cause: str = Field(default="", description="根本原因")
    contributing_factors: List[str] = Field(default_factory=list, description="促成因素")


class Solution(BaseModel):
    """解决方案"""
    troubleshooting_steps: List[TroubleshootingStep] = Field(
        default_factory=list, description="排查步骤"
    )
    steps: List[str] = Field(default_factory=list, description="解决步骤")
    commands: List[str] = Field(default_factory=list, description="执行命令")
    verification: Optional[str] = Field(default=None, description="验证方法")


class DocumentMetadata(BaseModel):
    """提取的元数据"""
    error_codes: List[str] = Field(default_factory=list, description="错误码如GAUSS-00123")
    log_patterns: List[str] = Field(default_factory=list, description="日志特征")
    environment: Dict[str, str] = Field(default_factory=dict, description="环境信息")
    service_components: List[str] = Field(default_factory=list, description="相关服务组件")
    keywords: List[str] = Field(default_factory=list, description="问题分类关键词")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="提取置信度")


class QualityCheckResult(BaseModel):
    """质量检查结果"""
    passed: bool = Field(default=False, description="是否通过质量检查")
    has_solution: bool = Field(default=False, description="是否有明确的解决方案")
    no_contradictions: bool = Field(default=True, description="是否无矛盾")
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="三要素完整性分数")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    rejection_reason: Optional[str] = Field(default=None, description="拒绝原因")


class ImageInfo(BaseModel):
    """图片信息"""
    image_id: str = Field(default="", description="图片ID")
    original_path: str = Field(default="", description="原始图片路径")
    relative_path: Optional[str] = Field(default=None, description="相对输出文件的路径")
    description: Optional[str] = Field(default=None, description="图片描述(LLM生成)")
    alt_text: Optional[str] = Field(default=None, description="原始alt文本")
    surrounding_context: Optional[str] = Field(default=None, description="图片周围的文本上下文")
    section: Optional[str] = Field(default=None, description="图片所属章节(problem/cause/solution)")


class StructureResult(BaseModel):
    """LLM提取的结构结果"""
    problem: ProblemElement = Field(default_factory=ProblemElement)
    cause: CauseAnalysis = Field(default_factory=CauseAnalysis)
    solution: Solution = Field(default_factory=Solution)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RuleResult(BaseModel):
    """规则提取结果"""
    error_codes: List[str] = Field(default_factory=list)
    log_patterns: List[str] = Field(default_factory=list)
    environment: Dict[str, str] = Field(default_factory=dict)
    service_components: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class CleanedDocument(BaseModel):
    """清理后的结构化文档"""
    source_file: str = Field(default="", description="原始文件路径")
    original_title: Optional[str] = Field(default=None, description="原始标题")
    problem: ProblemElement = Field(default_factory=ProblemElement)
    cause: CauseAnalysis = Field(default_factory=CauseAnalysis)
    solution: Solution = Field(default_factory=Solution)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    quality: QualityCheckResult = Field(default_factory=QualityCheckResult)
    images: List[ImageInfo] = Field(default_factory=list, description="文档中的图片信息")
    original_text: Optional[str] = Field(default=None, description="原文内容")

    def to_markdown(self) -> str:
        """转换为结构化Markdown格式"""
        lines = []

        # 标题
        title = self.original_title or "故障排查文档"
        lines.append(f"# 故障排查文档 - {title}")
        lines.append("")
        lines.append(f"**来源文件**: {self.source_file}")
        lines.append(f"**置信度**: {self.metadata.confidence_score:.2f}")
        lines.append("")

        # 问题现象
        lines.append("## 问题现象")
        lines.append("")
        if self.problem.description:
            lines.append(f"**描述**: {self.problem.description}")
        if self.problem.symptoms:
            lines.append("**症状**:")
            for symptom in self.problem.symptoms:
                lines.append(f"- {symptom}")
        if self.problem.affected_components:
            lines.append("**受影响组件**: " + ", ".join(self.problem.affected_components))
        lines.append("")

        # 原因分析
        lines.append("## 原因分析")
        lines.append("")
        if self.cause.root_cause:
            lines.append(f"**根本原因**: {self.cause.root_cause}")
        if self.cause.contributing_factors:
            lines.append("**促成因素**:")
            for factor in self.cause.contributing_factors:
                lines.append(f"- {factor}")
        lines.append("")

        # 排查步骤
        if self.solution.troubleshooting_steps:
            lines.append("## 排查步骤")
            lines.append("")
            for i, step in enumerate(self.solution.troubleshooting_steps, 1):
                step_text = f"{i}. {step.description}"
                if step.command:
                    step_text += f" → `{step.command}`"
                if step.result:
                    step_text += f" → 结果: {step.result}"
                lines.append(step_text)
            lines.append("")

        # 解决方案
        lines.append("## 解决方案")
        lines.append("")
        if self.solution.steps:
            lines.append("**解决步骤**:")
            for i, step in enumerate(self.solution.steps, 1):
                lines.append(f"{i}. {step}")
        if self.solution.commands:
            lines.append("**执行命令**:")
            for cmd in self.solution.commands:
                lines.append(f"- `{cmd}`")
        if self.solution.verification:
            lines.append(f"**验证方法**: {self.solution.verification}")
        lines.append("")

        # 元数据
        lines.append("## 元数据")
        lines.append("")
        if self.metadata.error_codes:
            lines.append(f"- **错误码**: {', '.join(self.metadata.error_codes)}")
        if self.metadata.log_patterns:
            lines.append(f"- **日志特征**: {', '.join(self.metadata.log_patterns)}")
        if self.metadata.environment:
            env_str = ", ".join(f"{k}: {v}" for k, v in self.metadata.environment.items())
            lines.append(f"- **环境**: {env_str}")
        if self.metadata.service_components:
            lines.append(f"- **相关服务**: {', '.join(self.metadata.service_components)}")
        if self.metadata.keywords:
            lines.append(f"- **关键词**: {', '.join(self.metadata.keywords)}")
        lines.append("")

        # 质量信息
        lines.append("## 质量信息")
        lines.append("")
        lines.append(f"- **质量检查**: {'通过' if self.quality.passed else '未通过'}")
        lines.append(f"- **完整性分数**: {self.quality.completeness_score:.2f}")
        if self.quality.issues:
            lines.append("- **发现的问题**:")
            for issue in self.quality.issues:
                lines.append(f"  - {issue}")
        lines.append("")

        # 相关图片
        if self.images:
            lines.append("## 相关图片")
            lines.append("")
            for img in self.images:
                # 图片标题
                img_title = f"### {img.image_id}"
                if img.section:
                    img_title += f" ({img.section})"
                lines.append(img_title)
                lines.append("")
                # 图片引用
                if img.relative_path:
                    lines.append(f"![{img.alt_text or img.image_id}]({img.relative_path})")
                elif img.original_path:
                    lines.append(f"![{img.alt_text or img.image_id}]({img.original_path})")
                lines.append("")
                # 图片描述
                if img.description:
                    lines.append(f"**图片描述**: {img.description}")
                if img.surrounding_context:
                    lines.append(f"**上下文**: {img.surrounding_context[:200]}...")
                lines.append("")

        return "\n".join(lines)

    def save_to_file(self, output_path: str) -> str:
        """保存为Markdown文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return str(path)

    @classmethod
    def from_file(cls, file_path: str) -> "CleanedDocument":
        """从Markdown文件加载"""
        # TODO: 实现Markdown解析逻辑
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        # 简化实现：仅提取基本信息
        return cls(source_file=str(path), original_text=content)


# ============================================================================
# Configuration
# ============================================================================

class CleanerConfig(BaseModel):
    """文档清理器配置"""
    # 规则提取配置
    min_document_length: int = Field(default=100, description="最小文档长度")
    max_document_length: int = Field(default=10000, description="最大文档长度(截取)")

    # 质量检查配置
    min_completeness_score: float = Field(default=0.5, ge=0.0, le=1.0, description="最小完整性分数")
    require_solution_steps: bool = Field(default=True, description="必须包含解决步骤")
    solution_keywords: Set[str] = Field(
        default_factory=lambda: {"执行", "运行", "修改", "配置", "重启", "安装", "更新", "设置", "调整"},
        description="解决方案关键词"
    )

    # LLM配置
    llm_model: str = Field(default="gpt-4-turbo", description="LLM模型")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API密钥")
    llm_base_url: Optional[str] = Field(default=None, description="LLM API Base URL (兼容OpenAI API)")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM温度")
    llm_max_tokens: Optional[int] = Field(default=None, description="LLM最大输出token")

    # 输出配置
    output_format: str = Field(default="markdown", description="输出格式")
    include_original_text: bool = Field(default=False, description="是否包含原文")

    # 图片处理配置
    process_images: bool = Field(default=True, description="是否处理文档中的图片")
    image_provider: Literal["minimax", "openai"] = Field(
        default="minimax", description="图片理解API提供商(minimax/openai)"
    )
    image_description_prompt: str = Field(
        default="描述这张图片的内容，包括图片中的文字、图形、图表、错误信息等关键信息",
        description="图片描述prompt"
    )
    include_images_in_output: bool = Field(default=True, description="是否在输出中包含图片")
    image_output_dir: Optional[str] = Field(default=None, description="图片输出目录(相对于输出文件)")
    # MiniMax配置
    minimax_api_key: Optional[str] = Field(default=None, description="MiniMax API密钥(用于图片理解)")
    minimax_api_host: str = Field(default="https://api.minimax.chat", description="MiniMax API地址")
    # OpenAI兼容配置(用于图片理解)
    image_openai_api_key: Optional[str] = Field(default=None, description="图片理解OpenAI API密钥")
    image_openai_base_url: Optional[str] = Field(default=None, description="图片理解OpenAI API Base URL")
    image_openai_model: str = Field(default="gpt-4o", description="图片理解模型(如gpt-4o, deepseek-vl等)")
    image_timeout: int = Field(default=60, description="图片理解API超时时间(秒)")