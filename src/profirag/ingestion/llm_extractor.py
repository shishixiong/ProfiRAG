"""LLM-based extractor for semantic structure extraction."""

import json
import re
import logging
from typing import Optional, List, Dict, Any

from llama_index.core.llms import LLM

from ..config.settings import CustomOpenAILLM
from .cleaner_config import (
    StructureResult,
    ProblemElement,
    CauseAnalysis,
    Solution,
    TroubleshootingStep,
    RuleResult,
    CleanerConfig,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

STRUCTURE_EXTRACTION_PROMPT = """你是一个故障排查文档分析专家。请从以下工单/问题单文档中提取核心要素。

【原文档内容】
{document_text}

【已知信息】（规则提取器已识别）
{rule_hints}

请按以下JSON格式返回：

```json
{{
  "problem": {{
    "description": "问题现象的简洁描述（一句话概括）",
    "symptoms": ["症状1", "症状2"],
    "affected_components": ["受影响的组件或服务"]
  }},
  "cause": {{
    "root_cause": "根本原因（一句话概括）",
    "contributing_factors": ["促成因素1", "促成因素2"]
  }},
  "solution": {{
    "troubleshooting_steps": [
      {{"description": "排查步骤描述", "command": "执行的命令（如有）", "result": "排查结果（如有）"}}
    ],
    "steps": ["最终解决方案步骤1", "解决方案步骤2"],
    "commands": ["需要执行的具体命令"],
    "verification": "验证问题已解决的方法"
  }},
  "confidence_score": 0.85
}}
```

提取原则：
1. **问题现象(problem)**：描述用户遇到的具体问题，包括症状和受影响的组件
2. **原因分析(cause)**：描述问题的根本原因，可能包括多个促成因素
3. **排查步骤(troubleshooting_steps)**：文档中的诊断过程（如检查日志、查看配置、运行诊断命令）
4. **解决步骤(steps)**：最终的解决方案（不是排查过程）
5. **置信度(confidence_score)**：根据信息完整性评估，范围0-1

注意事项：
- 如果文档没有明确的三要素结构，根据语义推断提取
- solution的steps必须包含可操作的内容，不能只是"联系厂商"或"咨询技术支持"
- 如果信息不完整，如实填写空数组，并降低confidence_score
- commands字段提取文档中具体的命令行指令
- 已知信息中的错误码、服务组件可作为提取的参考
"""


CONTRADICTION_CHECK_PROMPT = """请检查以下故障排查文档是否存在信息矛盾或逻辑问题。

【问题现象】
{problem}

【原因分析】
{cause}

【解决方案】
{solution}

判断标准：
1. 原因分析是否与问题现象匹配？问题描述的现象能否由所述原因解释？
2. 解决方案是否针对所述的根本原因？方案能否解决所述问题？
3. 是否存在前后矛盾的说法（如：原因说是权限问题，方案却说重启服务）？

请按以下JSON格式返回：

```json
{{
  "has_contradictions": false,
  "description": "如果有矛盾，描述矛盾内容；如果没有，说明一致性良好",
  "match_score": 0.85
}}
```

match_score范围0-1，表示问题-原因-解决方案的匹配程度。
"""


COMPLETENESS_CHECK_PROMPT = """请评估以下故障排查文档的完整性。

【问题现象】
{problem}

【原因分析】
{cause}

【解决方案】
{solution}

评估标准：
1. 问题现象：是否有清晰的描述？
2. 原因分析：是否说明了根本原因？
3. 解决方案：是否有可操作的步骤或命令？

请按以下JSON格式返回：

```json
{{
  "completeness_score": 0.7,
  "missing_elements": ["排查步骤缺失", "验证方法未说明"],
  "assessment": "简要评估说明"
}}
```

completeness_score范围0-1：
- 1.0: 三要素完整且有排查步骤和验证方法
- 0.7-0.9: 三要素基本完整，缺少部分细节
- 0.5-0.6: 有两要素，缺少一个核心要素
- <0.5: 信息严重不足
"""


class LLMExtractor:
    """LLM提取器 - 语义理解和结构提取"""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        config: Optional[CleanerConfig] = None,
    ):
        self.config = config or CleanerConfig()
        self._llm = llm or self._create_default_llm()

    def _create_default_llm(self) -> LLM:
        """创建默认LLM（支持OpenAI兼容API）"""
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

    def extract_structure(
        self,
        text: str,
        hints: Optional[RuleResult] = None
    ) -> StructureResult:
        """从文档中提取三要素结构"""

        # 生成规则提示
        rule_hints = "未识别到特定信息"
        if hints:
            from .rule_extractor import RuleExtractor
            extractor = RuleExtractor(self.config)
            rule_hints = extractor.get_hints_for_llm(hints)

        # 截取文档内容
        max_len = self.config.max_document_length
        if len(text) > max_len:
            text = text[:max_len]

        # 构建prompt
        prompt = STRUCTURE_EXTRACTION_PROMPT.format(
            document_text=text,
            rule_hints=rule_hints,
        )

        # 调用LLM
        try:
            logger.debug(f"Calling LLM with prompt length: {len(prompt)}")
            response = self._llm.complete(prompt)
            logger.debug(f"LLM response received, length: {len(response.text)}")
            logger.debug(f"LLM response preview: {response.text[:300]}...")
            return self._parse_structure_response(response.text)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            return StructureResult()

    def _parse_structure_response(self, response: str) -> StructureResult:
        """解析LLM响应"""
        try:
            logger.debug(f"LLM raw response: {response[:500]}...")

            # 提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个响应
                json_str = response.strip()

            logger.debug(f"Parsed JSON string: {json_str[:200]}...")
            data = json.loads(json_str)

            # 解析problem
            problem_data = data.get("problem", {})
            problem = ProblemElement(
                description=problem_data.get("description", ""),
                symptoms=problem_data.get("symptoms", []),
                affected_components=problem_data.get("affected_components", []),
            )

            # 解析cause
            cause_data = data.get("cause", {})
            cause = CauseAnalysis(
                root_cause=cause_data.get("root_cause", ""),
                contributing_factors=cause_data.get("contributing_factors", []),
            )

            # 解析solution
            solution_data = data.get("solution", {})
            troubleshooting_steps = []
            for step_data in solution_data.get("troubleshooting_steps", []):
                troubleshooting_steps.append(TroubleshootingStep(
                    description=step_data.get("description", ""),
                    command=step_data.get("command"),
                    result=step_data.get("result"),
                ))
            solution = Solution(
                troubleshooting_steps=troubleshooting_steps,
                steps=solution_data.get("steps", []),
                commands=solution_data.get("commands", []),
                verification=solution_data.get("verification"),
            )

            confidence = data.get("confidence_score", 0.5)

            return StructureResult(
                problem=problem,
                cause=cause,
                solution=solution,
                confidence_score=confidence,
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Full response: {response}")
            # 尝试修复常见的JSON问题
            try:
                # 尝试找到第一个 { 和最后一个 }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                    logger.debug(f"Attempting to parse extracted JSON: {json_str[:200]}...")
                    data = json.loads(json_str)
                    # 继续正常的解析流程
                    problem_data = data.get("problem", {})
                    problem = ProblemElement(
                        description=problem_data.get("description", ""),
                        symptoms=problem_data.get("symptoms", []),
                        affected_components=problem_data.get("affected_components", []),
                    )
                    cause_data = data.get("cause", {})
                    cause = CauseAnalysis(
                        root_cause=cause_data.get("root_cause", ""),
                        contributing_factors=cause_data.get("contributing_factors", []),
                    )
                    solution_data = data.get("solution", {})
                    troubleshooting_steps = []
                    for step_data in solution_data.get("troubleshooting_steps", []):
                        troubleshooting_steps.append(TroubleshootingStep(
                            description=step_data.get("description", ""),
                            command=step_data.get("command"),
                            result=step_data.get("result"),
                        ))
                    solution = Solution(
                        troubleshooting_steps=troubleshooting_steps,
                        steps=solution_data.get("steps", []),
                        commands=solution_data.get("commands", []),
                        verification=solution_data.get("verification"),
                    )
                    confidence = data.get("confidence_score", 0.5)
                    return StructureResult(
                        problem=problem,
                        cause=cause,
                        solution=solution,
                        confidence_score=confidence,
                    )
            except Exception as e2:
                logger.error(f"JSON repair also failed: {e2}")
            return StructureResult()
        except Exception as e:
            logger.error(f"Structure parsing failed: {e}")
            return StructureResult()

    def check_completeness(self, structure: StructureResult) -> Dict[str, Any]:
        """检查三要素完整性"""

        prompt = COMPLETENESS_CHECK_PROMPT.format(
            problem=json.dumps(structure.problem.model_dump(), ensure_ascii=False),
            cause=json.dumps(structure.cause.model_dump(), ensure_ascii=False),
            solution=json.dumps(structure.solution.model_dump(), ensure_ascii=False),
        )

        try:
            response = self._llm.complete(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"Completeness check failed: {e}")
            return {
                "completeness_score": 0.5,
                "missing_elements": ["检查失败"],
                "assessment": "LLM调用失败",
            }

    def check_contradictions(self, structure: StructureResult) -> Dict[str, Any]:
        """检查信息矛盾"""

        prompt = CONTRADICTION_CHECK_PROMPT.format(
            problem=json.dumps(structure.problem.model_dump(), ensure_ascii=False),
            cause=json.dumps(structure.cause.model_dump(), ensure_ascii=False),
            solution=json.dumps(structure.solution.model_dump(), ensure_ascii=False),
        )

        try:
            response = self._llm.complete(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"Contradiction check failed: {e}")
            return {
                "has_contradictions": False,
                "description": "检查失败，默认无矛盾",
                "match_score": 0.5,
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}