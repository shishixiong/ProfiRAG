"""Prompt templates for RAG generation"""

from typing import Optional, List


# Default RAG prompt template
DEFAULT_PROMPT_TEMPLATE = """Please answer the question based on the provided reference documents.

Answer requirements:
1. **Document-based**: Only use information from the reference documents, do not fabricate or use external knowledge
2. **Cite sources**: Mark information sources in your answer using [Document N] format, e.g., "According to [Document 1]..."
3. **Clear structure**: Organize your answer using sections, lists, etc. for easy reading
4. **Honest disclosure**: If the documents don't fully answer the question, clearly state "Based on the available documents, I cannot fully answer this question" and provide partial information if possible
5. **Table handling**: If tables are involved, clearly present key data without omitting important information
6. **Code handling**: For code-related questions, preserve code formatting completely

Question: {query_str}

Reference Documents:
{context_str}

Answer:"""


# Chinese prompt template
CHINESE_PROMPT_TEMPLATE = """请根据以下参考文档回答问题。

回答要求：
1. **基于文档回答**：只使用参考文档中的信息，不要编造或使用外部知识
2. **标注引用**：在回答中标注信息来源，格式为 [文档N]，例如"根据[文档1]..."
3. **结构清晰**：使用分段、列表等方式组织回答，便于阅读
4. **如实说明**：如果文档信息不足以完整回答，明确说明"根据现有文档，我无法完全回答此问题"，并尽可能提供部分信息
5. **表格处理**：如果涉及表格数据，清晰呈现关键数据，不要遗漏重要信息
6. **代码处理**：如果是代码相关问题，保持代码格式完整

问题: {query_str}

参考文档:
{context_str}

回答:"""


# Technical documentation prompt (for technical manuals like GaussDB)
TECHNICAL_PROMPT_TEMPLATE_ZH = """你是一个专业的技术文档助手，请根据参考文档回答用户的技术问题。

回答规范：
1. **准确性优先**：严格依据文档内容，不添加文档以外的假设或推断
2. **引用标注**：关键信息必须标注来源，格式：[文档N] 或"根据文档N..."
3. **结构化输出**：
   - 简单问题：直接回答
   - 复杂问题：使用"**标题**"分段，或列表形式
   - 操作步骤：按步骤编号，1. 2. 3. ...
4. **代码示例**：如果文档中有代码示例，完整保留格式，使用 ``` 代码块包裹
5. **参数说明**：涉及参数时，说明参数名称、默认值、可选值
6. **版本差异**：如果文档提到版本差异，明确指出适用版本
7. **部分回答**：文档信息不足时，说明"文档中未详细说明XX部分"，提供已知信息

问题: {query_str}

参考文档:
{context_str}

回答:"""


# Compact prompt template (for efficient generation)
COMPACT_PROMPT_TEMPLATE = """Context:
{context_str}

Question: {query_str}
Answer based on the context above:"""


# Refine prompt template (for iterative refinement)
REFINE_PROMPT_TEMPLATE = """We have an existing answer to the question:
{existing_answer}

We have the opportunity to refine this answer (only if needed) with some more context below.

Context:
{context_msg}

Question: {query_str}

Given the new context, refine the original answer to better answer the question.
If the context is not helpful, output the same original answer.
Refined Answer:"""


class PromptTemplates:
    """Collection of prompt templates for different RAG scenarios."""

    @staticmethod
    def get_template(language: str = "en", style: str = "default") -> str:
        """Get appropriate prompt template.

        Args:
            language: Language code ("en" or "zh")
            style: Style name ("default", "compact", "refine", "technical")

        Returns:
            Prompt template string
        """
        if language == "zh":
            templates_zh = {
                "default": CHINESE_PROMPT_TEMPLATE,
                "compact": COMPACT_PROMPT_TEMPLATE,
                "refine": REFINE_PROMPT_TEMPLATE,
                "technical": TECHNICAL_PROMPT_TEMPLATE_ZH,
            }
            return templates_zh.get(style, CHINESE_PROMPT_TEMPLATE)

        templates_en = {
            "default": DEFAULT_PROMPT_TEMPLATE,
            "compact": COMPACT_PROMPT_TEMPLATE,
            "refine": REFINE_PROMPT_TEMPLATE,
        }

        return templates_en.get(style, DEFAULT_PROMPT_TEMPLATE)

    @staticmethod
    def format_context(nodes: List, max_length: Optional[int] = None) -> str:
        """Format context from nodes.

        Args:
            nodes: List of NodeWithScore objects
            max_length: Maximum total context length

        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0

        for i, node_with_score in enumerate(nodes):
            text = node_with_score.node.text
            if max_length and total_length + len(text) > max_length:
                # Truncate if exceeds max length
                remaining = max_length - total_length
                if remaining > 100:
                    text = text[:remaining] + "..."
                break

            context_parts.append(f"[{i+1}] {text}")
            total_length += len(text)

        return "\n\n".join(context_parts)

    @staticmethod
    def format_prompt(
        query_str: str,
        context_str: str,
        template: Optional[str] = None,
        **kwargs
    ) -> str:
        """Format complete prompt.

        Args:
            query_str: Query string
            context_str: Context string
            template: Prompt template (uses default if None)
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        template = template or DEFAULT_PROMPT_TEMPLATE
        return template.format(
            query_str=query_str,
            context_str=context_str,
            **kwargs
        )