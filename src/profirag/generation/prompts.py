"""Prompt templates for RAG generation"""

from typing import Optional, List


# Default RAG prompt template
DEFAULT_PROMPT_TEMPLATE = """Below is a question followed by some relevant context.
Please answer the question based on the provided context.
If the context does not contain enough information to answer the question,
say that you don't know or need more information.

Question: {query_str}

Context:
{context_str}

Answer:"""


# Chinese prompt template
CHINESE_PROMPT_TEMPLATE = """请根据以下相关内容回答问题。
如果提供的内容不足以回答问题，请说明您不知道或需要更多信息。

问题: {query_str}

相关内容:
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
            style: Style name ("default", "compact", "refine")

        Returns:
            Prompt template string
        """
        if language == "zh":
            return CHINESE_PROMPT_TEMPLATE

        templates = {
            "default": DEFAULT_PROMPT_TEMPLATE,
            "compact": COMPACT_PROMPT_TEMPLATE,
            "refine": REFINE_PROMPT_TEMPLATE,
        }

        return templates.get(style, DEFAULT_PROMPT_TEMPLATE)

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