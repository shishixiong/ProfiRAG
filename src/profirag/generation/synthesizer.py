"""Response synthesizer for RAG generation"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import (
    BaseSynthesizer as LlamaResponseSynthesizer,
    get_response_synthesizer,
)

from .prompts import PromptTemplates, DEFAULT_PROMPT_TEMPLATE
from ..ingestion.image_processor import ImageResult


class ResponseSynthesizer:
    """Response synthesizer for generating answers from retrieved context.

    Supports multiple response modes:
    - compact: Combine all context into one prompt
    - refine: Iteratively refine answer with each context chunk
    - tree_summarize: Summarize context tree structure
    """

    def __init__(
        self,
        llm: Any,
        response_mode: str = "compact",
        streaming: bool = False,
        template: Optional[str] = None,
        max_context_length: Optional[int] = 80000,
        **kwargs
    ):
        """Initialize response synthesizer.

        Args:
            llm: LLM instance for generation
            response_mode: Response generation mode:
                - "compact": Single prompt with all context
                - "refine": Iterative refinement
                - "tree_summarize": Hierarchical summarization
            streaming: Enable streaming output
            template: Custom prompt template
            max_context_length: Maximum context length (truncate if exceeds)
            **kwargs: Additional arguments for LlamaIndex synthesizer
        """
        self.llm = llm
        self.response_mode = response_mode
        self.streaming = streaming
        self.template = template or DEFAULT_PROMPT_TEMPLATE
        self.max_context_length = max_context_length
        self.kwargs = kwargs

        # Initialize LlamaIndex synthesizer
        self._synthesizer: Optional[LlamaResponseSynthesizer] = None

    def _get_synthesizer(self) -> LlamaResponseSynthesizer:
        """Get or create LlamaIndex response synthesizer."""
        if self._synthesizer is None:
            self._synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode=self.response_mode,
                streaming=self.streaming,
                **self.kwargs
            )
        return self._synthesizer

    def synthesize(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
        **kwargs
    ) -> str:
        """Synthesize response from retrieved nodes.

        Args:
            query_str: Query string
            nodes: List of NodeWithScore objects with context
            **kwargs: Additional arguments

        Returns:
            Generated response (string or streaming generator)
        """
        synthesizer = self._get_synthesizer()

        # Truncate context if needed
        if self.max_context_length:
            nodes = self._truncate_context(nodes, self.max_context_length)

        response = synthesizer.synthesize(
            query=query_str,
            nodes=nodes,
            **kwargs
        )

        return response.response

    def _truncate_context(
        self,
        nodes: List[NodeWithScore],
        max_length: int
    ) -> List[NodeWithScore]:
        """Truncate context to maximum length.

        Args:
            nodes: List of nodes
            max_length: Maximum total context length

        Returns:
            Truncated list of nodes
        """
        truncated = []
        total_length = 0

        for node_with_score in nodes:
            text_length = len(node_with_score.node.text)
            if total_length + text_length > max_length:
                break
            truncated.append(node_with_score)
            total_length += text_length

        return truncated

    def synthesize_streaming(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
        **kwargs
    ):
        """Stream response generation.

        Args:
            query_str: Query string
            nodes: List of NodeWithScore objects
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        if not self.streaming:
            # Enable streaming temporarily
            synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode=self.response_mode,
                streaming=True,
                **self.kwargs
            )
        else:
            synthesizer = self._get_synthesizer()

        # Truncate context if needed
        if self.max_context_length:
            nodes = self._truncate_context(nodes, self.max_context_length)

        response = synthesizer.synthesize(
            query=query_str,
            nodes=nodes,
            **kwargs
        )

        for chunk in response.response_gen:
            yield chunk

    def synthesize_custom(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synthesize with custom prompt template.

        Args:
            query_str: Query string
            nodes: List of NodeWithScore objects
            custom_prompt: Custom prompt template
            **kwargs: Additional arguments

        Returns:
            Generated response string
        """
        prompt_template = custom_prompt or self.template
        context_str = PromptTemplates.format_context(nodes, self.max_context_length)
        prompt = PromptTemplates.format_prompt(
            query_str=query_str,
            context_str=context_str,
            template=prompt_template
        )

        response = self.llm.complete(prompt)
        return response.text


class StreamingResponseHandler:
    """Handler for streaming response processing."""

    def __init__(
        self,
        synthesizer: ResponseSynthesizer,
        callback: Optional[callable] = None
    ):
        """Initialize streaming handler.

        Args:
            synthesizer: Response synthesizer instance
            callback: Optional callback for each chunk
        """
        self.synthesizer = synthesizer
        self.callback = callback

    def handle_stream(
        self,
        query_str: str,
        nodes: List[NodeWithScore]
    ) -> str:
        """Handle streaming response and return final result.

        Args:
            query_str: Query string
            nodes: List of NodeWithScore objects

        Returns:
            Complete response string
        """
        full_response = []
        for chunk in self.synthesizer.synthesize_streaming(query_str, nodes):
            full_response.append(chunk)
            if self.callback:
                self.callback(chunk)

        return "".join(full_response)


class ResponseFormatter:
    """Format responses with metadata."""

    @staticmethod
    def format_with_sources(
        response: str,
        nodes: List[NodeWithScore],
        include_scores: bool = True
    ) -> Dict[str, Any]:
        """Format response with source information.

        Args:
            response: Response string
            nodes: Source nodes
            include_scores: Include relevance scores

        Returns:
            Formatted response dictionary
        """
        sources = []
        for i, node_with_score in enumerate(nodes):
            source_info = {
                "index": i + 1,
                "text": node_with_score.node.text[:200] + "..." if len(node_with_score.node.text) > 200 else node_with_score.node.text,
                "metadata": node_with_score.node.metadata,
                "node_id": node_with_score.node.node_id,
            }
            if include_scores:
                source_info["score"] = node_with_score.score
            sources.append(source_info)

        return {
            "response": response,
            "sources": sources,
            "num_sources": len(nodes),
        }

    @staticmethod
    def format_markdown(
        response: str,
        nodes: List[NodeWithScore],
        show_full_text: bool = False
    ) -> str:
        """Format response as markdown with sources.

        Args:
            response: Response string
            nodes: Source nodes
            show_full_text: Show full source text

        Returns:
            Markdown formatted string
        """
        md_parts = [f"## Response\n\n{response}\n"]

        if nodes:
            md_parts.append("\n## Sources\n")
            for i, node_with_score in enumerate(nodes):
                text = node_with_score.node.text if show_full_text else node_with_score.node.text[:300]
                if not show_full_text and len(node_with_score.node.text) > 300:
                    text += "..."

                md_parts.append(f"\n### Source [{i+1}] (Score: {node_with_score.score:.3f})")
                md_parts.append(f"\n```\n{text}\n```")

        return "".join(md_parts)

    @staticmethod
    def format_with_sources_and_images(
        response: str,
        nodes: List[NodeWithScore],
        images: List[ImageResult],
        include_scores: bool = True,
    ) -> Dict[str, Any]:
        """Format response with sources and images.

        Args:
            response: Response string
            nodes: Source nodes
            images: List of ImageResult objects
            include_scores: Include relevance scores

        Returns:
            Formatted response dictionary with images
        """
        # Get basic sources format
        result = ResponseFormatter.format_with_sources(response, nodes, include_scores)

        # Add images
        image_list = []
        for i, img in enumerate(images):
            image_info = {
                "index": i + 1,
                "path": img.image_path,
                "description": img.description,
                "score": img.score if include_scores else None,
                "source_chunk": img.source_chunk_id,
            }
            if img.metadata:
                image_info["metadata"] = img.metadata
            image_list.append(image_info)

        result["images"] = image_list
        result["num_images"] = len(images)

        return result

    @staticmethod
    def format_markdown_with_images(
        response: str,
        nodes: List[NodeWithScore],
        images: List[ImageResult],
        show_full_text: bool = False,
        show_images: bool = True,
    ) -> str:
        """Format response as markdown with sources and images.

        Args:
            response: Response string
            nodes: Source nodes
            images: List of ImageResult objects
            show_full_text: Show full source text
            show_images: Embed image references in markdown

        Returns:
            Markdown formatted string with images
        """
        md_parts = [f"## Response\n\n{response}\n"]

        # Add images section if available
        if images and show_images:
            md_parts.append("\n## Related Images\n")
            for i, img in enumerate(images):
                md_parts.append(f"\n### Image {i+1}\n")
                md_parts.append(f"![Image {i+1}]({img.image_path})\n")
                if img.description:
                    md_parts.append(f"\n*Description: {img.description}*\n")

        # Add sources section
        if nodes:
            md_parts.append("\n## Sources\n")
            for i, node_with_score in enumerate(nodes):
                text = node_with_score.node.text if show_full_text else node_with_score.node.text[:300]
                if not show_full_text and len(node_with_score.node.text) > 300:
                    text += "..."

                md_parts.append(f"\n### Source [{i+1}] (Score: {node_with_score.score:.3f})")
                md_parts.append(f"\n```\n{text}\n```")

        return "".join(md_parts)