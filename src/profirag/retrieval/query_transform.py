"""Pre-retrieval query transformation components"""

from typing import List, Dict, Any, Optional
from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform


class HyDEQueryTransform(BaseQueryTransform):
    """HyDE (Hypothetical Document Embeddings) query transform.

    Generates a hypothetical document that would answer the query,
    then uses its embedding for retrieval.
    """

    def __init__(
        self,
        llm: Any,
        hyde_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize HyDE transform.

        Args:
            llm: LLM instance for generating hypothetical documents
            hyde_prompt: Custom prompt template for HyDE
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.hyde_prompt = hyde_prompt or self._default_prompt()

    def _default_prompt(self) -> str:
        """Default HyDE prompt template."""
        return """Please write a passage that would answer the following question.

Question: {query_str}

Passage:"""

    def _run_query(self, query_str: str, **kwargs) -> str:
        """Generate hypothetical document."""
        prompt = self.hyde_prompt.format(query_str=query_str)
        response = self.llm.complete(prompt)
        return response.text

    def run(self, query_str: str, **kwargs) -> QueryBundle:
        """Transform query using HyDE.

        Args:
            query_str: Original query string
            **kwargs: Additional arguments

        Returns:
            QueryBundle with hypothetical document
        """
        hypothetical_doc = self._run_query(query_str, **kwargs)
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=[hypothetical_doc],
            embedding_mode="custom"
        )


class QueryRewriter:
    """Query rewriting component for improving query quality."""

    def __init__(
        self,
        llm: Any,
        rewrite_prompt: Optional[str] = None,
        **kwargs
    ):
        """Initialize query rewriter.

        Args:
            llm: LLM instance for rewriting
            rewrite_prompt: Custom prompt template
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.rewrite_prompt = rewrite_prompt or self._default_prompt()

    def _default_prompt(self) -> str:
        """Default rewrite prompt template."""
        return """Please rewrite the following query to make it more suitable for document retrieval.
Keep the core meaning but make it clearer and more specific.

Original query: {query_str}

Rewritten query:"""

    def rewrite(self, query_str: str) -> str:
        """Rewrite the query.

        Args:
            query_str: Original query string

        Returns:
            Rewritten query string
        """
        prompt = self.rewrite_prompt.format(query_str=query_str)
        response = self.llm.complete(prompt)
        return response.text.strip()


class MultiQueryGenerator:
    """Generate multiple query variants for broader retrieval coverage."""

    def __init__(
        self,
        llm: Any,
        num_queries: int = 3,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize multi-query generator.

        Args:
            llm: LLM instance for generating variants
            num_queries: Number of query variants to generate
            prompt_template: Custom prompt template
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.num_queries = num_queries
        self.prompt_template = prompt_template or self._default_prompt()

    def _default_prompt(self) -> str:
        """Default multi-query prompt template."""
        return """Generate {num_queries} different versions of the following query to improve document retrieval.
Each version should express the same intent but use different wording.

Original query: {query_str}

Generate {num_queries} alternative queries (one per line):"""

    def generate(self, query_str: str) -> List[str]:
        """Generate query variants.

        Args:
            query_str: Original query string

        Returns:
            List of query variant strings
        """
        prompt = self.prompt_template.format(
            query_str=query_str,
            num_queries=self.num_queries
        )
        response = self.llm.complete(prompt)

        # Parse variants
        variants = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith(("Query", "Alternative", "Version", "-")):
                # Remove numbered prefixes
                if line[0].isdigit() and (line[1] in ".: " or line[1:3] in (". ", ": ")):
                    line = line.split(".", 1)[-1].split(":", 1)[-1].strip()
                variants.append(line)

        # Limit to requested number
        return variants[:self.num_queries]


class PreRetrievalPipeline:
    """Pre-retrieval processing pipeline.

    Combines multiple query transformation techniques:
    - HyDE (Hypothetical Document Embeddings)
    - Query Rewriting
    - Multi-Query generation
    """

    def __init__(
        self,
        llm: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize pre-retrieval pipeline.

        Args:
            llm: LLM instance for transformations
            config: Configuration dictionary with keys:
                - use_hyde: Enable HyDE (default False)
                - use_rewrite: Enable query rewriting (default False)
                - multi_query: Enable multi-query generation (default False)
                - num_queries: Number of multi-query variants (default 3)
                - hyde_prompt: Custom HyDE prompt
                - rewrite_prompt: Custom rewrite prompt
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.config = config or {}

        # Initialize transforms
        self._hyde_transform: Optional[HyDEQueryTransform] = None
        self._query_rewriter: Optional[QueryRewriter] = None
        self._multi_query_generator: Optional[MultiQueryGenerator] = None

        if self.config.get("use_hyde"):
            self._hyde_transform = HyDEQueryTransform(
                llm=llm,
                hyde_prompt=self.config.get("hyde_prompt")
            )

        if self.config.get("use_rewrite"):
            self._query_rewriter = QueryRewriter(
                llm=llm,
                rewrite_prompt=self.config.get("rewrite_prompt")
            )

        if self.config.get("multi_query"):
            self._multi_query_generator = MultiQueryGenerator(
                llm=llm,
                num_queries=self.config.get("num_queries", 3),
                prompt_template=self.config.get("multi_query_prompt")
            )

    def transform(self, query_str: str) -> List[QueryBundle]:
        """Transform the query, returning one or more query variants.

        Args:
            query_str: Original query string

        Returns:
            List of QueryBundle objects (original + transformed variants)
        """
        query_bundles = []

        # Always include original query
        query_bundles.append(QueryBundle(query_str=query_str))

        # HyDE: Hypothetical document embedding
        if self._hyde_transform:
            hyde_bundle = self._hyde_transform.run(query_str)
            query_bundles.append(hyde_bundle)

        # Query rewriting
        if self._query_rewriter:
            rewritten = self._query_rewriter.rewrite(query_str)
            if rewritten and rewritten != query_str:
                query_bundles.append(QueryBundle(query_str=rewritten))

        # Multi-query generation
        if self._multi_query_generator:
            variants = self._multi_query_generator.generate(query_str)
            for variant in variants:
                if variant and variant != query_str:
                    query_bundles.append(QueryBundle(query_str=variant))

        return query_bundles

    def transform_single(self, query_str: str) -> QueryBundle:
        """Transform query to single QueryBundle (HyDE mode).

        Args:
            query_str: Original query string

        Returns:
            Single QueryBundle (original or transformed)
        """
        if self._hyde_transform:
            return self._hyde_transform.run(query_str)
        return QueryBundle(query_str=query_str)