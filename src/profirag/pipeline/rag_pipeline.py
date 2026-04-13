"""Main RAG pipeline integrating all components"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from ..config.settings import RAGConfig
from ..storage.registry import StorageRegistry
from ..storage.base import BaseVectorStore
from ..retrieval.query_transform import PreRetrievalPipeline
from ..retrieval.hybrid import HybridRetriever, BM25Index
from ..retrieval.reranker import Reranker
from ..generation.synthesizer import ResponseSynthesizer, ResponseFormatter


class RAGPipeline:
    """Advanced RAG pipeline implementing the complete RAG workflow.

    Pipeline stages:
    1. Pre-Retrieval: Query transformation (HyDE, rewriting, multi-query)
    2. Retrieval: Hybrid retrieval (vector + BM25)
    3. Post-Retrieval: Reranking
    4. Generation: Response synthesis

    Supports multiple vector store backends through storage abstraction.
    """

    def __init__(
        self,
        config: RAGConfig,
        **kwargs
    ):
        """Initialize RAG pipeline.

        Args:
            config: RAGConfig instance with complete configuration
            **kwargs: Additional arguments
        """
        self.config = config
        self.kwargs = kwargs

        # Initialize embedding model
        self._embed_model = self._create_embed_model()

        # Initialize LLM
        self._llm = self._create_llm()

        # Initialize vector store
        self._vector_store = self._create_vector_store()

        # Initialize index
        self._index = self._create_index()

        # Initialize BM25 index if enabled
        self._bm25_index: Optional[BM25Index] = None
        if config.retrieval.use_bm25:
            self._bm25_index = BM25Index()

        # Initialize components
        self._pre_retrieval = PreRetrievalPipeline(
            llm=self._llm,
            config={
                "use_hyde": config.pre_retrieval.use_hyde,
                "use_rewrite": config.pre_retrieval.use_rewrite,
                "multi_query": config.pre_retrieval.multi_query,
            }
        )

        self._hybrid_retriever = HybridRetriever(
            vector_index=self._index,
            bm25_index=self._bm25_index,
            alpha=config.retrieval.alpha,
        )

        self._reranker = Reranker(
            model=config.reranking.model,
            top_n=config.reranking.top_n,
            enabled=config.reranking.enabled,
        )

        self._synthesizer = ResponseSynthesizer(
            llm=self._llm,
            response_mode=config.generation.response_mode,
            streaming=config.generation.streaming,
        )

    def _create_embed_model(self) -> OpenAIEmbedding:
        """Create embedding model."""
        embed_kwargs = {
            "model": self.config.embedding.model,
            "dimension": self.config.embedding.dimension,
            "api_key": self.config.embedding.api_key,
        }
        if self.config.embedding.base_url:
            embed_kwargs["api_base"] = self.config.embedding.base_url
        return OpenAIEmbedding(**embed_kwargs)

    def _create_llm(self) -> OpenAI:
        """Create LLM instance."""
        llm_kwargs = {
            "model": self.config.llm.model,
            "api_key": self.config.llm.api_key,
            "temperature": self.config.llm.temperature,
        }
        if self.config.llm.max_tokens:
            llm_kwargs["max_tokens"] = self.config.llm.max_tokens
        if self.config.llm.base_url:
            llm_kwargs["api_base"] = self.config.llm.base_url
        return OpenAI(**llm_kwargs)

    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration."""
        return StorageRegistry.get_store(
            self.config.storage.type,
            self.config.storage.config
        )

    def _create_index(self) -> VectorStoreIndex:
        """Create vector store index."""
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store.to_llamaindex_vector_store()
        )
        return VectorStoreIndex.from_vector_store(
            self._vector_store.to_llamaindex_vector_store(),
            embed_model=self._embed_model,
            storage_context=storage_context,
        )

    def ingest_documents(
        self,
        documents: List[Document],
        update_bm25: bool = True,
        **kwargs
    ) -> List[str]:
        """Ingest documents into the vector store.

        Args:
            documents: List of Document objects to ingest
            update_bm25: Update BM25 index with new nodes
            **kwargs: Additional arguments for ingestion

        Returns:
            List of document IDs that were ingested
        """
        # Add documents to index
        for doc in documents:
            self._index.insert(doc, **kwargs)

        # Update BM25 index if enabled
        if update_bm25 and self._bm25_index:
            # Get nodes from ingested documents
            for doc in documents:
                ref_doc_info = self._vector_store.get_ref_doc_info(doc.doc_id)
                if ref_doc_info:
                    for node_id in ref_doc_info.node_ids:
                        node = self._vector_store.get_node(node_id)
                        if node:
                            self._bm25_index.add_nodes([node])

        return [doc.doc_id for doc in documents]

    def ingest_nodes(
        self,
        nodes: List[TextNode],
        update_bm25: bool = True,
        **kwargs
    ) -> List[str]:
        """Ingest nodes directly into the vector store.

        Args:
            nodes: List of TextNode objects
            update_bm25: Update BM25 index
            **kwargs: Additional arguments

        Returns:
            List of node IDs
        """
        node_ids = self._vector_store.add(nodes, **kwargs)

        if update_bm25 and self._bm25_index:
            self._bm25_index.add_nodes(nodes)

        return node_ids

    def query(
        self,
        query_str: str,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute RAG query and return response.

        Args:
            query_str: Query string
            top_k: Number of results to retrieve
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - response: Generated answer
                - source_nodes: Retrieved source nodes
                - metadata: Query and retrieval metadata
        """
        # 1. Pre-retrieval: Transform query
        query_bundles = self._pre_retrieval.transform(query_str)

        # 2. Retrieval: Hybrid search
        all_nodes: List[NodeWithScore] = []
        for qb in query_bundles:
            nodes = self._hybrid_retriever.retrieve(qb.query_str, top_k=top_k)
            all_nodes.extend(nodes)

        # Deduplicate nodes
        unique_nodes = self._deduplicate_nodes(all_nodes)

        # 3. Post-retrieval: Rerank
        reranked_nodes = self._reranker.rerank(query_str, unique_nodes)

        # 4. Generation: Synthesize response
        response = self._synthesizer.synthesize(query_str, reranked_nodes[:top_k])

        return {
            "response": response,
            "source_nodes": reranked_nodes[:top_k],
            "metadata": {
                "query_variants": [qb.query_str for qb in query_bundles],
                "total_nodes_retrieved": len(all_nodes),
                "unique_nodes": len(unique_nodes),
                "nodes_after_reranking": len(reranked_nodes),
            }
        }

    def query_stream(
        self,
        query_str: str,
        top_k: int = 10,
        **kwargs
    ):
        """Execute RAG query with streaming response.

        Args:
            query_str: Query string
            top_k: Number of results
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        # Execute retrieval
        query_bundles = self._pre_retrieval.transform(query_str)
        all_nodes = []
        for qb in query_bundles:
            nodes = self._hybrid_retriever.retrieve(qb.query_str, top_k=top_k)
            all_nodes.extend(nodes)

        unique_nodes = self._deduplicate_nodes(all_nodes)
        reranked_nodes = self._reranker.rerank(query_str, unique_nodes)

        # Stream response
        for chunk in self._synthesizer.synthesize_streaming(query_str, reranked_nodes[:top_k]):
            yield chunk

    def _deduplicate_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Remove duplicate nodes based on node_id.

        Args:
            nodes: List of nodes (possibly with duplicates)

        Returns:
            Deduplicated list
        """
        seen_ids: set = set()
        unique: List[NodeWithScore] = []

        for node in nodes:
            if node.node.node_id not in seen_ids:
                seen_ids.add(node.node.node_id)
                unique.append(node)

        return unique

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its associated nodes.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        return self._vector_store.delete(ref_doc_id=doc_id)

    def clear(self) -> None:
        """Clear all data from the pipeline."""
        self._vector_store.clear()
        if self._bm25_index:
            self._bm25_index.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "vector_store": {
                "type": self.config.storage.type,
                "count": self._vector_store.count(),
            },
            "bm25_index": {
                "enabled": self.config.retrieval.use_bm25,
                "count": self._bm25_index.count() if self._bm25_index else 0,
            },
            "reranking": {
                "enabled": self.config.reranking.enabled,
                "model": self.config.reranking.model,
            },
            "embedding": {
                "model": self.config.embedding.model,
                "dimension": self.config.embedding.dimension,
            },
            "llm": {
                "model": self.config.llm.model,
            },
        }

    @classmethod
    def from_config_file(cls, config_path: str) -> "RAGPipeline":
        """Create pipeline from YAML config file.

        Args:
            config_path: Path to YAML config file

        Returns:
            RAGPipeline instance
        """
        config = RAGConfig.from_yaml(config_path)
        return cls(config)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "RAGPipeline":
        """Create pipeline from .env file and environment variables.

        Args:
            env_file: Path to .env file (default: ".env" in current directory)

        Returns:
            RAGPipeline instance
        """
        config = RAGConfig.from_env(env_file)
        return cls(config)
