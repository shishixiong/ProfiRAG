"""Main RAG pipeline integrating all components"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata

from ..config.settings import RAGConfig
from ..embedding import CustomOpenAIEmbedding
from ..storage.registry import StorageRegistry
from ..storage.base import BaseVectorStore
from ..retrieval.query_transform import PreRetrievalPipeline
from ..retrieval.hybrid import HybridRetriever, BM25Index
from ..retrieval.reranker import Reranker
from ..generation.synthesizer import ResponseSynthesizer, ResponseFormatter
from ..ingestion.splitters import TextSplitter, ChineseTextSplitter
from ..ingestion.image_processor import ImageProcessor, ImageResult, RetrievalResult


class CustomOpenAILLM(OpenAI):
    """Custom OpenAI LLM that bypasses model name validation.

    Allows using custom model names (like MiniMax-M2.7) with OpenAI-compatible APIs.
    """

    @property
    def metadata(self) -> LLMMetadata:
        """Override metadata to bypass model validation."""
        # Get values from model_dump() to avoid pydantic __getattr__ issues
        model_dict = self.model_dump()

        return LLMMetadata(
            context_window=128000,  # Fixed context window for custom models
            num_output=model_dict.get('max_tokens') or -1,
            is_chat_model=True,  # All modern APIs use chat mode
            is_function_calling_model=True,
            model_name=model_dict.get('model', 'unknown'),
        )


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

        # Initialize splitter
        self._splitter = self._create_splitter()

        # Initialize image processor if enabled
        self._image_processor: Optional[ImageProcessor] = None
        if config.image_processing.enabled:
            self._image_processor = ImageProcessor(
                api_key=config.image_processing.minimax_api_key,
                api_host=config.image_processing.minimax_api_host,
                description_prompt=config.image_processing.description_prompt,
                storage_path=config.image_processing.storage_path,
                generate_descriptions=config.image_processing.generate_descriptions,
            )

    def _create_splitter(self):
        """Create text splitter based on configuration."""
        chunking = self.config.chunking
        if chunking.splitter_type == "chinese" or chunking.language == "zh":
            return ChineseTextSplitter(
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap,
            )
        else:
            return TextSplitter(
                splitter_type=chunking.splitter_type,
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap,
                embed_model=self._embed_model if chunking.splitter_type == "semantic" else None,
            )

    def _create_embed_model(self) -> CustomOpenAIEmbedding:
        """Create embedding model."""
        embed_kwargs = {
            "model": self.config.embedding.model,
            "api_key": self.config.embedding.api_key,
        }
        if self.config.embedding.dimension:
            embed_kwargs["dimensions"] = self.config.embedding.dimension
        if self.config.embedding.base_url:
            embed_kwargs["api_base"] = self.config.embedding.base_url
        return CustomOpenAIEmbedding(**embed_kwargs)

    def _create_llm(self) -> CustomOpenAILLM:
        """Create LLM instance using custom OpenAI-compatible wrapper."""
        llm_kwargs = {
            "model": self.config.llm.model,
            "api_key": self.config.llm.api_key,
            "temperature": self.config.llm.temperature,
            # Set context window for custom models (MiniMax has ~128k context)
            "context_window": 128000,
            "is_chat_model": True,
        }
        if self.config.llm.max_tokens:
            llm_kwargs["max_tokens"] = self.config.llm.max_tokens
        if self.config.llm.base_url:
            llm_kwargs["api_base"] = self.config.llm.base_url
        return CustomOpenAILLM(**llm_kwargs)

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
        use_custom_splitter: bool = True,
        process_images: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Ingest documents into the vector store.

        Args:
            documents: List of Document objects to ingest
            update_bm25: Update BM25 index with new nodes
            use_custom_splitter: Use configured splitter (True) or llama_index default (False)
            process_images: Process images and generate descriptions if enabled
            **kwargs: Additional arguments for ingestion

        Returns:
            Dictionary containing:
                - document_ids: List of ingested document IDs
                - text_node_ids: List of text node IDs
                - image_node_ids: List of image node IDs (if processed)
        """
        text_nodes = []
        image_nodes = []

        # Process images if enabled and documents have images
        if process_images and self._image_processor:
            for doc in documents:
                image_path = doc.metadata.get("image_path")
                image_map = doc.metadata.get("image_map", {})
                if image_path:
                    doc_images = self._image_processor.process_images_from_directory(
                        image_path,
                        doc.metadata.get("source_file", ""),
                        image_map,
                    )
                    image_nodes.extend(doc_images)

        if use_custom_splitter:
            # Split documents using configured splitter
            text_nodes = self._splitter.split_documents(documents)
            # Insert nodes directly
            for node in text_nodes:
                self._index.insert_nodes([node], **kwargs)

            # Update BM25 index if enabled
            if update_bm25 and self._bm25_index:
                self._bm25_index.add_nodes(text_nodes)
        else:
            # Add documents to index (uses llama_index default chunking)
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

        # Insert image nodes if any
        image_node_ids = []
        if image_nodes:
            self._index.insert_nodes(image_nodes, **kwargs)
            image_node_ids = [node.node_id for node in image_nodes]
            if update_bm25 and self._bm25_index:
                self._bm25_index.add_nodes(image_nodes)

        return {
            "document_ids": [doc.doc_id for doc in documents],
            "text_node_ids": [node.node_id for node in text_nodes],
            "image_node_ids": image_node_ids,
        }

    def ingest_nodes(
        self,
        nodes: List[TextNode],
        update_bm25: bool = True,
        **kwargs
    ) -> List[str]:
        """Ingest nodes directly into the vector store.

        Nodes will be automatically embedded using the configured embedding model.

        Args:
            nodes: List of TextNode objects (embedding will be generated if not set)
            update_bm25: Update BM25 index
            **kwargs: Additional arguments

        Returns:
            List of node IDs
        """
        # Use index to insert nodes (auto-generates embeddings)
        self._index.insert_nodes(nodes, **kwargs)
        node_ids = [node.node_id for node in nodes]

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

    def query_with_images(
        self,
        query_str: str,
        top_k: int = 10,
        include_images: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute RAG query and return response with associated images.

        Args:
            query_str: Query string
            top_k: Number of results to retrieve
            include_images: Whether to include associated images
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - response: Generated answer
                - sources: Source information
                - images: List of associated images with paths and descriptions
        """
        # 1. Pre-retrieval: Transform query
        query_bundles = self._pre_retrieval.transform(query_str)

        # 2. Retrieval with images
        all_text_nodes: List[NodeWithScore] = []
        all_images: List[ImageResult] = []

        for qb in query_bundles:
            result = self._hybrid_retriever.retrieve_with_images(
                qb.query_str,
                top_k=top_k,
                include_images=include_images,
            )
            all_text_nodes.extend(result.text_nodes)
            all_images.extend(result.images)

        # Deduplicate
        unique_nodes = self._deduplicate_nodes(all_text_nodes)
        unique_images = self._deduplicate_images(all_images)

        # 3. Post-retrieval: Rerank
        reranked_nodes = self._reranker.rerank(query_str, unique_nodes)

        # 4. Generation: Synthesize response
        response = self._synthesizer.synthesize(query_str, reranked_nodes[:top_k])

        # 5. Format response with sources and images
        return ResponseFormatter.format_with_sources_and_images(
            response,
            reranked_nodes[:top_k],
            unique_images,
        )

    def _deduplicate_images(self, images: List[ImageResult]) -> List[ImageResult]:
        """Remove duplicate images based on image_path.

        Args:
            images: List of ImageResult objects

        Returns:
            Deduplicated list keeping highest score
        """
        if not images:
            return []

        path_to_image: Dict[str, ImageResult] = {}
        for img in images:
            path = img.image_path
            if path not in path_to_image or img.score > path_to_image[path].score:
                path_to_image[path] = img

        return list(path_to_image.values())

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
