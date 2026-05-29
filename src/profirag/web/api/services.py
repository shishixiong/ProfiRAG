"""Business logic services wrapping existing ProfiRAG scripts."""

import os
import uuid
import json
import shutil
import time
import threading
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from profirag.ingestion.loaders import DocumentLoader
from profirag.ingestion.splitters import (
    TextSplitter,
    ChineseTextSplitter,
    MarkdownSplitter,
    extract_markdown_elements,
    build_sections,
    chunk_sections,
)
from profirag.ingestion.ast_splitter import ASTSplitter
from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.core import QueryBundle

from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.agent.react_agent import AgentFactory
from profirag.wiki.fetch_wiki_content import fetch_wiki_content
from profirag.ingestion.loaders import DocumentLoader as WikiDocumentLoader

TEMP_DIR = Path(tempfile.gettempdir()) / "profirag_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def generate_file_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _resolve_env_path(env_file: str) -> Path:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    return env_path


class FileService:
    """Handle file upload and storage."""

    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str) -> Dict[str, Any]:
        file_id = generate_file_id()
        file_type = Path(filename).suffix.lower()

        file_dir = TEMP_DIR / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        temp_path = file_dir / filename
        with open(temp_path, "wb") as f:
            f.write(file_content)

        return {
            "file_id": file_id,
            "filename": filename,
            "file_type": file_type,
            "size_bytes": len(file_content),
            "temp_path": str(temp_path),
        }

    @staticmethod
    def get_file_path(file_id: str) -> Optional[Path]:
        file_dir = TEMP_DIR / file_id
        if not file_dir.exists():
            return None
        files = [f for f in file_dir.iterdir() if f.is_file()]
        if files:
            return files[0]
        return None

    @staticmethod
    def cleanup_file(file_id: str) -> bool:
        file_dir = TEMP_DIR / file_id
        if file_dir.exists():
            shutil.rmtree(file_dir)
            return True
        return False


class PdfService:
    """Handle PDF to Markdown conversion."""

    @staticmethod
    def convert_pdf(
        file_path: str,
        pages: Optional[str] = None,
        write_images: bool = False,
        exclude_header_footer: bool = False,
        header_footer_min_occurrences: int = 3,
        extract_tables: bool = False,
    ) -> Dict[str, Any]:
        file_path = Path(file_path)
        output_dir = file_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_pages = None
        if pages:
            pdf_pages = PdfService._parse_pages(pages)

        loader = DocumentLoader(
            use_pymupdf4llm=True,
            pdf_write_images=write_images,
            pdf_image_path=str(output_dir / "images") if write_images else None,
            pdf_pages=pdf_pages,
            exclude_header_footer=exclude_header_footer,
            header_footer_auto_detect=True,
            header_footer_min_occurrences=header_footer_min_occurrences,
        )

        output_md_path = output_dir / (file_path.stem + ".md")
        saved_path, table_paths = loader.pdf_to_markdown_file(
            pdf_path=str(file_path),
            output_md_path=str(output_md_path),
            extract_tables=extract_tables,
        )

        with open(saved_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        return {
            "file_id": file_path.parent.name,
            "markdown_content": markdown_content,
            "markdown_path": str(saved_path),
            "table_files": [str(p) for p in table_paths] if table_paths else [],
            "image_files": [],
        }

    @staticmethod
    def _parse_pages(page_spec: str) -> list[int]:
        pages = []
        for part in page_spec.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        return [p - 1 for p in pages]

    @staticmethod
    def get_preview(file_id: str) -> Optional[Dict[str, Any]]:
        file_dir = TEMP_DIR / file_id
        if not file_dir.exists():
            return None

        output_dir = file_dir / "output"
        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            return None

        with open(md_files[0], "r", encoding="utf-8") as f:
            full_content = f.read()

        return {
            "file_id": file_id,
            "markdown_preview": full_content[:2000],
            "full_content_length": len(full_content),
            "tables_count": len(list(output_dir.glob("tables/*.md"))),
        }


class SplitService:
    """Handle document splitting."""

    @staticmethod
    def preview_split(
        file_path: str,
        splitter_type: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        ast_language: str = "python",
    ) -> Dict[str, Any]:
        file_path = Path(file_path)

        loader = DocumentLoader(encoding="utf-8")
        documents = loader.load_file(str(file_path))

        if not documents:
            return {"error": "Could not load document"}

        splitter = SplitService._create_splitter(
            splitter_type, chunk_size, chunk_overlap, ast_language
        )

        all_chunks: List[TextNode] = []
        for doc in documents:
            chunks = splitter.split_document(doc)
            source_file = doc.metadata.get("file_path", doc.metadata.get("file_name", file_path.name))
            for j, chunk in enumerate(chunks):
                chunk.metadata["source_file"] = source_file
                chunk.metadata["chunk_index"] = j
                chunk.metadata["total_chunks_in_doc"] = len(chunks)
            all_chunks.extend(chunks)

        chunks_preview = []
        for chunk in all_chunks[:20]:
            chunks_preview.append({
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "text_preview": chunk.text[:500] if len(chunk.text) > 500 else chunk.text,
                "metadata": {
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "source_file": chunk.metadata.get("source_file", ""),
                    "total_chunks_in_doc": chunk.metadata.get("total_chunks_in_doc", 0),
                    "header_path": chunk.metadata.get("header_path"),
                    "current_heading": chunk.metadata.get("current_heading"),
                    "has_code_block": chunk.metadata.get("has_code_block", False),
                    "has_table": chunk.metadata.get("has_table", False),
                    "has_images": chunk.metadata.get("has_images", False),
                    "char_count": len(chunk.text),
                }
            })

        return {
            "file_id": file_path.parent.name,
            "total_chunks": len(all_chunks),
            "chunks": chunks_preview,
            "summary": {
                "documents_loaded": len(documents),
                "avg_chunks_per_doc": len(all_chunks) / len(documents) if documents else 0,
                "total_chars": sum(len(c.text) for c in all_chunks),
                "avg_chunk_chars": sum(len(c.text) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
            }
        }

    @staticmethod
    def _create_splitter(
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        ast_language: str,
    ):
        if splitter_type == "ast":
            return ASTSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                language=ast_language,
            )
        elif splitter_type == "markdown":
            return MarkdownSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif splitter_type == "chinese":
            return ChineseTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            return TextSplitter(
                splitter_type=splitter_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    @staticmethod
    def download_chunks(
        file_id: str,
        output_format: str = "json",
    ) -> Optional[str]:
        cache_file = TEMP_DIR / file_id / "split_result.json"
        if not cache_file.exists():
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_dir = TEMP_DIR / file_id / "download"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            output_file = output_dir / "chunks.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif output_format == "jsonl":
            output_file = output_dir / "chunks.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in data.get("chunks", []):
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        else:
            output_file = output_dir / "chunks.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(data.get("chunks", [])):
                    f.write(f"=== Chunk {i} ===\n")
                    f.write(chunk.get("text_preview", chunk.get("text", "")))
                    f.write("\n\n")

        return str(output_file)


class ImportService:
    """Handle document import to vector store."""

    active_jobs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def start_import(
        file_paths: List[str],
        splitter_type: str = "markdown",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        ast_language: str = "python",
        index_mode: str = "hybrid",
        env_file: str = ".env",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job_id = generate_file_id()

        ImportService.active_jobs[job_id] = {
            "status": "pending",
            "documents_processed": 0,
            "documents_total": len(file_paths),
            "chunks_created": 0,
            "elapsed_seconds": 0,
            "start_time": time.time(),
            "error": None,
        }

        thread = threading.Thread(
            target=ImportService._run_import,
            args=(job_id, file_paths, splitter_type, chunk_size, chunk_overlap, ast_language, index_mode, env_file, metadata or {}),
        )
        thread.daemon = True
        thread.start()

        result = ImportService.active_jobs[job_id].copy()
        result["job_id"] = job_id
        return result

    @staticmethod
    def _run_import(
        job_id: str,
        file_paths: List[str],
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        ast_language: str,
        index_mode: str,
        env_file: str,
        metadata: Dict[str, Any],
    ):
        try:
            config_path = _resolve_env_path(env_file)
            config = load_config(str(config_path))

            config.storage.config["index_mode"] = index_mode
            config.chunking.splitter_type = splitter_type
            config.chunking.chunk_size = chunk_size
            config.chunking.chunk_overlap = chunk_overlap
            if ast_language:
                config.chunking.ast_language = ast_language

            ImportService.active_jobs[job_id]["status"] = "loading"
            pipeline = RAGPipeline(config)

            loader = DocumentLoader(encoding="utf-8")
            all_documents = []
            for fp in file_paths:
                docs = loader.load_file(fp)
                for doc in docs:
                    doc.metadata.update(metadata)
                all_documents.extend(docs)
                ImportService.active_jobs[job_id]["documents_processed"] += 1

            ImportService.active_jobs[job_id]["status"] = "running"

            start_time = time.time()
            result = pipeline.ingest_documents(all_documents)
            elapsed = time.time() - start_time

            stats = pipeline.get_stats()

            ImportService.active_jobs[job_id]["status"] = "completed"
            ImportService.active_jobs[job_id]["documents_processed"] = len(file_paths)
            ImportService.active_jobs[job_id]["chunks_created"] = len(result.get("text_node_ids", []))
            ImportService.active_jobs[job_id]["image_nodes_created"] = len(result.get("image_node_ids", []))
            ImportService.active_jobs[job_id]["elapsed_seconds"] = elapsed
            ImportService.active_jobs[job_id]["vector_store_count"] = stats["vector_store"]["count"]

        except Exception as e:
            ImportService.active_jobs[job_id]["status"] = "failed"
            ImportService.active_jobs[job_id]["error"] = str(e)

    @staticmethod
    def get_progress(job_id: str) -> Optional[Dict[str, Any]]:
        job = ImportService.active_jobs.get(job_id)
        if job:
            result = job.copy()
            result["job_id"] = job_id
            return result
        return None

    @staticmethod
    def get_stats(job_id: str) -> Optional[Dict[str, Any]]:
        job = ImportService.active_jobs.get(job_id)
        if job and job.get("status") == "completed":
            return {
                "job_id": job_id,
                "documents_loaded": job.get("documents_processed", 0),
                "documents_ingested": job.get("chunks_created", 0),
                "chunks_created": job.get("chunks_created", 0),
                "vector_store_count": job.get("vector_store_count", 0),
                "elapsed_seconds": job.get("elapsed_seconds", 0),
            }
        return None

    @staticmethod
    def start_wiki_import(
        wiki_url: str,
        splitter_type: str = "markdown",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        index_mode: str = "hybrid",
        env_file: str = ".env",
        metadata: Optional[Dict[str, Any]] = None,
        cookie: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_id = generate_file_id()

        ImportService.active_jobs[job_id] = {
            "status": "pending",
            "documents_processed": 0,
            "documents_total": 1,
            "chunks_created": 0,
            "elapsed_seconds": 0,
            "start_time": time.time(),
            "error": None,
        }

        thread = threading.Thread(
            target=ImportService._run_wiki_import,
            args=(job_id, wiki_url, splitter_type, chunk_size, chunk_overlap, index_mode, env_file, metadata or {}, cookie),
        )
        thread.daemon = True
        thread.start()

        result = ImportService.active_jobs[job_id].copy()
        result["job_id"] = job_id
        return result

    @staticmethod
    def _run_wiki_import(
        job_id: str,
        wiki_url: str,
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        index_mode: str,
        env_file: str,
        metadata: Dict[str, Any],
        cookie: Optional[str] = None,
    ):
        try:
            config_path = _resolve_env_path(env_file)
            load_dotenv(str(config_path), override=True)

            ImportService.active_jobs[job_id]["status"] = "fetching"
            wiki_data = fetch_wiki_content(wiki_url, cookie=cookie)

            if wiki_data is None:
                ImportService.active_jobs[job_id]["status"] = "failed"
                ImportService.active_jobs[job_id]["error"] = f"Failed to fetch wiki content from {wiki_url}"
                return

            config_path = _resolve_env_path(env_file)
            config = load_config(str(config_path))

            config.storage.config["index_mode"] = index_mode
            config.chunking.splitter_type = splitter_type
            config.chunking.chunk_size = chunk_size
            config.chunking.chunk_overlap = chunk_overlap

            ImportService.active_jobs[job_id]["status"] = "loading"
            pipeline = RAGPipeline(config)

            loader = WikiDocumentLoader()
            base_metadata = {
                "source": wiki_url,
                "title": wiki_data.get("title", ""),
                "document_type": wiki_data.get("document_type", ""),
                "author": wiki_data.get("author", ""),
                "view_count": wiki_data.get("viewCount", 0),
                "last_update_time": wiki_data.get("lastUpdateTime", ""),
                "loader": "wiki"
            }
            base_metadata.update(metadata)

            document = loader.load_text(wiki_data.get("content", ""), metadata=base_metadata)

            ImportService.active_jobs[job_id]["status"] = "running"

            start_time = time.time()
            result = pipeline.ingest_documents([document])
            elapsed = time.time() - start_time

            stats = pipeline.get_stats()

            ImportService.active_jobs[job_id]["status"] = "completed"
            ImportService.active_jobs[job_id]["documents_processed"] = 1
            ImportService.active_jobs[job_id]["chunks_created"] = len(result.get("text_node_ids", []))
            ImportService.active_jobs[job_id]["image_nodes_created"] = len(result.get("image_node_ids", []))
            ImportService.active_jobs[job_id]["elapsed_seconds"] = elapsed
            ImportService.active_jobs[job_id]["vector_store_count"] = stats["vector_store"]["count"]

        except Exception as e:
            ImportService.active_jobs[job_id]["status"] = "failed"
            ImportService.active_jobs[job_id]["error"] = str(e)


class ChatService:
    """Handle RAG chat queries."""

    active_sessions: Dict[str, Any] = {}

    @staticmethod
    def _create_session_id() -> str:
        import uuid
        return str(uuid.uuid4())[:8]

    @staticmethod
    def query(
        query_str: str,
        top_k: int = 10,
        mode: str = "pipeline",
        env_file: str = ".env",
        conversation: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if conversation and mode in ("agent", "plan", "react"):
            return ChatService.query_with_conversation(
                query_str=query_str,
                session_id=conversation.get("session_id"),
                mode=mode,
                top_k=top_k,
                env_file=env_file,
            )

        config_path = _resolve_env_path(env_file)
        load_dotenv(str(config_path), override=True)

        config = load_config()

        pipeline = RAGPipeline(config)

        if mode == "plan":
            result = pipeline.query_with_agent(query_str, mode="plan", auto_approve=True)
            return ChatService._format_agent_response(result)
        elif mode == "agent":
            result = pipeline.query_with_agent(query_str, mode="agent")
            return ChatService._format_agent_response(result)
        else:
            result = pipeline.query_with_images(query_str, top_k=top_k, include_images=True)
            result["query"] = query_str
            return result

    @staticmethod
    def query_with_conversation(
        query_str: str,
        session_id: Optional[str],
        mode: str,
        top_k: int,
        env_file: str,
    ) -> Dict[str, Any]:
        config_path = _resolve_env_path(env_file)
        load_dotenv(str(config_path), override=True)

        config = load_config()
        pipeline = RAGPipeline(config)

        if session_id and session_id in ChatService.active_sessions:
            conv_manager = ChatService.active_sessions[session_id]
        else:
            conv_manager = AgentFactory.create_conversation_agent(
                agent_type=mode,
                retriever=pipeline._hybrid_retriever,
                synthesizer=pipeline._synthesizer,
                llm=pipeline._llm,
                max_history_turns=config.agent.conversation_config.max_history_turns,
                keep_recent_turns=config.agent.conversation_config.keep_recent_turns,
                enable_auto_context=config.agent.conversation_config.auto_context,
                verbose=config.agent.verbose,
                markdown_base_path=config.agent.markdown_base_path,
                pre_retrieval=pipeline._pre_retrieval,
                reranker=pipeline._reranker,
            )
            ChatService.active_sessions[conv_manager.state.session_id] = conv_manager

        result = conv_manager.query(query_str)

        response = {
            "query": result.get("original_query", query_str),
            "response": result.get("response", ""),
            "source_nodes": ChatService._extract_sources(result),
            "images": [],
            "metadata": {"mode": mode},
            "conversation": {
                "session_id": conv_manager.state.session_id,
                "turn_count": result.get("conversation_turns", 1),
                "injected_context": result.get("injected_context", False),
                "reference_detected": result.get("reference_detected", False),
            },
        }
        return response

    @staticmethod
    def _extract_sources(result: Dict) -> List[Dict]:
        sources = []
        for src in result.get("sources", result.get("source_nodes", [])):
            if hasattr(src, "node"):
                sources.append({
                    "node_id": src.node.node_id,
                    "text": src.node.text[:300],
                    "score": src.score,
                    "source_file": src.node.metadata.get("source_file") or src.node.metadata.get("source") or src.node.metadata.get("source_path") or "",
                })
            else:
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300],
                    "score": src.get("score", 0.0),
                    "source_file": src.get("source_file") or src.get("source") or src.get("source_path") or "",
                })
        return sources

    @staticmethod
    def clear_session(session_id: str) -> bool:
        if session_id in ChatService.active_sessions:
            del ChatService.active_sessions[session_id]
            return True
        return False

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        if session_id in ChatService.active_sessions:
            conv_manager = ChatService.active_sessions[session_id]
            return {
                "session_id": session_id,
                "turn_count": conv_manager.state.total_turns(),
                "summary": conv_manager.state.summary,
                "created_at": conv_manager.state.created_at.isoformat(),
            }
        return None

    @staticmethod
    def _format_agent_response(result: Dict) -> Dict:
        response = result.get("response", "")

        sources = []
        if "sources" in result:
            for src in result["sources"]:
                source_file = src.get("source_file") or src.get("source") or src.get("source_path") or ""
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                    "score": src.get("score", 0.0),
                    "source_file": source_file,
                    "header_path": src.get("header_path"),
                })

        if "source_nodes" in result and not sources:
            for src in result["source_nodes"]:
                source_file = src.get("source_file") or src.get("source") or src.get("source_path") or ""
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                    "score": src.get("score", 0.0),
                    "source_file": source_file,
                    "header_path": src.get("header_path"),
                })

        return {
            "query": result.get("question", result.get("query", "")),
            "response": response,
            "source_nodes": sources,
            "images": [],
            "metadata": {"mode": result.get("mode", "agent")},
        }


class SearchService:
    """Pure retrieval service without LLM synthesis."""

    @staticmethod
    def query(
        query_str: str,
        top_k: int = 20,
        rerank: bool = True,
        use_pre_retrieval: bool = False,
        retrieve_mode: str = "hybrid",
        env_file: str = ".env",
    ) -> Dict[str, Any]:
        config_path = _resolve_env_path(env_file)
        load_dotenv(str(config_path), override=True)
        config = load_config()

        pipeline = RAGPipeline(config)

        if rerank and pipeline._reranker.enabled:
            pipeline._reranker.set_top_n(top_k)

        if use_pre_retrieval:
            query_bundles = pipeline._pre_retrieval.transform(query_str)
        else:
            query_bundles = [QueryBundle(query_str=query_str)]

        all_nodes = []
        for qb in query_bundles:
            nodes = pipeline._hybrid_retriever.retrieve(qb.query_str, top_k=top_k * 2, retrieve_mode=retrieve_mode)
            all_nodes.extend(nodes)

        unique_nodes = pipeline._deduplicate_nodes(all_nodes)

        if rerank:
            unique_nodes = pipeline._reranker.rerank(query_str, unique_nodes)
            unique_nodes = pipeline._deduplicate_nodes(unique_nodes)

        unique_nodes = pipeline._expand_tables_in_nodes(unique_nodes)
        return SearchService._format_results(query_str, unique_nodes[:top_k], rerank)

    @staticmethod
    def _format_results(
        query_str: str,
        nodes: List[NodeWithScore],
        reranked: bool
    ) -> Dict[str, Any]:
        file_counts: Dict[str, int] = {}
        chunks: List[Dict[str, Any]] = []

        for node in nodes:
            source_file = (node.node.metadata.get('source', '') or node.node.metadata.get('source_file', '')
                   or node.node.metadata.get('source_path', ''))
            if not source_file:
                source_file = "unknown"
            file_counts[source_file] = file_counts.get(source_file, 0) + 1

            text = node.node.text
            chunks.append({
                "chunk_id": node.node.node_id,
                "heading": node.node.metadata.get("title") or node.node.metadata.get("current_heading"),
                "score": node.score,
                "text_preview": text[:200] if len(text) > 200 else text,
                "full_text": text,
                "source_file": source_file,
                "header_path": node.node.metadata.get("header_path"),
            })

        files = [{"filename": f, "chunk_count": c} for f, c in file_counts.items()]

        return {
            "query": query_str,
            "total_results": len(chunks),
            "files": files,
            "chunks": chunks,
            "metadata": {"reranked": reranked},
        }
