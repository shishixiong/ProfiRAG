"""Business logic services wrapping existing ProfiRAG scripts."""

import os
import sys
import uuid
import json
import shutil
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
from llama_index.core.schema import TextNode, Document

# Import ingest function
from profirag.config.settings import load_config
from profirag.pipeline.rag_pipeline import RAGPipeline
from profirag.agent.react_agent import AgentFactory

# Temp directory for uploaded files
TEMP_DIR = PROJECT_ROOT / "web" / "api" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def generate_file_id() -> str:
    """Generate unique file ID."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class FileService:
    """Handle file upload and storage."""

    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str) -> Dict[str, Any]:
        """Save uploaded file to temp directory."""
        file_id = generate_file_id()
        file_type = Path(filename).suffix.lower()

        # Create subdirectory for this file
        file_dir = TEMP_DIR / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        # Save file
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
        """Get file path by ID (returns the original uploaded file, not output files)."""
        file_dir = TEMP_DIR / file_id
        if not file_dir.exists():
            return None
        # Find the actual file in the directory (exclude subdirectories like 'output')
        files = [f for f in file_dir.iterdir() if f.is_file()]
        if files:
            return files[0]
        return None

    @staticmethod
    def cleanup_file(file_id: str) -> bool:
        """Remove file and its directory."""
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
        """Convert PDF to Markdown using DocumentLoader."""
        file_path = Path(file_path)
        output_dir = file_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse pages if provided
        pdf_pages = None
        if pages:
            pdf_pages = PdfService._parse_pages(pages)

        # Initialize loader
        loader = DocumentLoader(
            use_pymupdf4llm=True,
            pdf_write_images=write_images,
            pdf_image_path=str(output_dir / "images") if write_images else None,
            pdf_pages=pdf_pages,
            exclude_header_footer=exclude_header_footer,
            header_footer_auto_detect=True,
            header_footer_min_occurrences=header_footer_min_occurrences,
        )

        # Convert
        output_md_path = output_dir / (file_path.stem + ".md")
        saved_path, table_paths = loader.pdf_to_markdown_file(
            pdf_path=str(file_path),
            output_md_path=str(output_md_path),
            extract_tables=extract_tables,
        )

        # Read markdown content
        with open(saved_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        return {
            "file_id": file_path.parent.name,
            "markdown_content": markdown_content,
            "markdown_path": str(saved_path),
            "table_files": [str(p) for p in table_paths] if table_paths else [],
            "image_files": [],  # TODO: collect image paths
        }

    @staticmethod
    def _parse_pages(page_spec: str) -> list[int]:
        """Parse page specification like '1-5,10,15-20' to list of page numbers."""
        pages = []
        for part in page_spec.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        # Convert to 0-indexed
        return [p - 1 for p in pages]

    @staticmethod
    def get_preview(file_id: str) -> Optional[Dict[str, Any]]:
        """Get conversion preview."""
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
        """Preview document split result."""
        file_path = Path(file_path)

        # Load document
        loader = DocumentLoader(encoding="utf-8")
        documents = loader.load_file(str(file_path))

        if not documents:
            return {"error": "Could not load document"}

        # Create splitter
        splitter = SplitService._create_splitter(
            splitter_type, chunk_size, chunk_overlap, ast_language
        )

        # Split documents
        all_chunks: List[TextNode] = []
        for doc in documents:
            chunks = splitter.split_document(doc)
            # Add metadata
            source_file = doc.metadata.get("file_path", doc.metadata.get("file_name", file_path.name))
            for j, chunk in enumerate(chunks):
                chunk.metadata["source_file"] = source_file
                chunk.metadata["chunk_index"] = j
                chunk.metadata["total_chunks_in_doc"] = len(chunks)
            all_chunks.extend(chunks)

        # Build preview response
        chunks_preview = []
        for chunk in all_chunks[:20]:  # Limit to first 20 for preview
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
        """Create appropriate splitter instance."""
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
        """Download full chunk results."""
        # Read cached split result
        cache_file = TEMP_DIR / file_id / "split_result.json"
        if not cache_file.exists():
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Generate output file
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
        else:  # txt
            output_file = output_dir / "chunks.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(data.get("chunks", [])):
                    f.write(f"=== Chunk {i} ===\n")
                    f.write(chunk.get("text_preview", chunk.get("text", "")))
                    f.write("\n\n")

        return str(output_file)


class ImportService:
    """Handle document import to vector store."""

    # Store active import jobs
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
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Start import process asynchronously."""
        job_id = generate_file_id()

        # Initialize job status
        ImportService.active_jobs[job_id] = {
            "status": "pending",
            "documents_processed": 0,
            "documents_total": len(file_paths),
            "chunks_created": 0,
            "elapsed_seconds": 0,
            "start_time": time.time(),
            "error": None,
        }

        # Start import in background thread
        thread = threading.Thread(
            target=ImportService._run_import,
            args=(job_id, file_paths, splitter_type, chunk_size, chunk_overlap, ast_language, index_mode, env_file, metadata or {}),
        )
        thread.daemon = True
        thread.start()

        # Return immediately with job_id
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
        """Run import in background thread."""
        try:
            # Load configuration
            config_path = PROJECT_ROOT / env_file
            config = load_config(str(config_path))

            # Apply settings
            config.storage.config["index_mode"] = index_mode
            config.chunking.splitter_type = splitter_type
            config.chunking.chunk_size = chunk_size
            config.chunking.chunk_overlap = chunk_overlap
            if ast_language:
                config.chunking.ast_language = ast_language

            # Initialize pipeline
            ImportService.active_jobs[job_id]["status"] = "loading"
            pipeline = RAGPipeline(config)

            # Load documents
            loader = DocumentLoader(encoding="utf-8")
            all_documents = []
            for fp in file_paths:
                docs = loader.load_file(fp)
                # Apply custom metadata to each document
                for doc in docs:
                    doc.metadata.update(metadata)
                all_documents.extend(docs)
                ImportService.active_jobs[job_id]["documents_processed"] += 1

            ImportService.active_jobs[job_id]["status"] = "running"

            # Ingest documents
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
        """Get import job progress."""
        job = ImportService.active_jobs.get(job_id)
        if job:
            result = job.copy()
            result["job_id"] = job_id
            return result
        return None

    @staticmethod
    def get_stats(job_id: str) -> Optional[Dict[str, Any]]:
        """Get final import statistics."""
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


class ChatService:
    """Handle RAG chat queries."""

    # Session storage (in-memory for now)
    active_sessions: Dict[str, Any] = {}

    @staticmethod
    def _create_session_id() -> str:
        """Generate new session ID."""
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
        """Execute RAG query and return response with images.

        Args:
            query_str: User query string
            top_k: Number of results to retrieve
            mode: Query mode (pipeline, agent, plan)
            env_file: Path to environment config file
            conversation: Optional conversation context dict

        Returns:
            Dictionary with response, sources, and images
        """
        # Handle conversation mode
        if conversation and mode in ("agent", "plan", "react"):
            return ChatService.query_with_conversation(
                query_str=query_str,
                session_id=conversation.get("session_id"),
                mode=mode,
                top_k=top_k,
                env_file=env_file,
            )

        # Load environment variables from .env file
        config_path = PROJECT_ROOT / env_file
        load_dotenv(str(config_path), override=True)

        # Load configuration (now reads from environment variables)
        config = load_config()

        # Initialize pipeline
        pipeline = RAGPipeline(config)

        # Route based on mode
        if mode == "plan":
            result = pipeline.query_with_agent(query_str, mode="plan", auto_approve=True)
            return ChatService._format_agent_response(result)
        elif mode == "agent":
            result = pipeline.query_with_agent(query_str, mode="agent")
            return ChatService._format_agent_response(result)
        else:
            # Pipeline mode - default behavior
            result = pipeline.query_with_images(query_str, top_k=top_k, include_images=True)
            # Add query field to match ChatResponse schema
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
        """Execute query with conversation context.

        Args:
            query_str: User query
            session_id: Existing session ID (None for new session)
            mode: Agent mode (react/plan)
            top_k: Retrieval count
            env_file: Config file path

        Returns:
            Response with conversation info
        """
        # Load environment variables from .env file
        config_path = PROJECT_ROOT / env_file
        load_dotenv(str(config_path), override=True)

        # Load configuration
        config = load_config()
        pipeline = RAGPipeline(config)

        # Get or create conversation manager
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

        # Execute query
        result = conv_manager.query(query_str)

        # Format response
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
        """Extract sources from result."""
        sources = []
        for src in result.get("sources", result.get("source_nodes", [])):
            if hasattr(src, "node"):
                sources.append({
                    "node_id": src.node.node_id,
                    "text": src.node.text[:300],
                    "score": src.score,
                    "source_file": src.node.metadata.get("source_file"),
                })
            else:
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300],
                    "score": src.get("score", 0.0),
                    "source_file": src.get("source_file"),
                })
        return sources

    @staticmethod
    def clear_session(session_id: str) -> bool:
        """Clear conversation session."""
        if session_id in ChatService.active_sessions:
            del ChatService.active_sessions[session_id]
            return True
        return False

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """Get session info."""
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
        """Normalize agent response to ChatResponse format.

        Args:
            result: Agent query result dictionary

        Returns:
            Normalized dictionary with response, source_nodes, images, metadata
        """
        # Extract response text
        response = result.get("response", "")

        # Extract sources from agent result
        sources = []
        if "sources" in result:
            for src in result["sources"]:
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                    "score": src.get("score", 0.0),
                    "source_file": src.get("source_file"),
                    "header_path": src.get("header_path"),
                })

        # Also check source_nodes key (alternative format)
        if "source_nodes" in result and not sources:
            for src in result["source_nodes"]:
                sources.append({
                    "node_id": src.get("node_id", ""),
                    "text": src.get("text", "")[:300] if len(src.get("text", "")) > 300 else src.get("text", ""),
                    "score": src.get("score", 0.0),
                    "source_file": src.get("source_file"),
                    "header_path": src.get("header_path"),
                })

        return {
            "query": result.get("question", result.get("query", "")),
            "response": response,
            "source_nodes": sources,
            "images": [],  # Agent mode doesn't return images currently
            "metadata": {"mode": result.get("mode", "agent")},
        }