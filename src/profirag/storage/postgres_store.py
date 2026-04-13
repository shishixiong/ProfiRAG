"""PostgreSQL/pgvector vector store implementation"""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
from psycopg2.extras import execute_values

from .base import BaseVectorStore
from .registry import StorageRegistry


@StorageRegistry.register("postgres")
class PostgresStore(BaseVectorStore):
    """PostgreSQL/pgvector vector store backend implementation.

    Uses PostgreSQL with pgvector extension for vector storage.
    Suitable for production deployments with existing PostgreSQL infrastructure.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "profirag_vectors",
        dimension: int = 1536,
        embed_dim: int = 1536,
        schema_name: str = "public",
        **kwargs
    ):
        """Initialize PostgreSQL vector store.

        Args:
            connection_string: PostgreSQL connection string
                (e.g., "postgresql://user:pass@host:port/db")
            table_name: Name of the vector table
            dimension: Vector dimension (default 1536 for OpenAI)
            embed_dim: Embedding dimension (alias for dimension)
            schema_name: Schema name (default "public")
            **kwargs: Additional arguments
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema_name = schema_name
        self.dimension = dimension or embed_dim

        # Initialize connection
        self._ensure_extension_exists()

        # Create LlamaIndex PGVectorStore
        self._vector_store = PGVectorStore(
            connection_string=connection_string,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=self.dimension,
            **kwargs
        )

        # Initialize node store for full node data
        self._init_node_table()

    @property
    def client(self) -> Any:
        """Get the underlying PGVectorStore."""
        return self._vector_store

    def _ensure_extension_exists(self) -> None:
        """Ensure pgvector extension is installed."""
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
        finally:
            conn.close()

    def _init_node_table(self) -> None:
        """Initialize node content storage table."""
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                # Create table for full node content
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name}_nodes (
                        node_id TEXT PRIMARY KEY,
                        text TEXT,
                        metadata JSONB,
                        ref_doc_id TEXT
                    );
                """)
                # Create table for ref_doc info
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.table_name}_docs (
                        ref_doc_id TEXT PRIMARY KEY,
                        node_ids JSONB
                    );
                """)
                conn.commit()
        finally:
            conn.close()

    def add(self, nodes: List[TextNode], **kwargs) -> List[str]:
        """Add nodes to PostgreSQL storage.

        Args:
            nodes: List of TextNode objects to add
            **kwargs: Additional arguments

        Returns:
            List of node IDs that were added
        """
        if not nodes:
            return []

        # Add to vector store
        ids = self._vector_store.add(nodes, **kwargs)

        # Store full node content
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                # Insert node content
                node_data = [
                    (
                        node.node_id,
                        node.text,
                        node.metadata or {},
                        node.ref_doc_id
                    )
                    for node in nodes
                ]
                execute_values(
                    cur,
                    f"""
                        INSERT INTO {self.schema_name}.{self.table_name}_nodes
                        (node_id, text, metadata, ref_doc_id)
                        VALUES %s
                        ON CONFLICT (node_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        ref_doc_id = EXCLUDED.ref_doc_id;
                    """,
                    node_data
                )

                # Update ref_doc info
                ref_docs = {}
                for node in nodes:
                    if node.ref_doc_id:
                        if node.ref_doc_id not in ref_docs:
                            ref_docs[node.ref_doc_id] = []
                        ref_docs[node.ref_doc_id].append(node.node_id)

                for ref_doc_id, node_ids_list in ref_docs.items():
                    cur.execute(f"""
                        INSERT INTO {self.schema_name}.{self.table_name}_docs
                        (ref_doc_id, node_ids)
                        VALUES (%s, %s)
                        ON CONFLICT (ref_doc_id) DO UPDATE SET
                        node_ids = %s || (
                            SELECT node_ids FROM {self.schema_name}.{self.table_name}_docs
                            WHERE ref_doc_id = %s
                        );
                    """, (ref_doc_id, node_ids_list, node_ids_list, ref_doc_id))

                conn.commit()
        finally:
            conn.close()

        return ids

    def delete(self, ref_doc_id: Optional[str] = None, node_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """Delete nodes from PostgreSQL storage.

        Args:
            ref_doc_id: Document ID to delete all associated nodes
            node_ids: List of specific node IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful
        """
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                if ref_doc_id:
                    # Get node IDs for this doc
                    cur.execute(f"""
                        SELECT node_ids FROM {self.schema_name}.{self.table_name}_docs
                        WHERE ref_doc_id = %s;
                    """, (ref_doc_id,))
                    result = cur.fetchone()
                    if result:
                        node_ids_to_delete = result[0]
                        # Delete nodes
                        cur.execute(f"""
                            DELETE FROM {self.schema_name}.{self.table_name}_nodes
                            WHERE node_id = ANY(%s);
                        """, (node_ids_to_delete,))
                        # Delete doc entry
                        cur.execute(f"""
                            DELETE FROM {self.schema_name}.{self.table_name}_docs
                            WHERE ref_doc_id = %s;
                        """, (ref_doc_id,))
                        # Delete from vector store
                        self._vector_store.delete(ref_doc_id=ref_doc_id, **kwargs)
                elif node_ids:
                    # Delete specific nodes
                    cur.execute(f"""
                        DELETE FROM {self.schema_name}.{self.table_name}_nodes
                        WHERE node_id = ANY(%s);
                    """, (node_ids,))
                    self._vector_store.delete(node_ids=node_ids, **kwargs)
                else:
                    # Clear all
                    self.clear()

                conn.commit()
        finally:
            conn.close()

        return True

    def query(self, query: QueryBundle, similarity_top_k: int = 10, **kwargs) -> List[NodeWithScore]:
        """Query PostgreSQL for similar nodes.

        Args:
            query: QueryBundle containing query text and/or embedding
            similarity_top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of NodeWithScore objects
        """
        results = self._vector_store.query(query, similarity_top_k=similarity_top_k, **kwargs)

        # Fetch full node content
        if results:
            node_ids = [r.node.node_id for r in results]
            conn = psycopg2.connect(self.connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT node_id, text, metadata, ref_doc_id
                        FROM {self.schema_name}.{self.table_name}_nodes
                        WHERE node_id = ANY(%s);
                    """, (node_ids,))
                    node_data = {row[0]: row for row in cur.fetchall()}
            finally:
                conn.close()

            # Reconstruct full nodes
            full_results = []
            for node_with_score in results:
                node_id = node_with_score.node.node_id
                if node_id in node_data:
                    _, text, metadata, ref_doc_id = node_data[node_id]
                    full_node = TextNode(
                        id_=node_id,
                        text=text,
                        metadata=metadata or {},
                        ref_doc_id=ref_doc_id,
                    )
                    full_results.append(NodeWithScore(node=full_node, score=node_with_score.score))
                else:
                    full_results.append(node_with_score)

            return full_results

        return results

    def get_node(self, node_id: str) -> Optional[TextNode]:
        """Get a specific node by ID from PostgreSQL.

        Args:
            node_id: The node ID

        Returns:
            TextNode if found, None otherwise
        """
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT node_id, text, metadata, ref_doc_id
                    FROM {self.schema_name}.{self.table_name}_nodes
                    WHERE node_id = %s;
                """, (node_id,))
                result = cur.fetchone()
                if result:
                    return TextNode(
                        id_=result[0],
                        text=result[1],
                        metadata=result[2] or {},
                        ref_doc_id=result[3],
                    )
        finally:
            conn.close()

        return None

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info from PostgreSQL.

        Args:
            ref_doc_id: Reference document ID

        Returns:
            RefDocInfo if found, None otherwise
        """
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT node_ids FROM {self.schema_name}.{self.table_name}_docs
                    WHERE ref_doc_id = %s;
                """, (ref_doc_id,))
                result = cur.fetchone()
                if result:
                    return RefDocInfo(node_ids=result[0])
        finally:
            conn.close()

        return None

    def persist(self, persist_path: Optional[str] = None, **kwargs) -> None:
        """Persist PostgreSQL storage.

        Note: PostgreSQL handles persistence automatically. This method
        is provided for interface compatibility.

        Args:
            persist_path: Path to persist to (ignored for PostgreSQL)
            **kwargs: Additional arguments
        """
        # PostgreSQL handles persistence internally
        pass

    def count(self) -> int:
        """Count nodes in PostgreSQL storage.

        Returns:
            Number of nodes
        """
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT COUNT(*) FROM {self.schema_name}.{self.table_name}_nodes;
                """)
                return cur.fetchone()[0]
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear all nodes from PostgreSQL storage."""
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    TRUNCATE TABLE {self.schema_name}.{self.table_name}_nodes;
                    TRUNCATE TABLE {self.schema_name}.{self.table_name}_docs;
                """)
                # Also clear the vector table
                cur.execute(f"""
                    TRUNCATE TABLE {self.schema_name}.{self.table_name};
                """)
                conn.commit()
        finally:
            conn.close()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PostgresStore":
        """Create PostgresStore from configuration.

        Args:
            config: Configuration dictionary with keys:
                - connection_string: PostgreSQL connection string
                - host: Database host (default "localhost")
                - port: Database port (default 5432)
                - database: Database name (default "profirag")
                - user: Database user
                - password: Database password
                - table_name: Table name (default "profirag_vectors")
                - dimension: Vector dimension (default 1536)
                - schema_name: Schema name (default "public")

        Returns:
            PostgresStore instance
        """
        # Build connection string if components provided
        if "connection_string" in config:
            connection_string = config["connection_string"]
        else:
            host = config.get("host", "localhost")
            port = config.get("port", 5432)
            database = config.get("database", "profirag")
            user = config.get("user", "postgres")
            password = config.get("password", "")
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        return cls(
            connection_string=connection_string,
            table_name=config.get("table_name", "profirag_vectors"),
            dimension=config.get("dimension", 1536),
            schema_name=config.get("schema_name", "public"),
            **config.get("store_options", {})
        )