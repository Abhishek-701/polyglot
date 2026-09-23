"""pgvector-backed corpus store. See SPEC.md Section 8.7.

Verified against installed pgvector server 0.8.3 + psycopg==3.3.6 +
pgvector-python==0.5.0: register_vector(conn) lets a numpy float32 array be
passed/returned directly as the `vector` column type. HNSW index chosen per
SPEC.md 8.7 ("pgvector: HNSW index... Document why pgvector suffices at this
scale in an ADR") — see docs/decisions/002-pgvector-store.md.
"""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from polyglot.core.types import Passage

EMBEDDING_DIM = 1024  # BAAI/bge-m3


class RetrievalStore:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                section TEXT NOT NULL,
                chunk_index INT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )

    def document_hash(self, doc_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        doc_id: str,
        source_url: str,
        content_hash: str,
        chunks: list[tuple[str, int, str, np.ndarray]],
    ) -> None:
        """chunks: list of (section, chunk_index, text, embedding)."""
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO documents (doc_id, source_url, content_hash) VALUES (%s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET "
                "source_url = EXCLUDED.source_url, content_hash = EXCLUDED.content_hash",
                (doc_id, source_url, content_hash),
            )
            self._conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            for section, chunk_index, text, embedding in chunks:
                chunk_id = f"{doc_id}#{chunk_index}"
                self._conn.execute(
                    "INSERT INTO chunks (id, doc_id, section, chunk_index, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (chunk_id, doc_id, section, chunk_index, text, embedding),
                )

    def search(self, query_embedding: np.ndarray, k: int) -> list[Passage]:
        embedding = np.asarray(query_embedding, dtype=np.float32)
        rows = self._conn.execute(
            "SELECT chunks.id, chunks.doc_id, chunks.text, documents.source_url, "
            "1 - (chunks.embedding <=> %s) AS score "
            "FROM chunks JOIN documents ON chunks.doc_id = documents.doc_id "
            "ORDER BY chunks.embedding <=> %s LIMIT %s",
            (embedding, embedding, k),
        ).fetchall()
        return [
            Passage(id=row[0], doc_id=row[1], text=row[2], source_url=row[3], score=float(row[4]))
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def close(self) -> None:
        self._conn.close()
