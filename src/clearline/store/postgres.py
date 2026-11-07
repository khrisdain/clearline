"""Postgres with pgvector: an HNSW index for cosine search, access labels on every chunk row.

The access filter is part of the same SQL statement as the vector search, so the database never
returns a chunk the caller may not read. With a selective filter, HNSW can come back short of
`k`; pgvector's iterative index scans (0.8+) keep scanning until enough rows pass the filter.
"""

from __future__ import annotations

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg_pool import ConnectionPool

from ..models import Chunk, Classification, Document, Hit, Principal


def access(table: str = "") -> str:
    """The read check as SQL: cleared to the row's level, and in one of its groups if it names any."""
    t = f"{table}." if table else ""
    return f"{t}classification <= %(clearance)s AND (cardinality({t}groups) = 0 OR {t}groups && %(groups)s::text[])"


ACCESS = access()


def schema(dim: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS documents (
      id text PRIMARY KEY,
      title text NOT NULL,
      text text NOT NULL,
      classification smallint NOT NULL,
      groups text[] NOT NULL DEFAULT '{{}}',
      source text NOT NULL DEFAULT '',
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS chunks (
      id text PRIMARY KEY,
      document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      title text NOT NULL,
      section text NOT NULL,
      text text NOT NULL,
      classification smallint NOT NULL,
      groups text[] NOT NULL DEFAULT '{{}}',
      position int NOT NULL,
      embedding vector({dim}) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id, position);
    CREATE INDEX IF NOT EXISTS chunks_groups ON chunks USING gin (groups);
    """


def _configure(conn: Connection) -> None:
    register_vector(conn)


class PostgresStore:
    def __init__(self, url: str, dim: int, min_size: int = 1, max_size: int = 10) -> None:
        self.dim = dim
        with Connection.connect(url, autocommit=True) as conn:  # create the extension before registering the type
            conn.execute(schema(dim))
        self.pool = ConnectionPool(url, min_size=min_size, max_size=max_size, configure=_configure, open=True)

    def upsert(self, document: Document, chunks: list[Chunk]) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO documents (id, title, text, classification, groups, source) VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, text = EXCLUDED.text,
                     classification = EXCLUDED.classification, groups = EXCLUDED.groups, source = EXCLUDED.source,
                     updated_at = now()""",
                (document.id, document.title, document.text, int(document.classification), sorted(document.groups), document.source),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document.id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (id, document_id, title, section, text, classification, groups, position, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [(c.id, c.document_id, c.title, c.section, c.text, int(c.classification), sorted(c.groups), c.position,
                      np.array(c.embedding, dtype=np.float32)) for c in chunks],
                )

    def search(self, embedding: list[float], principal: Principal, k: int) -> list[Hit]:
        params = {"q": np.array(embedding, dtype=np.float32), "k": k, **_access(principal)}
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SET LOCAL hnsw.ef_search = 100")
            try:
                with conn.transaction():
                    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            except Exception:  # noqa: S110 - pgvector < 0.8 has no iterative scans; plain HNSW still works
                pass
            rows = conn.execute(
                f"""SELECT id, document_id, title, section, text, classification, groups, position,
                           1 - (embedding <=> %(q)s) AS score
                    FROM chunks WHERE {ACCESS} ORDER BY embedding <=> %(q)s LIMIT %(k)s""",  # noqa: S608 - constant SQL
                params,
            ).fetchall()
        return [Hit(chunk=_chunk(row), score=float(row[8])) for row in rows]

    def documents(self, principal: Principal) -> list[tuple[Document, int]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT d.id, d.title, d.classification, d.groups, d.source, count(c.id)
                    FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                    WHERE {access('d')}
                    GROUP BY d.id ORDER BY d.title""",  # noqa: S608 - constant SQL
                _access(principal),
            ).fetchall()
        return [(Document(id=r[0], title=r[1], text="", classification=Classification(r[2]), groups=frozenset(r[3]),
                          source=r[4]), int(r[5])) for r in rows]

    def chunks_of(self, document_id: str, principal: Principal) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""SELECT id, document_id, title, section, text, classification, groups, position
                    FROM chunks WHERE document_id = %(id)s AND {ACCESS} ORDER BY position""",  # noqa: S608 - constant SQL
                {"id": document_id, **_access(principal)},
            ).fetchall()
        return [_chunk(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM documents WHERE id = %s", (document_id,)).rowcount > 0

    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False


def _access(principal: Principal) -> dict[str, object]:
    return {"clearance": int(principal.clearance), "groups": sorted(principal.groups)}


def _chunk(row: tuple) -> Chunk:
    return Chunk(id=row[0], document_id=row[1], title=row[2], section=row[3], text=row[4],
                 classification=Classification(row[5]), groups=frozenset(row[6]), position=row[7])
