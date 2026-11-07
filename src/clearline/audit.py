"""A tamper-evident audit log: who asked what, and which documents the answer drew on.

Each entry stores the hash of the one before it, so editing or deleting any past entry breaks
every hash after it, and `verify()` says where.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from psycopg.types.json import Jsonb

GENESIS = "0" * 64


@dataclass
class AuditEntry:
    seq: int
    at: str
    subject: str
    action: str
    detail: dict[str, Any]
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "hash"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AuditLog(Protocol):
    def append(self, subject: str, action: str, detail: dict[str, Any]) -> AuditEntry: ...

    def entries(self, limit: int = 100) -> list[AuditEntry]: ...

    def verify(self) -> int | None:
        """None if the chain is intact, else the sequence number of the first bad entry."""
        ...


def check_chain(entries: list[AuditEntry]) -> int | None:
    previous = GENESIS
    for entry in entries:
        if entry.prev_hash != previous or entry.hash != entry.compute_hash():
            return entry.seq
        previous = entry.hash
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class MemoryAuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()

    def append(self, subject: str, action: str, detail: dict[str, Any]) -> AuditEntry:
        with self._lock:
            prev = self._entries[-1].hash if self._entries else GENESIS
            entry = AuditEntry(seq=len(self._entries) + 1, at=_now(), subject=subject, action=action, detail=detail, prev_hash=prev)
            entry.hash = entry.compute_hash()
            self._entries.append(entry)
            return entry

    def entries(self, limit: int = 100) -> list[AuditEntry]:
        with self._lock:
            return list(reversed(self._entries[-limit:]))

    def verify(self) -> int | None:
        with self._lock:
            return check_chain(list(self._entries))


class PostgresAuditLog:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_log (
      seq bigint PRIMARY KEY,
      at text NOT NULL,
      subject text NOT NULL,
      action text NOT NULL,
      detail jsonb NOT NULL,
      prev_hash text NOT NULL,
      hash text NOT NULL
    );
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        with pool.connection() as conn:
            conn.execute(self.SCHEMA)

    def append(self, subject: str, action: str, detail: dict[str, Any]) -> AuditEntry:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(7311)")  # one writer at a time keeps the chain linear
            last = conn.execute("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            entry = AuditEntry(seq=(last[0] + 1) if last else 1, at=_now(), subject=subject, action=action, detail=detail,
                               prev_hash=last[1] if last else GENESIS)
            entry.hash = entry.compute_hash()
            conn.execute("INSERT INTO audit_log VALUES (%s, %s, %s, %s, %s, %s, %s)",
                         (entry.seq, entry.at, entry.subject, entry.action, Jsonb(entry.detail), entry.prev_hash, entry.hash))
            return entry

    def entries(self, limit: int = 100) -> list[AuditEntry]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY seq DESC LIMIT %s", (limit,)).fetchall()
        return [AuditEntry(*row) for row in rows]

    def verify(self) -> int | None:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY seq").fetchall()
        return check_chain([AuditEntry(*row) for row in rows])
