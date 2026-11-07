"""An in-process store: brute-force cosine search over chunks, for tests, evals and demos."""

from __future__ import annotations

import math
import threading

from ..models import Chunk, Document, Hit, Principal


class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._lock = threading.Lock()

    def upsert(self, document: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            self._documents[document.id] = document
            self._chunks[document.id] = list(chunks)

    def search(self, embedding: list[float], principal: Principal, k: int) -> list[Hit]:
        with self._lock:
            visible = [c for doc in self._documents.values() if principal.can_read(doc.classification, doc.groups)
                       for c in self._chunks[doc.id]]
        hits = [Hit(chunk=c, score=_cosine(embedding, c.embedding)) for c in visible]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def documents(self, principal: Principal) -> list[tuple[Document, int]]:
        with self._lock:
            return [(d, len(self._chunks[d.id])) for d in self._documents.values()
                    if principal.can_read(d.classification, d.groups)]

    def chunks_of(self, document_id: str, principal: Principal) -> list[Chunk]:
        with self._lock:
            doc = self._documents.get(document_id)
            if not doc or not principal.can_read(doc.classification, doc.groups):
                return []
            return list(self._chunks[document_id])

    def delete(self, document_id: str) -> bool:
        with self._lock:
            self._chunks.pop(document_id, None)
            return self._documents.pop(document_id, None) is not None

    def ping(self) -> bool:
        return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
