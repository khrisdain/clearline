"""An in-process store: brute-force cosine search and BM25 over chunks, for tests, evals and demos."""

from __future__ import annotations

import math
import threading
from collections import Counter

from ..models import Chunk, Document, Hit, Principal
from ..text import terms

BM25_K1, BM25_B = 1.2, 0.75


class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._terms: dict[str, Counter[str]] = {}  # chunk id -> term counts, for BM25
        self._lock = threading.Lock()

    def upsert(self, document: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            for old in self._chunks.get(document.id, []):
                self._terms.pop(old.id, None)
            self._documents[document.id] = document
            self._chunks[document.id] = list(chunks)
            for chunk in chunks:
                self._terms[chunk.id] = Counter(terms(f"{chunk.title} {chunk.section} {chunk.text}"))

    def search(self, embedding: list[float], principal: Principal, k: int) -> list[Hit]:
        with self._lock:
            visible = [c for doc in self._documents.values() if principal.can_read(doc.classification, doc.groups)
                       for c in self._chunks[doc.id]]
        hits = [Hit(chunk=c, score=_cosine(embedding, c.embedding)) for c in visible]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def keyword_search(self, query: str, principal: Principal, k: int) -> list[Hit]:
        """BM25 over the chunks the principal may read. Corpus statistics (document frequency, average
        length) come from those chunks only, so scores reveal nothing about documents they can't see."""
        wanted = set(terms(query))
        with self._lock:
            visible = [c for doc in self._documents.values() if principal.can_read(doc.classification, doc.groups)
                       for c in self._chunks[doc.id]]
            counts = [self._terms[c.id] for c in visible]
        if not wanted or not visible:
            return []
        n = len(visible)
        lengths = [sum(c.values()) for c in counts]
        average = sum(lengths) / n or 1.0
        df = {t: sum(1 for c in counts if t in c) for t in wanted}
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in wanted if df[t]}
        hits = []
        for chunk, tf, length in zip(visible, counts, lengths, strict=True):
            score = sum(idf[t] * tf[t] * (BM25_K1 + 1) / (tf[t] + BM25_K1 * (1 - BM25_B + BM25_B * length / average))
                        for t in idf if t in tf)
            if score > 0:
                hits.append(Hit(chunk=chunk, score=score))
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
            for old in self._chunks.pop(document_id, []):
                self._terms.pop(old.id, None)
            return self._documents.pop(document_id, None) is not None

    def ping(self) -> bool:
        return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
