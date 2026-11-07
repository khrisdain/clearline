"""Everything wired together: provider, store, retrieval, agent and the audit log."""

from __future__ import annotations

from .agent import Agent, Retriever
from .audit import AuditLog, MemoryAuditLog, PostgresAuditLog
from .config import Settings
from .ingest import chunk_document, document_id
from .models import Answer, Classification, Document, DocumentIn, DocumentOut, Principal
from .providers import Provider, build_provider
from .redact import redact
from .store import Store, build_store


class Clearline:
    def __init__(self, settings: Settings, provider: Provider | None = None, store: Store | None = None,
                 audit: AuditLog | None = None) -> None:
        self.settings = settings
        self.provider = provider or build_provider(settings)
        self.store = store or build_store(settings)
        if audit is None:
            pool = getattr(self.store, "pool", None)
            audit = PostgresAuditLog(pool) if pool is not None else MemoryAuditLog()
        self.audit = audit
        self.retriever = Retriever(self.provider, self.store, settings)
        self.agent = Agent(self.provider, self.retriever, self.store, settings)

    def ingest(self, document: Document, actor: str = "system") -> DocumentOut:
        chunks = chunk_document(document)
        vectors = self.provider.embed([f"{c.title} — {c.section}\n{c.text}" for c in chunks], "document")
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        self.store.upsert(document, chunks)
        self.audit.append(actor, "ingest", {"document_id": document.id, "title": document.title,
                                            "classification": document.classification.label, "chunks": len(chunks)})
        return DocumentOut(id=document.id, title=document.title, classification=document.classification.label,
                           groups=sorted(document.groups), chunks=len(chunks))

    def ingest_input(self, body: DocumentIn, actor: str) -> DocumentOut:
        document = Document(
            id=document_id(body.title, body.source or body.title), title=body.title, text=body.text,
            classification=Classification.parse(body.classification),
            groups=frozenset(g.strip().lower() for g in body.groups if g.strip()), source=body.source,
        )
        return self.ingest(document, actor)

    def ask(self, question: str, principal: Principal) -> Answer:
        answer = self.agent.run(question, principal)
        self.audit.append(principal.subject, "ask", {
            "question": redact(question),
            "cited_documents": sorted({c.document_id for c in answer.citations}),
            "searches": sum(1 for s in answer.steps if s.tool == "search_documents"),
            "grounded": answer.grounded, "latency_ms": answer.latency_ms,
        })
        return answer
