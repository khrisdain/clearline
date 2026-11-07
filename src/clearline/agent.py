"""Retrieval and the question-answering agent.

Retrieval is two-stage: a wide vector search inside the store (already filtered to what the
asker may read), then a reranker narrows it to the few passages worth showing the model.

The agent is a small tool-use loop. The model can search (more than once, with its own
rephrasings) and open a whole document, then must answer from the numbered passages it was
shown, citing them as [n]. Citations to passages it was never shown are dropped, and an answer
with no valid citation is marked ungrounded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .config import Settings
from .models import Answer, Chunk, Citation, Hit, Message, Principal, Step, ToolCall, ToolSpec
from .providers import Provider
from .redact import redact
from .store import Store

SYSTEM_PROMPT = """You are Clearline, an assistant for the staff of a public-sector organisation.

Rules:
- Answer only from passages returned by your tools. Call search_documents before answering; search again with \
different words if the first results miss the point.
- Cite every claim with the bracketed number of the passage it came from, like [2]. Only cite numbers you were given.
- If the passages do not contain the answer, say you couldn't find it in the documents the user has access to. Never \
guess, never use outside knowledge, and never speculate about documents you were not shown.
- Be concise: a direct answer first, then the key details."""

TOOLS = [
    ToolSpec(
        name="search_documents",
        description="Search the organisation's documents the user may read. Returns numbered passages.",
        parameters={"type": "object", "properties": {"query": {"type": "string", "description": "What to look for."}},
                    "required": ["query"]},
    ),
    ToolSpec(
        name="read_document",
        description="Read a whole document by its document_id (from search results), as numbered passages.",
        parameters={"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
    ),
]

MARKER = re.compile(r"\[(\d+)\]")


class Retriever:
    def __init__(self, provider: Provider, store: Store, settings: Settings) -> None:
        self.provider, self.store, self.settings = provider, store, settings

    def search(self, query: str, principal: Principal, top_n: int | None = None) -> list[Hit]:
        top_n = top_n or self.settings.context_chunks
        [vector] = self.provider.embed([query], "query")
        candidates = self.store.search(vector, principal, self.settings.candidates)
        if not candidates:
            return []
        ranked = self.provider.rerank(query, [passage(h.chunk) for h in candidates], top_n)
        return [Hit(candidates[i].chunk, score) for i, score in ranked if score >= self.settings.min_rerank_score]


def passage(chunk: Chunk) -> str:
    return f"{chunk.title} — {chunk.section}\n{chunk.text}"


@dataclass
class Sources:
    """Passages shown to the model so far, numbered in the order it first saw them."""

    by_marker: dict[int, Chunk] = field(default_factory=dict)
    marker_of: dict[str, int] = field(default_factory=dict)

    def add(self, chunk: Chunk) -> int:
        if chunk.id not in self.marker_of:
            marker = len(self.by_marker) + 1
            self.marker_of[chunk.id], self.by_marker[marker] = marker, chunk
        return self.marker_of[chunk.id]

    def results(self, chunks: list[Chunk]) -> str:
        return json.dumps({"results": [
            {"marker": self.add(c), "document_id": c.document_id, "title": c.title, "section": c.section,
             "classification": c.classification.label, "text": c.text} for c in chunks]})


class Agent:
    def __init__(self, provider: Provider, retriever: Retriever, store: Store, settings: Settings) -> None:
        self.provider, self.retriever, self.store, self.settings = provider, retriever, store, settings

    def run(self, question: str, principal: Principal) -> Answer:
        started = time.perf_counter()
        prompt = redact(question) if self.settings.redact_prompts else question
        messages = [Message("system", SYSTEM_PROMPT), Message("user", prompt)]
        sources, steps = Sources(), []
        reply = Message("assistant")
        for _ in range(self.settings.agent_max_steps):
            reply = self.provider.chat(messages, TOOLS)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                chunks = self._run_tool(call, principal)
                steps.append(Step(tool=call.name, arguments=call.arguments, results=len(chunks)))
                messages.append(Message("tool", sources.results(chunks), tool_call_id=call.id))
        else:
            # Out of steps while still calling tools: ask for an answer from what it has. The tools stay
            # declared (some APIs reject a history with tool calls but no tools); any further calls are ignored.
            messages.append(Message("user", "Answer now from the passages you have, with citations. Do not call tools."))
            reply = self.provider.chat(messages, TOOLS)

        text, citations = cite(reply.content, sources)
        return Answer(answer=text, citations=citations, steps=steps, grounded=bool(citations),
                      latency_ms=round((time.perf_counter() - started) * 1000))

    def _run_tool(self, call: ToolCall, principal: Principal) -> list[Chunk]:
        if call.name == "search_documents":
            query = str(call.arguments.get("query", "")).strip()
            return [h.chunk for h in self.retriever.search(query, principal)] if query else []
        if call.name == "read_document":
            # Unknown and forbidden documents look the same: no hint that a document exists.
            return self.store.chunks_of(str(call.arguments.get("document_id", "")), principal)[:8]
        return []


def cite(text: str, sources: Sources) -> tuple[str, list[Citation]]:
    """Keep citations to passages the model was shown, drop the rest, list each source once."""
    used: list[int] = []

    def keep(match: re.Match[str]) -> str:
        marker = int(match.group(1))
        if marker not in sources.by_marker:
            return ""
        if marker not in used:
            used.append(marker)
        return match.group(0)

    cleaned = re.sub(r"\s+([.,;:])", r"\1", MARKER.sub(keep, text)).strip()
    citations = []
    for marker in used:
        chunk = sources.by_marker[marker]
        snippet = chunk.text if len(chunk.text) <= 280 else chunk.text[:277].rsplit(" ", 1)[0] + "…"
        citations.append(Citation(marker=marker, document_id=chunk.document_id, title=chunk.title, section=chunk.section,
                                  classification=chunk.classification.label, snippet=snippet))
    return cleaned, citations
