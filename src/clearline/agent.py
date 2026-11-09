"""Retrieval and the question-answering agent.

Retrieval is two-stage. First a wide search inside the store, already filtered to what the asker
may read: dense (embeddings), keyword (BM25 / full text) or both, merged with reciprocal rank
fusion. Then a reranker narrows the candidates to the few passages worth showing the model.

The agent is a small tool-use loop. The model can search (more than once, with its own
rephrasings) and open a whole document, then must answer from the numbered passages it was
shown. Cohere's Command returns span-level citations natively (which characters of the answer
each passage supports); other models cite with [n] markers, which are turned into the same spans.
Either way, citations to passages the model was never shown are dropped, and an answer with no
valid citation is marked ungrounded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from .config import Settings
from .models import Answer, Chunk, Citation, Hit, Message, Principal, Span, Step, ToolCall, ToolSpec
from .providers import Provider
from .redact import redact
from .store import Store

CITE_WITH_MARKERS = "- Cite every claim with the bracketed number of the passage it came from, like [2]. Only cite numbers \
you were given."
CITE_NATIVELY = "- Base every claim on a passage your tools returned; citations are attached automatically, so do not \
write [n] markers yourself."

SYSTEM_PROMPT = """You are Clearline, an assistant for the staff of a public-sector organisation.

Rules:
- Answer only from passages returned by your tools. Call search_documents before answering; search again with \
different words if the first results miss the point.
{cite_rule}
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


@dataclass
class Retrieval:
    """What one search did, stage by stage, for the agent trace and the evals."""

    hits: list[Hit]
    dense: int = 0
    keyword: int = 0
    fused: int = 0
    reranked: bool = False

    def describe(self) -> dict[str, int | bool]:
        return {"dense": self.dense, "keyword": self.keyword, "fused": self.fused, "reranked": self.reranked,
                "kept": len(self.hits)}


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists by summing 1 / (k + rank). Rank-based, so a cosine similarity and a BM25
    score never have to be put on the same scale (Cormack et al., 2009)."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.chunk.id, hit.chunk)
    return [Hit(chunks[i], score) for i, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


class Retriever:
    def __init__(self, provider: Provider, store: Store, settings: Settings) -> None:
        self.provider, self.store, self.settings = provider, store, settings

    def search(self, query: str, principal: Principal, top_n: int | None = None) -> list[Hit]:
        return self.retrieve(query, principal, top_n).hits

    def retrieve(self, query: str, principal: Principal, top_n: int | None = None) -> Retrieval:
        top_n = top_n or self.settings.context_chunks
        mode, k = self.settings.retrieval, self.settings.candidates
        dense = keyword = []
        if mode in ("dense", "hybrid"):
            [vector] = self.provider.embed([query], "query")
            dense = self.store.search(vector, principal, k)
        if mode in ("keyword", "hybrid"):
            keyword = self.store.keyword_search(query, principal, k)
        candidates = reciprocal_rank_fusion([r for r in (dense, keyword) if r], self.settings.rrf_k)[:k]
        trace = Retrieval([], dense=len(dense), keyword=len(keyword), fused=len(candidates))
        if not candidates:
            return trace
        if not self.settings.rerank:
            trace.hits = candidates[:top_n]
            return trace
        ranked = self.provider.rerank(query, [passage(h.chunk) for h in candidates], top_n)
        trace.hits = [Hit(candidates[i].chunk, score) for i, score in ranked if score >= self.settings.min_rerank_score]
        trace.reranked = True
        return trace


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
        native = getattr(self.provider, "native_citations", False)
        system = SYSTEM_PROMPT.format(cite_rule=CITE_NATIVELY if native else CITE_WITH_MARKERS)
        messages = [Message("system", system), Message("user", prompt)]
        sources, steps = Sources(), []
        reply = Message("assistant")
        for _ in range(self.settings.agent_max_steps):
            reply = self.provider.chat(messages, TOOLS)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                chunks, retrieval = self._run_tool(call, principal)
                steps.append(Step(tool=call.name, arguments=call.arguments, results=len(chunks), retrieval=retrieval))
                messages.append(Message("tool", sources.results(chunks), tool_call_id=call.id))
        else:
            # Out of steps while still calling tools: ask for an answer from what it has. The tools stay
            # declared (some APIs reject a history with tool calls but no tools); any further calls are ignored.
            messages.append(Message("user", "Answer now from the passages you have, with citations. Do not call tools."))
            reply = self.provider.chat(messages, TOOLS)

        mode: Literal["native", "markers"] = "native" if reply.spans else "markers"
        if reply.spans:
            text = reply.content.rstrip()  # Cohere's offsets index the raw text, so no lstrip
            spans = checked_spans(reply.spans, sources, text)
        else:
            text, spans = cite(reply.content, sources)
        citations = citations_for(spans, sources)
        return Answer(answer=text, citations=citations, steps=steps, grounded=bool(citations),
                      latency_ms=round((time.perf_counter() - started) * 1000), spans=spans,
                      coverage=coverage(text, spans), citation_mode=mode, model=getattr(self.provider, "model", ""))

    def _run_tool(self, call: ToolCall, principal: Principal) -> tuple[list[Chunk], dict[str, int | bool] | None]:
        if call.name == "search_documents":
            query = str(call.arguments.get("query", "")).strip()
            if not query:
                return [], None
            retrieval = self.retriever.retrieve(query, principal)
            return [h.chunk for h in retrieval.hits], retrieval.describe()
        if call.name == "read_document":
            # Unknown and forbidden documents look the same: no hint that a document exists.
            return self.store.chunks_of(str(call.arguments.get("document_id", "")), principal)[:8], None
        return [], None


SENTENCE_END = re.compile(r"[.!?;:\n]\s+")


def cite(text: str, sources: Sources) -> tuple[str, list[Span]]:
    """[n] markers -> spans. First drop markers for passages the model was never shown; then each
    remaining marker supports the text from the start of its sentence (or the previous marker) up to it."""
    def keep(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in sources.by_marker else ""

    cleaned = re.sub(r"\s+([.,;:])", r"\1", MARKER.sub(keep, text)).strip()
    spans: list[Span] = []
    previous_end = 0
    for match in MARKER.finditer(cleaned):
        marker = int(match.group(1))
        if spans and not cleaned[previous_end:match.start()].strip():  # "[1][2]": both support the same span
            if marker not in spans[-1].markers:
                spans[-1].markers.append(marker)
            previous_end = match.end()
            continue
        end = len(cleaned[:match.start()].rstrip())
        start = previous_end
        for boundary in SENTENCE_END.finditer(cleaned, previous_end, end):
            start = boundary.end()
        start += len(cleaned[start:end]) - len(cleaned[start:end].lstrip())
        if start < end:
            spans.append(Span(start=start, end=end, text=cleaned[start:end], markers=[marker]))
        previous_end = match.end()
    return cleaned, spans


def checked_spans(spans: list[Span], sources: Sources, text: str) -> list[Span]:
    """Native citations: keep only markers for passages the model was actually shown."""
    kept = []
    for span in spans:
        markers = [m for m in dict.fromkeys(span.markers) if m in sources.by_marker]
        if markers and span.end <= len(text):
            kept.append(span.model_copy(update={"markers": markers}))
    return kept


def citations_for(spans: list[Span], sources: Sources) -> list[Citation]:
    """Each cited passage once, in order of first citation."""
    used = list(dict.fromkeys(m for span in spans for m in span.markers))
    citations = []
    for marker in used:
        chunk = sources.by_marker[marker]
        snippet = chunk.text if len(chunk.text) <= 280 else chunk.text[:277].rsplit(" ", 1)[0] + "…"
        citations.append(Citation(marker=marker, document_id=chunk.document_id, title=chunk.title, section=chunk.section,
                                  classification=chunk.classification.label, snippet=snippet))
    return citations


def coverage(text: str, spans: list[Span]) -> float:
    """Share of the answer's visible characters (not spaces, not [n] markers) inside a cited span."""
    visible = [i for i in range(len(text)) if not text[i].isspace()]
    for match in MARKER.finditer(text):
        visible = [i for i in visible if not match.start() <= i < match.end()]
    if not visible:
        return 0.0
    covered = {i for span in spans for i in range(span.start, span.end)}
    return round(sum(1 for i in visible if i in covered) / len(visible), 3)
