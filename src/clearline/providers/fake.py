"""A deterministic, offline provider so the tests and CI evals run without an API key.

Embeddings are hashed bags of (lightly stemmed) words, reranking is query-term coverage, and the
"model" follows a fixed policy: search once, then answer with the evidence sentence that best
matches the question, cited; or decline when nothing returned covers most of the question. It exercises every code path the real
models do (tool calls, tool results, citations, refusals) without pretending to be smart.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from ..models import Message, ToolCall, ToolSpec
from ..text import terms
from .base import EmbedKind


class FakeProvider:
    name = "fake"
    model = "offline stand-in"
    embed_model = "hashed bag of words"
    rerank_model = "query-term coverage"
    native_citations = False

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for term in terms(text):
            digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        wanted = set(terms(query))
        if not wanted:
            return []
        scored = [(i, len(wanted & set(terms(doc))) / len(wanted)) for i, doc in enumerate(documents)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]

    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> Message:
        question = next(m.content for m in reversed(messages) if m.role == "user")
        results = [m for m in messages if m.role == "tool"]
        if not results:
            return Message(role="assistant", content="I'll search the documents.",
                           tool_calls=[ToolCall(id="call_0", name="search_documents", arguments={"query": question})])
        evidence = json.loads(results[-1].content).get("results", [])
        if not evidence:
            return Message(role="assistant", content="I couldn't find this in the documents you have access to.")
        wanted = set(terms(question))
        marker, sentence = max(
            ((item["marker"], sentence) for item in evidence for sentence in re.split(r"(?<=[.!?])\s+", item["text"])),
            key=lambda pair: len(wanted & set(terms(pair[1]))),
        )
        # Stand-in for the model's judgement: decline unless the evidence covers most of the question.
        passage_terms = set(terms(next(item["title"] + " " + item["section"] + " " + item["text"]
                                        for item in evidence if item["marker"] == marker)))
        if len(wanted & passage_terms) < 0.6 * len(wanted):
            return Message(role="assistant", content="I couldn't find this in the documents you have access to.")
        return Message(role="assistant", content=f"{sentence.strip()} [{marker}]")
