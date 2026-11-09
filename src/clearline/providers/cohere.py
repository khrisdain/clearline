"""Cohere's API: Embed for retrieval, Rerank to sharpen it, Command for the tool-using agent.

Search results go back to Command as documents (one per passage, id "passage-<n>"), so its replies
carry native span-level citations: which characters of the answer each passage supports. Those are
mapped back to passage numbers here; the agent then checks them like any other citation."""

from __future__ import annotations

import json
import re
from typing import Any

import cohere

from ..config import Settings
from ..models import Message, Span, ToolCall, ToolSpec
from .base import EmbedKind

PASSAGE_ID = re.compile(r"passage-(\d+)")


class CohereProvider:
    name = "cohere"
    native_citations = True

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        self.model = settings.cohere_chat_model
        self.embed_model = f"{settings.cohere_embed_model} ({settings.embedding_dim}d)"
        self.rerank_model = settings.cohere_rerank_model
        key = settings.cohere_api_key.get_secret_value() if settings.cohere_api_key else None
        self.client = client or cohere.ClientV2(api_key=key, timeout=60)

    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):  # the API's batch limit
            response = self.client.embed(
                model=self.settings.cohere_embed_model, texts=texts[start : start + 96],
                input_type="search_query" if kind == "query" else "search_document",
                embedding_types=["float"], output_dimension=self.settings.embedding_dim,
            )
            vectors.extend(list(v) for v in response.embeddings.float_ or [])
        return vectors

    def embed_typed(self, texts: list[str], kind: EmbedKind, dim: int) -> dict[str, Any]:
        """float, int8 and packed-binary embeddings at one Matryoshka size, in one call per batch (for the
        retrieval study, which compares them)."""
        import numpy as np

        out: dict[str, list[Any]] = {"float": [], "int8": [], "ubinary": []}
        for start in range(0, len(texts), 96):
            response = self.client.embed(
                model=self.settings.cohere_embed_model, texts=texts[start : start + 96],
                input_type="search_query" if kind == "query" else "search_document",
                embedding_types=["float", "int8", "ubinary"], output_dimension=dim,
            )
            out["float"].extend(response.embeddings.float_ or [])
            out["int8"].extend(response.embeddings.int8 or [])
            out["ubinary"].extend(response.embeddings.ubinary or [])
        return {"float": np.asarray(out["float"], dtype=np.float32), "int8": np.asarray(out["int8"], dtype=np.int8),
                "ubinary": np.asarray(out["ubinary"], dtype=np.uint8)}

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        response = self.client.rerank(
            model=self.settings.cohere_rerank_model, query=query, documents=documents, top_n=min(top_n, len(documents)),
        )
        return [(r.index, float(r.relevance_score)) for r in response.results]

    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> Message:
        tool_specs: list[Any] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        history: list[Any] = [to_cohere(m) for m in messages]
        response = self.client.chat(model=self.settings.cohere_chat_model, messages=history, tools=tool_specs or None,
                                    temperature=0.1)
        message = response.message
        calls = [ToolCall(id=str(c.id), name=str(c.function.name), arguments=json.loads(c.function.arguments or "{}"))
                 for c in (message.tool_calls or []) if c.function is not None]
        text = "".join(getattr(item, "text", "") or "" for item in (message.content or []))
        spans = [span for c in (getattr(message, "citations", None) or []) if (span := to_span(c, text))]
        return Message(role="assistant", content=text or (message.tool_plan or ""), tool_calls=calls, spans=spans)


def to_span(citation: Any, text: str) -> Span | None:
    """A Cohere citation -> a Span over the answer, with the passage numbers it cites."""
    markers: list[int] = []
    for source in getattr(citation, "sources", None) or []:
        found = PASSAGE_ID.search(str(getattr(source, "id", "") or ""))
        payload = getattr(source, "tool_output", None) or getattr(source, "document", None) or {}
        if found:
            markers.append(int(found.group(1)))
        elif isinstance(payload, dict) and str(payload.get("marker", "")).isdigit():
            markers.append(int(payload["marker"]))
    start, end = int(getattr(citation, "start", 0) or 0), int(getattr(citation, "end", 0) or 0)
    if not markers or not 0 <= start < end <= len(text):
        return None
    return Span(start=start, end=end, text=text[start:end], markers=list(dict.fromkeys(markers)))


def to_cohere(message: Message) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant", "tool_plan": message.content or "Searching the documents.",
            "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                           for c in message.tool_calls],
        }
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": as_documents(message.content)}
    return {"role": message.role, "content": message.content}


def as_documents(content: str) -> str | list[dict[str, Any]]:
    """Tool results as Cohere documents, so Command can cite each passage by id."""
    try:
        results = json.loads(content).get("results", [])
    except (ValueError, AttributeError):
        return content
    if not results:
        return content
    return [{"type": "document", "document": {"id": f"passage-{r['marker']}", "data": {
        "title": r["title"], "section": r["section"], "classification": r["classification"], "text": r["text"],
        "document_id": r["document_id"]}}} for r in results]
