"""A local Ollama server, for on-prem and air-gapped installs where no data may leave the network.

Ollama has no rerank endpoint, so reranking falls back to cosine similarity of embeddings.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from ..config import Settings
from ..models import Message, ToolCall, ToolSpec
from .base import EmbedKind


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.ollama_url, timeout=120)

    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        # nomic-embed-text expects task prefixes for asymmetric search.
        prefix = "search_query: " if kind == "query" else "search_document: "
        response = self.client.post("/api/embed", json={"model": self.settings.ollama_embed_model,
                                                        "input": [prefix + t for t in texts]})
        response.raise_for_status()
        return response.json()["embeddings"]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        q, *docs = self.embed([query], "query") + self.embed(documents, "document")
        scored = sorted(((i, max(0.0, cosine(q, d))) for i, d in enumerate(docs)), key=lambda p: p[1], reverse=True)
        return scored[:top_n]

    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> Message:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_chat_model, "stream": False, "options": {"temperature": 0.1},
            "messages": [to_ollama(m) for m in messages],
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                      for t in tools],
        }
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        message = response.json()["message"]
        calls = [ToolCall(id=f"call_{i}", name=c["function"]["name"], arguments=c["function"].get("arguments") or {})
                 for i, c in enumerate(message.get("tool_calls") or [])]
        return Message(role="assistant", content=message.get("content", ""), tool_calls=calls)


def to_ollama(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls]
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
