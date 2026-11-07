"""Cohere's API: Embed for retrieval, Rerank to sharpen it, Command for the tool-using agent."""

from __future__ import annotations

import json
from typing import Any

import cohere

from ..config import Settings
from ..models import Message, ToolCall, ToolSpec
from .base import EmbedKind


class CohereProvider:
    name = "cohere"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
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
        text = "".join(getattr(item, "text", "") for item in (message.content or []))
        return Message(role="assistant", content=text or (message.tool_plan or ""), tool_calls=calls)


def to_cohere(message: Message) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant", "tool_plan": message.content or "Searching the documents.",
            "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                           for c in message.tool_calls],
        }
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    return {"role": message.role, "content": message.content}
