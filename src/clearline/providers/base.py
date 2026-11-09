"""The three things Clearline needs from a model provider."""

from typing import Literal, Protocol

from ..models import Message, ToolSpec

EmbedKind = Literal["document", "query"]


class Provider(Protocol):
    name: str
    model: str  # the chat model answering questions
    embed_model: str
    rerank_model: str
    native_citations: bool  # True if chat replies carry span citations (Message.spans) instead of [n] markers

    def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        """One vector per text. Queries and documents may be embedded differently."""
        ...

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """(index into `documents`, relevance 0..1), best first, at most `top_n`."""
        ...

    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> Message:
        """The assistant's next turn: text, tool calls, or both."""
        ...
