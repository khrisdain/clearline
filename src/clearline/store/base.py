from typing import Protocol

from ..models import Chunk, Document, Hit, Principal


class Store(Protocol):
    def upsert(self, document: Document, chunks: list[Chunk]) -> None:
        """Replace a document and all its chunks."""
        ...

    def search(self, embedding: list[float], principal: Principal, k: int) -> list[Hit]:
        """Nearest chunks the principal may read. Access is filtered before ranking, so
        restricted text never leaves the store for someone without access."""
        ...

    def documents(self, principal: Principal) -> list[tuple[Document, int]]:
        """Documents the principal may read, with their chunk counts."""
        ...

    def chunks_of(self, document_id: str, principal: Principal) -> list[Chunk]:
        """A document's chunks in order, or nothing if the principal may not read it."""
        ...

    def delete(self, document_id: str) -> bool: ...

    def ping(self) -> bool: ...
