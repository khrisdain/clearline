"""Where chunks live. Every read takes the asking Principal and filters inside the query."""

from ..config import Settings
from .base import Store


def build_store(settings: Settings) -> Store:
    if settings.store == "postgres":
        from .postgres import PostgresStore

        return PostgresStore(settings.database_url, settings.embedding_dim)
    from .memory import MemoryStore

    return MemoryStore()


__all__ = ["Store", "build_store"]
