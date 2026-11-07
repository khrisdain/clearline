"""Model providers behind one interface: Cohere, a local Ollama server, or an offline fake."""

from ..config import Settings
from .base import Provider


def build_provider(settings: Settings) -> Provider:
    if settings.provider == "cohere":
        from .cohere import CohereProvider

        return CohereProvider(settings)
    if settings.provider == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(settings)
    from .fake import FakeProvider

    return FakeProvider(dim=settings.embedding_dim)


__all__ = ["Provider", "build_provider"]
