"""Runtime settings, read from the environment (prefix CLEARLINE_) or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLEARLINE_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"

    # Model provider: Cohere's API, a local Ollama server (on-prem / air-gapped), or a
    # deterministic offline fake used by the tests and CI evals.
    provider: Literal["fake", "cohere", "ollama"] = "fake"
    cohere_api_key: SecretStr | None = Field(default=None, validation_alias="COHERE_API_KEY")
    cohere_embed_model: str = "embed-v4.0"
    cohere_rerank_model: str = "rerank-v3.5"
    cohere_chat_model: str = "command-a-03-2025"
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.1"

    # Vector store: in-process (tests, demos) or Postgres with pgvector.
    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://clearline:clearline@localhost:5432/clearline"
    embedding_dim: int = 1024

    # Retrieval: candidates pulled from the store, then reranked down to the context size.
    candidates: int = 20
    context_chunks: int = 5
    min_rerank_score: float = 0.15
    agent_max_steps: int = 4

    # Auth: OIDC bearer tokens (Keycloak) in production; fixed dev users locally.
    auth_mode: Literal["dev", "oidc"] = "dev"
    oidc_issuer: str = "http://localhost:8081/realms/clearline"
    oidc_audience: str = "clearline-api"
    oidc_client_id: str = "clearline-cli"  # the public client the web page signs in with (PKCE)
    oidc_jwks_url: str | None = None
    clearance_claim: str = "clearance"
    groups_claim: str = "groups"

    redact_prompts: bool = True

    @model_validator(mode="after")
    def _safe_for_production(self) -> "Settings":
        if self.env == "production" and self.auth_mode == "dev":
            raise ValueError("CLEARLINE_AUTH_MODE=dev is not allowed when CLEARLINE_ENV=production")
        if self.provider == "cohere" and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when CLEARLINE_PROVIDER=cohere")
        return self

    @property
    def jwks_url(self) -> str:
        return self.oidc_jwks_url or f"{self.oidc_issuer.rstrip('/')}/protocol/openid-connect/certs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
