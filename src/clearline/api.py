"""The HTTP API (FastAPI)."""

from dataclasses import asdict
from importlib.resources import files
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from .auth import DEV_USERS, Authenticator
from .config import Settings, get_settings
from .models import Answer, AskIn, DocumentIn, DocumentOut, Principal
from .service import Clearline


def create_app(settings: Settings | None = None, service: Clearline | None = None,
               authenticator: Authenticator | None = None) -> FastAPI:
    settings = settings or get_settings()
    clearline = service or Clearline(settings)
    authenticate = authenticator or Authenticator(settings)
    app = FastAPI(title="Clearline", version=__version__,
                  description="Ask questions of your organisation's documents; answers cite only what you may read.")
    app.state.clearline = clearline
    User = Annotated[Principal, Depends(authenticate)]

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        ready = clearline.store.ping()
        return JSONResponse({"store": "ok" if ready else "unavailable"}, status_code=200 if ready else 503)

    @app.get("/", include_in_schema=False)
    def page() -> HTMLResponse:
        return HTMLResponse(files("clearline").joinpath("ui/index.html").read_text(encoding="utf-8"))

    @app.get("/ui/config", include_in_schema=False)
    def ui_config() -> dict[str, Any]:
        """What the web page needs to sign people in. Nothing secret: the client is public and uses PKCE."""
        provider = clearline.provider
        models = {"provider": provider.name, "chat": provider.model, "embed": provider.embed_model,
                  "rerank": provider.rerank_model if settings.rerank else None, "retrieval": settings.retrieval,
                  "citations": "native" if provider.native_citations else "markers"}
        if settings.auth_mode == "dev":
            return {"auth_mode": "dev", "dev_users": sorted(DEV_USERS), "models": models}
        return {"auth_mode": "oidc", "issuer": settings.oidc_issuer, "client_id": settings.oidc_client_id, "models": models}

    @app.get("/v1/me")
    def me(user: User) -> dict[str, Any]:
        return {"subject": user.subject, "clearance": user.clearance.label, "groups": sorted(user.groups),
                "admin": user.is_admin, "auditor": user.is_auditor}

    @app.post("/v1/ask", response_model=Answer)
    def ask(body: AskIn, user: User) -> Answer:
        return clearline.ask(body.question, user)

    @app.get("/v1/documents", response_model=list[DocumentOut])
    def documents(user: User) -> list[DocumentOut]:
        return [DocumentOut(id=d.id, title=d.title, classification=d.classification.label, groups=sorted(d.groups),
                            chunks=n) for d, n in clearline.store.documents(user)]

    @app.post("/v1/documents", response_model=DocumentOut, status_code=201)
    def add_document(body: DocumentIn, user: User) -> DocumentOut:
        if not user.is_admin:
            raise HTTPException(403, "Only administrators can add documents")
        try:
            return clearline.ingest_input(body, user.subject)
        except KeyError as error:
            raise HTTPException(422, f"Unknown classification: {body.classification}") from error

    @app.delete("/v1/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, user: User) -> None:
        if not user.is_admin:
            raise HTTPException(403, "Only administrators can remove documents")
        if not clearline.store.delete(document_id):
            raise HTTPException(404, "No such document")
        clearline.audit.append(user.subject, "delete", {"document_id": document_id})

    @app.get("/v1/audit")
    def audit(user: User, limit: int = 100) -> list[dict[str, Any]]:
        if not user.is_auditor:
            raise HTTPException(403, "Only auditors can read the audit log")
        return [asdict(e) for e in clearline.audit.entries(min(max(limit, 1), 1000))]

    @app.get("/v1/audit/verify")
    def verify(user: User) -> dict[str, Any]:
        if not user.is_auditor:
            raise HTTPException(403, "Only auditors can read the audit log")
        broken = clearline.audit.verify()
        return {"intact": broken is None, "first_broken_entry": broken}

    return app
