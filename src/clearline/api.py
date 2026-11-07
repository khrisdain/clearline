"""The HTTP API (FastAPI)."""

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .auth import Authenticator
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
