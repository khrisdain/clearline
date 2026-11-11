from fastapi.testclient import TestClient

from clearline.api import create_app
from clearline.auth import Authenticator
from clearline.config import Settings
from clearline.service import Clearline


def ask(client, question, user="ana"):
    return client.post("/v1/ask", json={"question": question}, headers={"X-Dev-User": user})


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200


def test_ask_returns_a_cited_answer(client):
    body = ask(client, "How many days a week can I telework?").json()
    assert body["grounded"] and "two days a week" in body["answer"]
    assert body["citations"][0]["title"] == "Telework Policy"


def test_people_see_different_document_lists(client):
    def titles(user):
        return {d["title"] for d in client.get("/v1/documents", headers={"X-Dev-User": user}).json()}

    assert "2027 Operating Budget Forecast" in titles("ben")
    assert "2027 Operating Budget Forecast" not in titles("ana")
    assert "Cyber Incident Response Runbook" in titles("chloe")
    assert "Cyber Incident Response Runbook" not in titles("ben")


def test_only_admins_add_documents_and_new_ones_are_searchable(client):
    doc = {"title": "Snow Clearing Schedule", "classification": "unclassified", "groups": [],
           "text": "# Priorities\nArterial roads are cleared within 8 hours after a snowfall ends."}
    assert client.post("/v1/documents", json=doc, headers={"X-Dev-User": "ana"}).status_code == 403
    created = client.post("/v1/documents", json=doc, headers={"X-Dev-User": "admin"})
    assert created.status_code == 201 and created.json()["chunks"] == 1
    assert "8 hours" in ask(client, "How soon are arterial roads cleared after snowfall?").json()["answer"]


def test_an_unknown_classification_is_rejected(client):
    doc = {"title": "X", "text": "y", "classification": "cosmic"}
    assert client.post("/v1/documents", json=doc, headers={"X-Dev-User": "admin"}).status_code == 422


def test_only_auditors_read_the_audit_log_and_it_verifies(client):
    ask(client, "What is the daily meal allowance?")
    assert client.get("/v1/audit", headers={"X-Dev-User": "ana"}).status_code == 403
    entries = client.get("/v1/audit", headers={"X-Dev-User": "chloe"}).json()
    assert entries[0]["action"] == "ask" and entries[0]["subject"] == "ana"
    assert client.get("/v1/audit/verify", headers={"X-Dev-User": "chloe"}).json() == {"intact": True, "first_broken_entry": None}


def test_unknown_dev_user_is_unauthorised(client):
    assert ask(client, "hi", user="mallory").status_code == 401


def test_the_web_page_and_its_sign_in_config_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "<title>Clearline</title>" in page.text
    config = client.get("/ui/config").json()
    assert config["auth_mode"] == "dev" and config["dev_users"] == ["admin", "ana", "ben", "chloe"]
    assert config["models"] == {"provider": "fake", "chat": "offline stand-in", "embed": "hashed bag of words",
                                "rerank": "query-term coverage", "retrieval": "hybrid", "citations": "markers"}


def test_me_describes_the_signed_in_user(client):
    assert client.get("/v1/me", headers={"X-Dev-User": "chloe"}).json() == {
        "subject": "chloe", "clearance": "Protected B", "groups": ["it", "security", "staff"], "admin": False, "auditor": True}


def test_the_page_signs_in_through_the_identity_provider_in_oidc_mode():
    settings = Settings(env="test", provider="fake", store="memory", auth_mode="oidc", embedding_dim=512,
                        oidc_issuer="https://sso.example.gc.ca/realms/lakeview", oidc_client_id="clearline-web")
    client = TestClient(create_app(settings, Clearline(settings), Authenticator(settings, key_for_token=lambda token: "")))
    config = client.get("/ui/config").json()
    assert (config["auth_mode"], config["issuer"], config["client_id"]) == (
        "oidc", "https://sso.example.gc.ca/realms/lakeview", "clearline-web")
    assert client.get("/v1/me").status_code == 401


def test_search_steps_report_each_retrieval_stage(client):
    body = ask(client, "How many days a week can I telework?").json()
    [step] = [s for s in body["steps"] if s["tool"] == "search_documents"]
    stages = step["retrieval"]
    assert stages["dense"] > 0 and stages["keyword"] > 0 and stages["reranked"] is True
    assert stages["kept"] == step["results"] <= stages["fused"]
    assert body["citation_mode"] == "markers" and 0 < body["coverage"] <= 1
