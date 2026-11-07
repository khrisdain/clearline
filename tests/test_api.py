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
