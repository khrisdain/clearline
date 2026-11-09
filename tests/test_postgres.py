"""The same access guarantees against real Postgres + pgvector. Runs when CLEARLINE_TEST_DATABASE_URL is set
(CI starts a pgvector service; locally: docker compose up db)."""

import os
from pathlib import Path

import pytest

from clearline.audit import PostgresAuditLog
from clearline.auth import DEV_USERS
from clearline.config import Settings
from clearline.ingest import load_directory
from clearline.service import Clearline

URL = os.environ.get("CLEARLINE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="CLEARLINE_TEST_DATABASE_URL not set")
SAMPLE = Path(__file__).parent.parent / "sample_data"


@pytest.fixture(scope="module")
def pg() -> Clearline:
    from clearline.store.postgres import PostgresStore

    store = PostgresStore(URL, dim=512)
    with store.pool.connection() as conn:
        conn.execute("TRUNCATE chunks, documents")
        conn.execute("DROP TABLE IF EXISTS audit_log")
    clearline = Clearline(Settings(env="test", provider="fake", store="postgres", database_url=URL, embedding_dim=512),
                          store=store, audit=PostgresAuditLog(store.pool))
    for document in load_directory(SAMPLE):
        clearline.ingest(document)
    return clearline


@pytest.mark.parametrize("user", sorted(DEV_USERS))
def test_vector_search_filters_inside_the_query(pg: Clearline, user: str):
    principal = DEV_USERS[user]
    [vector] = pg.provider.embed(["budget property tax incident escalation passwords"], "query")
    hits = pg.store.search(vector, principal, 100)
    assert hits, user
    assert all(principal.can_read(h.chunk.classification, h.chunk.groups) for h in hits)


def test_document_lists_and_reads_respect_access(pg: Clearline):
    ana = {d.title for d, _ in pg.store.documents(DEV_USERS["ana"])}
    ben = {d.title: d.id for d, _ in pg.store.documents(DEV_USERS["ben"])}
    assert "2027 Operating Budget Forecast" in ben and "2027 Operating Budget Forecast" not in ana
    assert pg.store.chunks_of(ben["2027 Operating Budget Forecast"], DEV_USERS["ana"]) == []
    assert pg.store.chunks_of(ben["2027 Operating Budget Forecast"], DEV_USERS["ben"])


def test_answers_and_the_audit_chain(pg: Clearline):
    assert "2.9 percent" in pg.ask("What property tax increase is forecast for 2027?", DEV_USERS["ben"]).answer
    assert not pg.ask("What property tax increase is forecast for 2027?", DEV_USERS["ana"]).grounded
    assert pg.audit.verify() is None
    with pg.store.pool.connection() as conn:
        conn.execute("UPDATE audit_log SET subject = 'someone-else' WHERE seq = 2")
    assert pg.audit.verify() == 2


@pytest.mark.parametrize("user", sorted(DEV_USERS))
def test_keyword_search_filters_inside_the_query(pg: Clearline, user: str):
    principal = DEV_USERS[user]
    hits = pg.store.keyword_search("budget tax incident passwords phishing telework travel", principal, 100)
    assert hits, user
    for hit in hits:
        assert principal.can_read(hit.chunk.classification, hit.chunk.groups), (user, hit.chunk.title)


def test_postgres_hybrid_retrieval_matches_the_memory_store(pg: Clearline, service: Clearline):
    for question in ("How do I report a phishing email?", "How much is the daily meal allowance?"):
        expected = service.retriever.search(question, DEV_USERS["chloe"])[0].chunk.title
        assert pg.retriever.search(question, DEV_USERS["chloe"])[0].chunk.title == expected
