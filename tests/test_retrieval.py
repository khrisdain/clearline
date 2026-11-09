"""Hybrid retrieval: BM25 keyword search, reciprocal rank fusion, and the access filter on both."""

from clearline.agent import Retriever, reciprocal_rank_fusion
from clearline.auth import DEV_USERS
from clearline.config import Settings
from clearline.models import Chunk, Classification, Hit


def chunk(i: str) -> Chunk:
    return Chunk(id=i, document_id="d", title="T", section="S", text=i, classification=Classification.UNCLASSIFIED,
                 groups=frozenset(), position=0)


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    fused = reciprocal_rank_fusion([[Hit(a, 0.9), Hit(b, 0.8)], [Hit(b, 12.0), Hit(c, 3.0)]], k=60)
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]  # b is in both lists
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12
    assert reciprocal_rank_fusion([]) == []


def test_bm25_ranks_the_passage_with_the_rare_exact_term_first(service):
    hits = service.store.keyword_search("phishing", DEV_USERS["ben"], 5)
    assert hits and hits[0].chunk.title == "IT Security Standard"
    assert all(h.score > 0 for h in hits)


def test_keyword_search_respects_access_like_vector_search(service):
    query = "property tax increase 2027 budget forecast"
    assert any(h.chunk.title.startswith("2027") for h in service.store.keyword_search(query, DEV_USERS["ben"], 20))
    assert not any(h.chunk.title.startswith("2027") for h in service.store.keyword_search(query, DEV_USERS["ana"], 20))
    assert service.store.keyword_search("the and of", DEV_USERS["ben"], 5) == []  # only stop words


def test_hybrid_retrieval_reports_each_stage(service, settings):
    retrieval = Retriever(service.provider, service.store, settings).retrieve("daily meal allowance", DEV_USERS["ana"])
    assert retrieval.dense and retrieval.keyword and retrieval.reranked
    assert retrieval.fused <= settings.candidates
    assert retrieval.hits[0].chunk.title == "Travel and Expense Policy"
    assert retrieval.describe()["kept"] == len(retrieval.hits)


def test_every_retrieval_mode_finds_the_answer(service):
    for mode in ("dense", "keyword", "hybrid"):
        for rerank in (True, False):
            settings = Settings(env="test", provider="fake", store="memory", embedding_dim=512, retrieval=mode, rerank=rerank)
            hits = Retriever(service.provider, service.store, settings).search("How many days a week can I telework?",
                                                                               DEV_USERS["ana"])
            assert hits[0].chunk.title == "Telework Policy", (mode, rerank)
