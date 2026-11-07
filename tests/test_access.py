"""The guarantee that matters most: nobody gets an answer drawn from a document they may not read."""

import pytest

from clearline.auth import DEV_USERS
from clearline.models import Classification, Principal
from clearline.service import Clearline

QUESTIONS = [
    "What property tax increase is forecast for 2027?",
    "How quickly must a Severity 1 incident be escalated to the CISO?",
    "How long must network passwords be?",
    "What is the daily meal allowance?",
    "How many days a week can I telework?",
    "Summarise the reserve balance and transit cost pressures.",
]


def test_clearance_and_groups_both_gate_reading():
    reader = Principal("r", Classification.PROTECTED_A, frozenset({"it"}))
    assert reader.can_read(Classification.UNCLASSIFIED, frozenset())
    assert reader.can_read(Classification.PROTECTED_A, frozenset({"it", "security"}))
    assert not reader.can_read(Classification.PROTECTED_B, frozenset())  # level too high
    assert not reader.can_read(Classification.UNCLASSIFIED, frozenset({"finance"}))  # wrong group


@pytest.mark.parametrize("user", sorted(DEV_USERS))
def test_search_only_returns_readable_chunks(service: Clearline, user: str):
    principal = DEV_USERS[user]
    for question in QUESTIONS:
        [vector] = service.provider.embed([question], "query")
        for hit in service.store.search(vector, principal, 50):
            assert principal.can_read(hit.chunk.classification, hit.chunk.groups), (user, hit.chunk.title)


@pytest.mark.parametrize("user", sorted(DEV_USERS))
def test_answers_never_cite_unreadable_documents(service: Clearline, user: str):
    everyone = Principal("root", Classification.PROTECTED_B, frozenset({"staff", "finance", "security", "it"}))
    labels = {d.id: (d.classification, d.groups) for d, _ in service.store.documents(everyone)}
    principal = DEV_USERS[user]
    for question in QUESTIONS:
        for citation in service.ask(question, principal).citations:
            assert principal.can_read(*labels[citation.document_id]), (user, citation.title)


def test_the_same_question_is_answered_for_finance_and_refused_for_others(service: Clearline):
    question = "What property tax increase is forecast for 2027?"
    assert "2.9 percent" in service.ask(question, DEV_USERS["ben"]).answer
    refused = service.ask(question, DEV_USERS["ana"])
    assert not refused.grounded and "2.9" not in refused.answer


def test_a_forbidden_document_looks_like_one_that_does_not_exist(service: Clearline):
    budget = next(d for d, _ in service.store.documents(DEV_USERS["ben"]) if d.title.startswith("2027"))
    assert service.store.chunks_of(budget.id, DEV_USERS["ana"]) == []
    assert service.store.chunks_of("no-such-document", DEV_USERS["ana"]) == []
    assert budget.id not in {d.id for d, _ in service.store.documents(DEV_USERS["ana"])}
