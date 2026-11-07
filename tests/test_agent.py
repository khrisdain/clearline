import json
from collections.abc import Callable

from clearline.agent import Agent, Retriever
from clearline.auth import DEV_USERS
from clearline.config import Settings
from clearline.models import Message, ToolCall, ToolSpec
from clearline.providers.fake import FakeProvider
from clearline.service import Clearline


class Scripted(FakeProvider):
    """The fake's embeddings and reranking, with chat replies scripted per test."""

    def __init__(self, replies: list[Message | Callable[[list[Message]], Message]]) -> None:
        super().__init__(dim=512)
        self.replies, self.seen = list(replies), []

    def chat(self, messages: list[Message], tools: list[ToolSpec]) -> Message:
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def search(query: str, call_id: str = "c1") -> Message:
    return Message("assistant", tool_calls=[ToolCall(call_id, "search_documents", {"query": query})])


def agent_with(service: Clearline, settings: Settings, provider: Scripted) -> Agent:
    return Agent(provider, Retriever(provider, service.store, settings), service.store, settings)


def test_citations_to_passages_never_shown_are_dropped(service, settings):
    provider = Scripted([search("meal allowance"), Message("assistant", "It is $98 a day [1], or maybe $500 [7].")])
    answer = agent_with(service, settings, provider).run("meal allowance?", DEV_USERS["ana"])
    assert answer.answer == "It is $98 a day [1], or maybe $500."
    assert [c.marker for c in answer.citations] == [1]
    assert answer.grounded


def test_an_answer_without_valid_citations_is_ungrounded(service, settings):
    provider = Scripted([search("meal allowance"), Message("assistant", "Probably around $90.")])
    answer = agent_with(service, settings, provider).run("meal allowance?", DEV_USERS["ana"])
    assert not answer.grounded and answer.citations == []


def test_the_agent_can_search_twice_and_numbering_continues(service, settings):
    def answer_from(messages: list[Message]) -> Message:
        first, second = (json.loads(m.content)["results"] for m in messages if m.role == "tool")
        meals = next(r["marker"] for r in first if r["section"] == "Meals and incidentals")
        cars = next(r["marker"] for r in second if r["section"] == "Personal vehicles")
        assert cars > max(r["marker"] for r in first)  # new passages get new numbers
        return Message("assistant", f"Meals are $98 [{meals}]; driving pays 68 cents a km [{cars}].")

    provider = Scripted([search("meal allowance"), search("kilometre vehicle rate", "c2"), answer_from])
    answer = agent_with(service, settings, provider).run("travel costs?", DEV_USERS["ana"])
    assert [s.tool for s in answer.steps] == ["search_documents", "search_documents"]
    assert {c.section for c in answer.citations} == {"Meals and incidentals", "Personal vehicles"}


def test_out_of_steps_forces_an_answer_without_tools(service, settings):
    settings.agent_max_steps = 2
    provider = Scripted([search("a"), search("b", "c2"), Message("assistant", "Done [1].")])
    answer = agent_with(service, settings, provider).run("anything", DEV_USERS["ana"])
    assert answer.answer.startswith("Done")
    assert provider.seen[-1][-1].content.startswith("Answer now")


def test_personal_information_is_masked_before_the_model_sees_it(service, settings):
    provider = Scripted([Message("assistant", "Noted.")])
    agent_with(service, settings, provider).run("My SIN is 046 454 286, email jo@lakeview.ca", DEV_USERS["ana"])
    prompt = provider.seen[0][1].content
    assert "046 454 286" not in prompt and "jo@lakeview.ca" not in prompt
    assert "[SIN]" in prompt and "[EMAIL]" in prompt


def test_reading_a_forbidden_document_returns_nothing(service, settings):
    budget = next(d for d, _ in service.store.documents(DEV_USERS["ben"]) if d.title.startswith("2027"))
    provider = Scripted([Message("assistant", tool_calls=[ToolCall("c1", "read_document", {"document_id": budget.id})]),
                         Message("assistant", "Nothing found.")])
    answer = agent_with(service, settings, provider).run("open the budget", DEV_USERS["ana"])
    assert answer.steps[0].results == 0
    assert '"results": []' in provider.seen[-1][-1].content
