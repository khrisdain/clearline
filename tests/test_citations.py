"""Citations as spans: [n] markers from most models, native span citations from Cohere's Command."""

from types import SimpleNamespace as NS

from test_agent import Scripted, agent_with, search

from clearline.agent import Sources, cite, coverage
from clearline.auth import DEV_USERS
from clearline.models import Chunk, Classification, Message, Span
from clearline.providers.cohere import as_documents, to_span


def sources(n: int = 3) -> Sources:
    out = Sources()
    for i in range(n):
        out.add(Chunk(id=f"c{i}", document_id="d", title="T", section="S", text="x",
                      classification=Classification.UNCLASSIFIED, groups=frozenset(), position=i))
    return out


def test_markers_become_spans_over_their_sentence():
    text, spans = cite("Meals are $98 [1]; driving pays 68 cents a km [2][3]. Parking needs receipts.", sources())
    assert [(s.text, s.markers) for s in spans] == [("Meals are $98", [1]), ("driving pays 68 cents a km", [2, 3])]
    assert all(text[s.start:s.end] == s.text for s in spans)
    assert 0.5 < coverage(text, spans) < 1  # the last sentence is uncited


def test_unknown_markers_are_removed_and_cover_nothing():
    text, spans = cite("It is $98 a day [1], or maybe $500 [7].", sources())
    assert text == "It is $98 a day [1], or maybe $500."
    assert [s.markers for s in spans] == [[1]]
    assert coverage("No citations here.", []) == 0.0


def test_native_spans_are_checked_against_what_the_model_saw(service, settings):
    def answer(messages: list[Message]) -> Message:
        text = "Meals are $98 a day. Lunch is free on Fridays."
        return Message("assistant", text, spans=[Span(start=0, end=19, text=text[:19], markers=[1, 1]),
                                                 Span(start=21, end=46, text=text[21:46], markers=[99])])

    provider = Scripted([search("meal allowance"), answer])
    provider.native_citations = True
    result = agent_with(service, settings, provider).run("meals?", DEV_USERS["ana"])
    assert result.citation_mode == "native"
    assert [s.markers for s in result.spans] == [[1]]  # the span citing a passage never shown is dropped
    assert [c.marker for c in result.citations] == [1] and result.grounded
    assert "do not write [n] markers" in provider.seen[0][0].content


def test_cohere_citations_map_back_to_passage_numbers():
    text = "Meals are $98 a day."
    cited = NS(start=0, end=19, text=text[:19], sources=[NS(type="tool", id="passage-2", tool_output={}),
                                                          NS(type="tool", id="call_1:0", tool_output={"marker": 3})])
    span = to_span(cited, text)
    assert span is not None and span.markers == [2, 3] and span.text == "Meals are $98 a day"
    assert to_span(NS(start=0, end=99, sources=[NS(id="passage-1")]), text) is None  # out of range
    assert to_span(NS(start=0, end=5, sources=[NS(id="other", tool_output={})]), text) is None


def test_tool_results_go_to_cohere_as_citable_documents():
    content = ('{"results": [{"marker": 4, "document_id": "d1", "title": "Travel", "section": "Meals", '
               '"classification": "Unclassified", "text": "Meals are $98."}]}')
    [doc] = as_documents(content)
    assert doc["type"] == "document" and doc["document"]["id"] == "passage-4"
    assert doc["document"]["data"]["text"] == "Meals are $98."
    assert as_documents('{"results": []}') == '{"results": []}'
