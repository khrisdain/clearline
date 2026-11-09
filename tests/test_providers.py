"""The Cohere and Ollama adapters, against stand-in clients (no network)."""

import json
from types import SimpleNamespace as NS

import httpx

from clearline.config import Settings
from clearline.models import Message, ToolCall
from clearline.providers.cohere import CohereProvider
from clearline.providers.ollama import OllamaProvider


class FakeCohere:
    def __init__(self):
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        return NS(embeddings=NS(float_=[[0.1, 0.2]] * len(kwargs["texts"])))

    def rerank(self, **kwargs):
        self.calls.append(("rerank", kwargs))
        return NS(results=[NS(index=1, relevance_score=0.9), NS(index=0, relevance_score=0.2)])

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        call = NS(id="tc_1", function=NS(name="search_documents", arguments='{"query": "meals"}'))
        return NS(message=NS(tool_calls=[call], content=None, tool_plan="I will search."))


def cohere_settings():
    return Settings(env="test", provider="cohere", COHERE_API_KEY="test-key", embedding_dim=256)


def test_cohere_embeds_in_batches_with_the_right_input_type():
    client = FakeCohere()
    vectors = CohereProvider(cohere_settings(), client).embed([f"t{i}" for i in range(100)], "document")
    assert len(vectors) == 100
    embeds = [kw for name, kw in client.calls if name == "embed"]
    assert [len(kw["texts"]) for kw in embeds] == [96, 4]
    assert embeds[0]["input_type"] == "search_document" and embeds[0]["output_dimension"] == 256


def test_cohere_rerank_and_tool_calls_round_trip():
    client = FakeCohere()
    provider = CohereProvider(cohere_settings(), client)
    assert provider.rerank("q", ["a", "b"], 2) == [(1, 0.9), (0, 0.2)]
    reply = provider.chat([Message("system", "s"), Message("user", "meals?"),
                           Message("assistant", tool_calls=[ToolCall("tc_0", "search_documents", {"query": "x"})]),
                           Message("tool", '{"results": []}', tool_call_id="tc_0")], [])
    assert reply.tool_calls[0].arguments == {"query": "meals"}
    sent = client.calls[-1][1]["messages"]
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"query": "x"})
    assert sent[3] == {"role": "tool", "tool_call_id": "tc_0", "content": '{"results": []}'}


def test_ollama_chat_and_cosine_rerank():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0] if "cats" in t else [0.0, 1.0] for t in body["input"]]})
        return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
            {"function": {"name": "search_documents", "arguments": {"query": "cats"}}}]}})

    provider = OllamaProvider(Settings(env="test", provider="ollama"),
                              httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler)))
    assert provider.rerank("cats", ["dogs", "cats"], 2)[0][0] == 1
    reply = provider.chat([Message("user", "cats?")], [])
    assert reply.tool_calls[0].name == "search_documents" and reply.tool_calls[0].id == "call_0"


def test_cohere_reply_with_native_citations_becomes_spans():
    class Citing(FakeCohere):
        def chat(self, **kwargs):
            self.calls.append(("chat", kwargs))
            citation = NS(start=10, end=19, text="$98 a day", sources=[NS(type="tool", id="passage-1", tool_output={})])
            return NS(message=NS(tool_calls=None, tool_plan=None, content=[NS(type="text", text="Meals are $98 a day.")],
                                 citations=[citation]))

    reply = CohereProvider(cohere_settings(), Citing()).chat([Message("user", "meals?")], [])
    assert reply.content == "Meals are $98 a day."
    assert [(s.start, s.end, s.text, s.markers) for s in reply.spans] == [(10, 19, "$98 a day", [1])]
