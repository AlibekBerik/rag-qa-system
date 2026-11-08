"""API tests with stand-ins for the embedding model, ChromaDB and the LLM.

They check the request flow and the grounding prompt without downloading
models or calling Groq.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeEmbedModel:
    def encode(self, texts):
        return FakeVector([[0.1, 0.2, 0.3] for _ in texts])


class FakeCollection:
    def __init__(self):
        self.last_query = None

    def query(self, query_embeddings, n_results):
        self.last_query = {"embeddings": query_embeddings, "n_results": n_results}
        return {
            "documents": [["Yurts are made of felt.", "The dombra has two strings.", "Felt is pressed wool."]],
            "metadatas": [[{"topic": "Yurt"}, {"topic": "Dombra"}, {"topic": "Yurt"}]],
        }


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


@pytest.fixture
def fakes(monkeypatch):
    collection = FakeCollection()
    llm = FakeLLM("Yurts are made of felt and wood.")
    monkeypatch.setattr(api, "get_embed_model", lambda: FakeEmbedModel())
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    monkeypatch.setattr(api, "get_llm_client", lambda: llm)
    return SimpleNamespace(collection=collection, llm=llm)


def test_root():
    assert TestClient(api.app).get("/").json()["message"].endswith("is running")


def test_ask_returns_answer_and_sources(fakes):
    response = TestClient(api.app).post("/ask", params={"question": "What is a yurt made of?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Yurts are made of felt and wood."
    # Duplicates removed, most relevant source first
    assert body["sources"] == ["Yurt", "Dombra"]
    assert fakes.collection.last_query["n_results"] == api.TOP_K


def test_prompt_contains_retrieved_context_and_refusal_rule(fakes):
    TestClient(api.app).post("/ask", params={"question": "What is a yurt made of?"})

    prompt = fakes.llm.prompts[0]
    assert "Yurts are made of felt." in prompt
    assert "ONLY the context" in prompt
    assert "don't have enough information" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_build_prompt_keeps_chunk_order():
    prompt = api.build_prompt("q", ["first", "second"])
    assert prompt.index("first") < prompt.index("second")
