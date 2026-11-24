import asyncio

import pytest

from boardforge.semantic import (
    RETRIEVAL_DOCUMENT,
    RETRIEVAL_QUERY,
    VertexGeminiEmbeddingProvider,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeAsyncClient:
    def __init__(self, capture, response, **kwargs):
        self.capture = capture
        self.response = response
        self.capture["client_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.capture.setdefault("requests", []).append((url, kwargs))
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


def _provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "boardforge-ai-2026")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("BOARDFORGE_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("BOARDFORGE_EMBEDDING_DIMENSIONS", "768")
    return VertexGeminiEmbeddingProvider()


@pytest.mark.parametrize("task_type", [RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT])
def test_vertex_embedding_http_contract(monkeypatch, task_type):
    provider = _provider(monkeypatch)
    capture = {}
    response = FakeResponse(
        payload={
            "predictions": [
                {"embeddings": {"values": [0.01] * 768}}
            ]
        }
    )

    async def fake_token():
        return "token"

    monkeypatch.setattr(provider, "_access_token", fake_token)
    monkeypatch.setattr(
        "boardforge.semantic.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(capture, response, **kwargs),
    )

    vectors = asyncio.run(provider.embed(["backend engineer India"], task_type=task_type))

    assert len(vectors) == 1
    assert len(vectors[0]) == 768
    url, request = capture["requests"][0]
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "boardforge-ai-2026/locations/us-central1/publishers/google/models/"
        "gemini-embedding-001:predict"
    )
    assert request["headers"]["Authorization"] == "Bearer token"
    body = request["json"]
    assert body["instances"] == [
        {"content": "backend engineer India", "task_type": task_type}
    ]
    assert body["parameters"]["outputDimensionality"] == 768
    assert body["parameters"]["autoTruncate"] is True


def test_vertex_embedding_dimension_guard(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "boardforge-ai-2026")
    monkeypatch.setenv("BOARDFORGE_EMBEDDING_DIMENSIONS", "1536")
    with pytest.raises(ValueError, match="fixed at 768 dimensions"):
        VertexGeminiEmbeddingProvider()


def test_vertex_embedding_response_dimension_is_validated(monkeypatch):
    provider = _provider(monkeypatch)
    capture = {}
    response = FakeResponse(
        payload={
            "predictions": [
                {"embeddings": {"values": [0.01] * 10}}
            ]
        }
    )

    async def fake_token():
        return "token"

    monkeypatch.setattr(provider, "_access_token", fake_token)
    monkeypatch.setattr(
        "boardforge.semantic.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(capture, response, **kwargs),
    )

    with pytest.raises(RuntimeError, match="expected 768, got 10"):
        asyncio.run(
            provider.embed(["backend engineer India"], task_type=RETRIEVAL_QUERY)
        )


def test_vertex_embedding_retries_rate_limit(monkeypatch):
    provider = _provider(monkeypatch)
    provider._max_retries = 1
    capture = {}
    responses = [
        FakeResponse(status_code=429),
        FakeResponse(payload={
            "predictions": [{"embeddings": {"values": [0.01] * 768}}]
        }),
    ]

    async def fake_token():
        return "token"

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_access_token", fake_token)
    monkeypatch.setattr("boardforge.semantic.asyncio.sleep", no_sleep)
    monkeypatch.setattr(
        "boardforge.semantic.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(capture, responses, **kwargs),
    )

    vectors = asyncio.run(
        provider.embed(["backend engineer India"], task_type=RETRIEVAL_QUERY)
    )

    assert len(capture["requests"]) == 2
    assert len(vectors[0]) == 768
