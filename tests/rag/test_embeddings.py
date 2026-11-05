import json

import httpx
import pytest

from docaudit.llm import LLMError
from docaudit.rag import HashingEmbedder, OllamaEmbedder


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=32)
    first, second = await embedder.embed(["pay the invoice", "pay the invoice"])
    assert first == second
    assert len(first) == 32
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert (await embedder.embed([""]))[0] == [0.0] * 32


async def test_ollama_embedder_sends_batches_and_keeps_the_order() -> None:
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == "embed-test"
        batches.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[float(t)] for t in body["input"]]})

    transport = httpx.MockTransport(handler)
    async with OllamaEmbedder(
        base_url="http://test", model="embed-test", batch_size=2, transport=transport
    ) as embedder:
        vectors = await embedder.embed(["1", "2", "3"])

    assert batches == [["1", "2"], ["3"]]
    assert vectors == [[1.0], [2.0], [3.0]]


async def test_ollama_embedder_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="model not found"))
    async with OllamaEmbedder(base_url="http://test", model="m", transport=transport) as embedder:
        with pytest.raises(LLMError):
            await embedder.embed(["x"])
