"""Turning text into vectors.

Embeddings have their own interface, separate from ``LLMProvider``: chat and embeddings are
different capabilities, and Anthropic doesn't offer the second one. So you can answer with Claude
and still embed with a local Ollama model.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

import httpx

from docaudit.llm.types import LLMError


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""
        ...


class HashingEmbedder:
    """A deterministic, offline embedder: texts sharing words get similar vectors.

    It has no idea what words mean, so it is useless for real retrieval. It exists so tests and
    experiments can run without a model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            # Not the builtin hash(): it changes on every run.
            digest = hashlib.md5(word.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class OllamaEmbedder:
    """Calls ``POST /api/embed`` on a local Ollama server, in batches."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        batch_size: int = 32,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout_s)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = await self.client.post(
                    "/api/embed", json={"model": self.model, "input": batch}
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMError(f"Ollama embedding request failed: {exc}") from exc
            embeddings: list[list[float]] = response.json()["embeddings"]
            if len(embeddings) != len(batch):
                raise LLMError(f"Expected {len(batch)} embeddings, got {len(embeddings)}")
            vectors.extend(embeddings)
        return vectors

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
