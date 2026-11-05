from collections.abc import Sequence

from docaudit.rag import Chunk, SearchHit


def make_chunk(
    chunk_id: str = "DOC#1.1",
    text: str = "Some clause text.",
    *,
    doc_id: str = "DOC",
    section: str = "1.1",
    doc_title: str = "A Title",
    heading: str = "1. A Heading",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        section=section,
        heading=heading,
        text=text,
    )


def hits_for(*chunk_ids: str) -> list[SearchHit]:
    """A ranking with the given chunks, best first."""
    return [
        SearchHit(chunk=make_chunk(chunk_id), score=float(len(chunk_ids) - position))
        for position, chunk_id in enumerate(chunk_ids)
    ]


class ScriptedEmbedder:
    """Returns the vector assigned to each text and records every call."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


class ScriptedRetriever:
    """Returns a fixed ranking, cut at ``k``, and records every call."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((query, k))
        return self.hits[:k]
