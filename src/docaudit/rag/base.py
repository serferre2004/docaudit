from typing import Protocol

from docaudit.rag.schemas import SearchHit


class Retriever(Protocol):
    """Anything that can rank chunks for a query.

    Vector, keyword and hybrid search all share this shape, so the code that uses a retriever
    doesn't know which strategy it got, and comparing strategies is a one-line change.
    """

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return at most ``k`` hits, best first."""
        ...
