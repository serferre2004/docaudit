"""What the endpoints share, and how they get it (provided).

The graph, the tracker and the settings are built ONCE, when the app starts (see ``app.py``), and
reach each request through ``Depends(get_services)``. Tests build their own ``Services`` with a
``FakeProvider`` and an in-memory checkpointer, so they never touch Ollama or the disk.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request

from docaudit.agent import AgentGraph
from docaudit.config import Settings
from docaudit.observability import UsageTracker


@dataclass
class DocumentLocks:
    """One ``asyncio.Lock`` per document id.

    Enough for ONE process. With several workers (``uvicorn --workers 4``) or several containers
    each has its own locks, and they protect nothing: you'd need a lock the processes share, such
    as a row in a database. The locks are never removed; a long-running server would evict them.
    """

    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def lock_for(self, document_id: str) -> asyncio.Lock:
        return self._locks.setdefault(document_id, asyncio.Lock())


@dataclass
class Services:
    graph: AgentGraph
    tracker: UsageTracker
    settings: Settings
    locks: DocumentLocks = field(default_factory=DocumentLocks)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


# Declare a handler parameter as `services: ServicesDep` and FastAPI fills it in.
ServicesDep = Annotated[Services, Depends(get_services)]
