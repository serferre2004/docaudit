"""Checkpointers for the graph (provided).

A checkpointer saves the state after every node, keyed by a *thread id*. Two things depend on it:

* ``interrupt()`` (human-in-the-loop): the graph stops, and the saved state is what it resumes
  from, possibly in another process, hours later.
* Recovery: if a node fails (Ollama is down), running the same thread again continues from the
  last saved step instead of redoing the expensive ones.

Saved state is serialized. LangGraph refuses to rebuild arbitrary classes from a checkpoint
unless they are listed, so a tampered checkpoint file can't make it instantiate whatever it wants.
``ALLOWED_TYPES`` lists every model this graph keeps in its state.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ALLOWED_TYPES = [
    ("docaudit.extraction.schemas", "Invoice"),
    ("docaudit.extraction.schemas", "LineItem"),
    ("docaudit.llm.types", "ToolCall"),
    ("docaudit.llm.types", "ToolResult"),
    ("docaudit.tools.schemas", "Finding"),
    ("docaudit.tools.schemas", "ToolExecution"),
    ("docaudit.agent.schemas", "Decision"),
    ("docaudit.agent.schemas", "HumanDecision"),
    ("docaudit.agent.schemas", "TrailEvent"),
    ("docaudit.agent.schemas", "AuditRecord"),
]

SERDE = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """Lives as long as the process. Right for tests and for one-shot evals."""
    return InMemorySaver(serde=SERDE)


@asynccontextmanager
async def sqlite_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Persists to a file, so a run can stop in one command and resume in another."""
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as conn:
        yield AsyncSqliteSaver(conn, serde=SERDE)
