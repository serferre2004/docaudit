"""Server-Sent Events.   >>> YOUR TURN: implement ``document_events``. <<<

Tests that define "done": tests/api/test_streaming.py

A document takes about a minute on a CPU. Without streaming the client stares at a spinner and
can't tell "working" from "stuck". With SSE the response stays open and the server writes one
event per step as it happens. The format is plain text, one event per block:

    event: step
    data: {"node": "extract", "detail": "Extracted after 1 attempts"}

    event: done
    data: {"document_id": "...", "status": "done", ...}

SSE is one-way (server to client) over plain HTTP, which is all this needs. WebSockets would be
the choice if the client also had to talk back mid-stream.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel

from docaudit.agent import thread_config
from docaudit.api.deps import Services
from docaudit.api.schemas import StepEvent, StreamError
from docaudit.api.views import load_view
from docaudit.observability import track_document


def sse(event: str, data: BaseModel) -> str:
    """One SSE event (provided). ``model_dump_json`` never has newlines, which SSE forbids."""
    return f"event: {event}\ndata: {data.model_dump_json()}\n\n"


async def document_events(services: Services, document_id: str, text: str) -> AsyncIterator[str]:
    """Run a document and yield its events as they happen.

    Requirements:
      * Hold the document's lock for the whole run, and attribute the calls with
        ``track_document``, exactly as ``submit_document`` does.
      * ``services.graph.astream(<same input as run_document>, thread_config(document_id),
        stream_mode="updates")`` yields one dict per finished node: ``{node_name: update}``. For
        every ``TrailEvent`` in ``update["trail"]``, yield a ``step`` event (``StepEvent``).
        Skip the ``"__interrupt__"`` key: that is the pause, not a node.
      * When the stream ends, yield one ``done`` event with the ``load_view`` of the document.
    """
    lock = services.locks.lock_for(document_id)
    async with lock:
        # The route checked before streaming, but without the lock: a second request for the same
        # id can pass that check before this generator starts. Check again now that we hold it.
        if await load_view(services, document_id) is not None:
            yield sse("error", StreamError(detail=f"Document {document_id} already submitted"))
            return
        with track_document(document_id):
            async for chunk in services.graph.astream(
                {"document_id": document_id, "document": text, "trail": []},
                thread_config(document_id),
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if node == "__interrupt__" or not update:
                        continue
                    for event in update.get("trail", []):
                        yield sse("step", StepEvent(node=node, detail=event.detail))
        view = await load_view(services, document_id)
    assert view is not None
    yield sse("done", view)
