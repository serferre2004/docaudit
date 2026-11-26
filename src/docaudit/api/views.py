"""Reading a document's state from its checkpoints.   >>> YOUR TURN: implement ``load_view``. <<<

Tests that define "done": tests/api/test_routes.py (every endpoint returns a view)
"""

from docaudit.agent import thread_config
from docaudit.api.deps import Services
from docaudit.api.schemas import DocumentView


async def load_view(services: Services, document_id: str) -> DocumentView | None:
    """What the API knows about ``document_id``, or ``None`` if it was never submitted.

    The checkpointer is the source of truth: no separate "documents" table that could disagree.

    Requirements:
      * ``await services.graph.aget_state(thread_config(document_id))``. An unknown thread comes
        back with empty ``values``: return ``None``.
      * ``status``: ``waiting_for_human`` if the snapshot has ``interrupts``; ``done`` if the
        values hold a ``record``; otherwise ``running`` (still going, or died half way).
      * ``decision`` and ``record`` from the values when present, and ``usage`` from
        ``services.tracker.summary(document_id)``.
    """
    state = await services.graph.aget_state(thread_config(document_id))
    if not state.values:
        return None

    status = "running"
    if state.interrupts:
        status = "waiting_for_human"
    elif state.values.get("record"):
        status = "done"
    return DocumentView(
        document_id=document_id,
        status=status,
        decision=state.values.get("decision"),
        record=state.values.get("record"),
        usage=services.tracker.summary(document_id),
    )
