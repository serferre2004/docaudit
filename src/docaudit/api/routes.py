"""The document endpoints.   >>> YOUR TURN: implement the four handlers. <<<

Tests that define "done": tests/api/test_routes.py and tests/api/test_streaming.py

    POST /documents                  run a document until it is done or waits for a human
    POST /documents/stream           the same, sending each step as it finishes (SSE)
    GET  /documents/{id}             where it stands, and what it cost
    POST /documents/{id}/review      a human's decision for a document that is waiting

Every handler takes ``services: ServicesDep``. The router already requires
the API key, so the handlers never check it.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from fastapi.responses import StreamingResponse

from docaudit.agent import HumanDecision, resume_document, run_document
from docaudit.api.auth import require_api_key
from docaudit.api.deps import ServicesDep
from docaudit.api.schemas import DOCUMENT_ID_PATTERN, DocumentView, SubmitDocument
from docaudit.api.streaming import document_events
from docaudit.api.views import load_view
from docaudit.observability import track_document

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])

DocumentIdPath = Annotated[str, Path(pattern=DOCUMENT_ID_PATTERN)]


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def _not_found(document_id: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Unknown document {document_id!r}")


@router.post("", status_code=status.HTTP_201_CREATED)
async def submit_document(body: SubmitDocument, services: ServicesDep) -> DocumentView:
    """Run a new document and return its view.

    Requirements:
      * ``409`` if the document id was already submitted (``load_view`` isn't ``None``), or if
        its lock is held right now (``lock.locked()``): answer at once, don't queue behind it.
      * Otherwise hold the lock, run it with ``run_document`` inside ``track_document`` so its
        model calls are attributed to it, and return ``201`` with the view.
    """
    lock = services.locks.lock_for(body.document_id)
    if lock.locked():
        raise _conflict(f"Document {body.document_id} is locked")
    async with lock:
        if await load_view(services, body.document_id) is not None:
            raise _conflict(f"Document {body.document_id} already submitted")
        with track_document(body.document_id):
            await run_document(app=services.graph, document_id=body.document_id, document=body.text)
        view = await load_view(services, body.document_id)
    assert view is not None
    return view


@router.post("/stream")
async def stream_document(body: SubmitDocument, services: ServicesDep) -> StreamingResponse:
    """Like ``submit_document``, but answer at once and send the steps as Server-Sent Events.

    Requirements:
      * The same two ``409`` checks, BEFORE streaming starts: once the first byte is sent the
        status code is ``200`` and can't change.
      * Return ``StreamingResponse(document_events(...), media_type="text/event-stream")``.
    """
    lock = services.locks.lock_for(body.document_id)
    if lock.locked():
        raise _conflict(f"Document {body.document_id} is locked")
    if await load_view(services, body.document_id) is not None:
        raise _conflict(f"Document {body.document_id} already submitted")
    response = StreamingResponse(
        content=document_events(services, body.document_id, body.text),
        media_type="text/event-stream",
    )
    return response


@router.get("/{document_id}")
async def get_document(document_id: DocumentIdPath, services: ServicesDep) -> DocumentView:
    """The document's view, or ``404``."""
    view = await load_view(services, document_id)
    if view is None:
        raise _not_found(document_id)
    return view


@router.post("/{document_id}/review")
async def review_document(
    document_id: DocumentIdPath,
    human: HumanDecision,
    services: ServicesDep,
) -> DocumentView:
    """Resume a waiting document with a human's decision.

    Requirements:
      * ``404`` if the document is unknown; ``409`` if it isn't waiting for a human.
      * Two reviewers at once must not both win (Lesson 5, exercise E): ``409`` if the lock is
        held, and check the status INSIDE the lock, so the second one sees ``done``.
      * ``resume_document`` and return the view.
    """
    lock = services.locks.lock_for(document_id)
    if lock.locked():
        raise _conflict(f"Document {document_id} is locked")
    async with lock:
        view = await load_view(services, document_id)
        if view is None:
            raise _not_found(document_id)
        if view.status != "waiting_for_human":
            raise _conflict(f"Document {document_id} isn't waiting for human")
        await resume_document(services.graph, document_id, human)
        # Read it back while still holding the lock, so the answer is this review's result.
        view = await load_view(services, document_id)
    assert view is not None
    return view
