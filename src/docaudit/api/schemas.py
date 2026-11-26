"""What goes over the wire (provided).

The request models are the API's first line of defense: FastAPI validates every request against
them and answers ``422`` before your code runs. Limits belong here, not in the handler: a
document id that becomes a checkpoint key, and a text that becomes model input (and money).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from docaudit.agent import Action, AuditRecord, Decision
from docaudit.observability import UsageSummary

DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
DocumentId = Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]

DocumentStatus = Literal["running", "waiting_for_human", "done"]


class SubmitDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: DocumentId
    text: str = Field(min_length=1, max_length=50_000)


class DocumentView(BaseModel):
    """Where a document stands. ``running`` also covers a run that died half way."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    status: DocumentStatus
    decision: Decision | None = None
    record: AuditRecord | None = None
    usage: UsageSummary


class StepEvent(BaseModel):
    """One node finished: what the stream sends while a document runs."""

    model_config = ConfigDict(frozen=True)

    node: str
    detail: str


class StreamError(BaseModel):
    """The stream could not run the document. The status is already 200, so it says so here."""

    model_config = ConfigDict(frozen=True)

    detail: str


class DemoCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    expected: Action
    text: str
