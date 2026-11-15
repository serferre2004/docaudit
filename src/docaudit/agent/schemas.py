"""Types for the approval workflow (provided).

The workflow ends every document in one of two final actions, ``approve`` or ``flag``. The third
action, ``escalate``, is never final: it means "a human has to choose", and the graph waits until
one does.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from docaudit.tools.schemas import Finding, ToolExecution

Action = Literal["approve", "flag", "escalate"]
FinalAction = Literal["approve", "flag"]


class Policy(BaseModel):
    """The company's rules for what may be approved without a person.

    ``auto_approve_limit_usd``: a clean invoice above this amount still needs a human signature.
    """

    model_config = ConfigDict(frozen=True)

    auto_approve_limit_usd: Decimal = Decimal("3000")


class Decision(BaseModel):
    """What the policy decided, and why. ``reasons`` is what a reviewer reads first."""

    model_config = ConfigDict(frozen=True)

    action: Action
    reasons: tuple[str, ...] = Field(min_length=1)


class HumanDecision(BaseModel):
    """A reviewer's answer to an escalation. A human can only approve or flag."""

    model_config = ConfigDict(frozen=True)

    action: FinalAction
    reviewer: str = Field(min_length=1)
    note: str = ""


class TrailEvent(BaseModel):
    """One step of the audit trail: which node ran and what it concluded."""

    model_config = ConfigDict(frozen=True)

    step: str
    detail: str


class AuditRecord(BaseModel):
    """The final, explainable record of one document. This is what an auditor reads later."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    invoice_number: str | None
    action: FinalAction
    decided_by: str = Field(description="'policy', or 'human:<reviewer>'")
    reasons: tuple[str, ...]
    findings: tuple[Finding, ...] = ()
    tool_calls: tuple[ToolExecution, ...] = ()
    trail: tuple[TrailEvent, ...] = ()


class AgentOutcome(BaseModel):
    """Where a document stands after ``run_document`` or ``resume_document``."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    status: Literal["done", "waiting_for_human"]
    decision: Decision | None = None
    record: AuditRecord | None = None
