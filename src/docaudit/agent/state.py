"""The graph's state (provided).

LangGraph passes this dict from node to node. A node never mutates it: it RETURNS a dict with only
the keys it changed, and LangGraph merges that into the state. Every merge is saved by the
checkpointer, which is what lets a run stop (for a human, or a crash) and continue later.

``trail`` is special: its ``Annotated[..., operator.add]`` is a *reducer*. When a node returns
``{"trail": [event]}``, LangGraph APPENDS the event instead of replacing the list. Every other key
is replaced.
"""

import operator
from typing import Annotated, TypedDict

from docaudit.agent.schemas import AuditRecord, Decision, HumanDecision, TrailEvent
from docaudit.extraction.schemas import Invoice
from docaudit.tools.schemas import Finding, ToolExecution


class AgentState(TypedDict, total=False):
    document_id: str
    """Identifies the document. It is also the checkpoint thread id."""
    document: str
    """The raw text of the document, as received."""
    invoice: Invoice | None
    """Set by ``extract``; ``None`` if extraction failed."""
    error: str | None
    """Why the pipeline could not finish (extraction or audit failed). Forces an escalation."""
    findings: list[Finding]
    executions: list[ToolExecution]
    gaps: list[str]
    """Checks that should have happened but could not be confirmed (see ``audit_gaps``)."""
    decision: Decision
    human: HumanDecision
    record: AuditRecord
    trail: Annotated[list[TrailEvent], operator.add]
