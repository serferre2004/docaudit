"""The graph's nodes.   >>> YOUR TURN <<<

Tests that define "done": tests/agent/test_nodes.py (and test_graph.py, which runs them together)

A node is a function ``state -> dict``: it reads what it needs and returns ONLY the keys it changes.
LangGraph merges the dict into the state and saves a checkpoint. Every node also returns one
``TrailEvent`` in ``"trail"``, which the reducer appends: that list is the audit trail.

The nodes are methods so they can share what doesn't change between documents (the provider, the
ERP data, the policy) without putting it in the state. The state only holds what a checkpoint
needs to resume.
"""

from typing import Any

from langgraph.types import interrupt

from docaudit.agent.policy import audit_gaps, decide, total_in_usd
from docaudit.agent.schemas import AuditRecord, Decision, HumanDecision, Policy, TrailEvent
from docaudit.agent.state import AgentState
from docaudit.extraction import ExtractionError, InvoiceExtractor
from docaudit.extraction.schemas import Invoice
from docaudit.llm.base import LLMProvider
from docaudit.tools import ToolLoopError, audit_invoice
from docaudit.tools.schemas import ErpData


def _require_invoice(state: AgentState) -> Invoice:
    """The routing guarantees an invoice here; reaching this without one is a bug in the graph."""
    invoice = state.get("invoice")
    if invoice is None:
        raise ValueError("this step needs an extracted invoice")
    return invoice


class AuditNodes:
    """The five steps of the workflow.

    ``__init__(self, provider, erp, policy=Policy())`` stores its arguments.
    """

    def __init__(self, provider: LLMProvider, erp: ErpData, policy: Policy | None = None) -> None:
        self.provider = provider
        self.erp = erp
        self.policy = policy or Policy()

    async def extract(self, state: AgentState) -> dict[str, Any]:
        """Document text -> ``Invoice``, with Lesson 2's ``InvoiceExtractor``.

        Success: ``invoice`` and a trail event that says how many attempts it took.
        ``ExtractionError``: ``invoice=None`` and ``error`` explaining it. Don't raise: an
        unreadable document is a normal outcome, and it goes to a human.
        ``LLMError`` propagates: the provider is down, and the run should stop where it is.
        """
        extractor = InvoiceExtractor(self.provider)
        try:
            result = await extractor.extract(state["document"])
        except ExtractionError as exc:
            return {
                "invoice": None,
                "error": str(exc),
                "trail": [TrailEvent(step="extract", detail="Extraction failed")],
            }
        return {
            "invoice": result.invoice,
            "trail": [
                TrailEvent(step="extract", detail=f"Extracted after {result.attempts} attempts")
            ],
        }

    async def audit(self, state: AgentState) -> dict[str, Any]:
        """Run Lesson 4's ``audit_invoice`` on the extracted invoice.

        Success: ``findings``, ``executions`` (``result.loop.executions``) and ``gaps`` from
        ``audit_gaps``, plus a trail event that counts the tool calls and names the findings.
        ``ToolLoopError``: ``error`` explaining it. ``LLMError`` propagates.
        """
        invoice = _require_invoice(state)
        try:
            result = await audit_invoice(self.provider, self.erp, invoice)
        except ToolLoopError as exc:
            return {
                "error": str(exc),
                "trail": [TrailEvent(step="audit", detail="Tool loop stopped without an answer")],
            }
        executions = result.loop.executions
        codes = ", ".join(f.code for f in result.findings) or "none"
        return {
            "findings": result.findings,
            "executions": executions,
            "gaps": audit_gaps(invoice, executions),
            "trail": [
                TrailEvent(
                    step="audit",
                    detail=f"{len(executions)} tool calls in {result.loop.turns} turns. "
                    f"Findings: {codes}",
                )
            ],
        }

    def decide(self, state: AgentState) -> dict[str, Any]:
        """Apply the policy.

        If ``state`` has an ``error``, escalate with that error as the only reason. Otherwise call
        ``decide`` with the findings, the gaps and ``total_in_usd``. Returns ``decision`` and a
        trail event with the action and the reasons.
        """
        if "error" in state:
            decision = Decision(action="escalate", reasons=(state["error"],))
        else:
            decision = decide(
                state["findings"],
                state["gaps"],
                total_in_usd(_require_invoice(state), self.erp),
                self.policy,
            )

        return {
            "decision": decision,
            "trail": [
                TrailEvent(
                    step="decide",
                    detail=f"Decision: {decision.action}. Reasons: {decision.reasons}",
                )
            ],
        }

    def human_review(self, state: AgentState) -> dict[str, Any]:
        """Stop the graph until a person decides.

        Call ``interrupt(payload)`` with a JSON-friendly dict a reviewer needs: ``document_id``,
        ``invoice_number`` (or ``None``) and the decision's ``reasons`` as a list. When the graph
        is resumed, ``interrupt`` RETURNS the value passed to ``Command(resume=...)``: validate it
        with ``HumanDecision.model_validate``. Return ``human`` and a trail event naming the
        reviewer, their action and their note.

        Careful: on resume, LangGraph runs this node again FROM THE START, and this time
        ``interrupt`` returns instead of stopping. Anything before ``interrupt`` runs twice, so
        keep it free of side effects.
        """
        summary_dict = {
            "document_id": state["document_id"],
            "invoice_number": state["invoice"].invoice_number if state["invoice"] else None,
            "decision": state["decision"].action,
            "reasons": list(state["decision"].reasons),
        }
        human_decision = interrupt(summary_dict)
        decision = HumanDecision.model_validate(human_decision)
        return {
            "human": decision,
            "trail": [
                TrailEvent(
                    step="human_review",
                    detail=f"Reviewed by: {decision.reviewer}. Action taken: {decision.action}. "
                    f"Note: {decision.note}",
                )
            ],
        }

    def record(self, state: AgentState) -> dict[str, Any]:
        """Build the final ``AuditRecord``.

        The action comes from the human if there is one (``decided_by="human:<reviewer>"``,
        reasons = the policy's reasons plus the reviewer's note when it isn't empty), otherwise
        from the policy (``decided_by="policy"``). Reaching here with an ``escalate`` decision and
        no human is a bug in the graph: raise ``ValueError``.

        The record carries the findings, the tool calls, and the trail INCLUDING this node's own
        event, which is also returned in ``"trail"``.
        """
        decision = state["decision"]
        human = state.get("human")
        if human is None:
            if decision.action == "escalate":
                raise ValueError("Decision taken was escalate but there's no human review")
            action, decided_by, reasons = decision.action, "policy", decision.reasons
        else:
            action, decided_by = human.action, f"human:{human.reviewer}"
            reasons = decision.reasons
            if human.note:
                reasons += (human.note,)

        invoice = state.get("invoice")
        event = TrailEvent(step="record", detail=f"Final action: {action} ({decided_by})")
        audit_record = AuditRecord(
            document_id=state["document_id"],
            invoice_number=invoice.invoice_number if invoice else None,
            action=action,
            decided_by=decided_by,
            reasons=reasons,
            findings=tuple(state.get("findings", [])),
            tool_calls=tuple(state.get("executions", [])),
            trail=(*state["trail"], event),
        )
        return {"record": audit_record, "trail": [event]}
