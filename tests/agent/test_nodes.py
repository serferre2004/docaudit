"""Nodes tested one at a time, by calling them with a hand-made state."""

from decimal import Decimal

import pytest

from docaudit.agent import AgentState, AuditNodes, Decision, HumanDecision, Policy, TrailEvent
from docaudit.llm import FakeProvider, LLMError
from docaudit.tools import Finding

from .helpers import asks, call, clean_audit, executed, make_erp, make_invoice

DOC = "INVOICE INV-1 from Acme Ltd."


def nodes(*replies: object, limit: int = 3000) -> AuditNodes:
    fake = FakeProvider(replies=list(replies))  # type: ignore[arg-type]
    return AuditNodes(fake, make_erp(), Policy(auto_approve_limit_usd=Decimal(limit)))


# --- extract -----------------------------------------------------------------------


async def test_extract_returns_the_invoice_and_one_trail_event() -> None:
    update = await nodes(make_invoice().model_dump_json()).extract({"document": DOC})
    assert update["invoice"] == make_invoice()
    assert [e.step for e in update["trail"]] == ["extract"]
    assert "error" not in update, "return only the keys you change"


async def test_an_unreadable_document_is_an_error_not_an_exception() -> None:
    update = await nodes("nope", "still nope", "never").extract({"document": DOC})
    assert update["invoice"] is None
    assert "Extraction failed" in update["error"] or "extraction" in update["error"].lower()
    assert update["trail"][0].step == "extract"


async def test_extract_lets_provider_failures_propagate() -> None:
    with pytest.raises(LLMError):
        await nodes().extract({"document": DOC})


# --- audit -------------------------------------------------------------------------


async def test_audit_returns_findings_executions_and_gaps() -> None:
    update = await nodes(*clean_audit()).audit({"invoice": make_invoice()})
    assert update["findings"] == []
    assert [e.call.name for e in update["executions"]] == [
        "lookup_vendor",
        "get_purchase_order",
        "check_invoice_totals",
    ]
    assert update["gaps"] == []
    assert update["trail"][0].step == "audit"


async def test_audit_reports_what_it_skipped() -> None:
    update = await nodes(asks(call("check_invoice_totals")), "done").audit(
        {"invoice": make_invoice()}
    )
    assert len(update["gaps"]) == 2


async def test_a_loop_that_never_ends_is_an_error_not_an_exception() -> None:
    forever = [asks(call("check_invoice_totals"))] * 12
    update = await nodes(*forever).audit({"invoice": make_invoice()})
    assert "error" in update
    assert update["trail"][0].step == "audit"


# --- decide ------------------------------------------------------------------------


def test_decide_applies_the_policy() -> None:
    state: AgentState = {"invoice": make_invoice(), "findings": [], "gaps": []}
    update = nodes().decide(state)
    assert update["decision"].action == "approve"
    assert update["trail"][0].step == "decide"


def test_decide_uses_the_policy_limit() -> None:
    state: AgentState = {"invoice": make_invoice(), "findings": [], "gaps": []}
    assert nodes(limit=100).decide(state)["decision"].action == "escalate"


def test_an_error_escalates_with_the_error_as_the_reason() -> None:
    state: AgentState = {"invoice": None, "error": "Extraction failed: garbage"}
    decision = nodes().decide(state)["decision"]
    assert decision == Decision(action="escalate", reasons=("Extraction failed: garbage",))


# --- record ------------------------------------------------------------------------

FINDING = Finding(invoice_number="INV-1", code="MISSING_PO", detail="none")


def test_record_by_policy() -> None:
    state: AgentState = {
        "document_id": "doc-1",
        "invoice": make_invoice(),
        "findings": [FINDING],
        "executions": [executed("check_invoice_totals", {"totals_ok": True})],
        "decision": Decision(action="flag", reasons=("MISSING_PO: none",)),
        "trail": [TrailEvent(step="decide", detail="flag")],
    }
    update = nodes().record(state)
    record = update["record"]
    assert (record.action, record.decided_by) == ("flag", "policy")
    assert record.invoice_number == "INV-1"
    assert record.findings == (FINDING,)
    assert len(record.tool_calls) == 1
    assert [e.step for e in record.trail] == ["decide", "record"]
    assert [e.step for e in update["trail"]] == ["record"]


def test_record_by_a_human_keeps_the_reasons_and_adds_the_note() -> None:
    state: AgentState = {
        "document_id": "doc-1",
        "invoice": make_invoice(),
        "decision": Decision(action="escalate", reasons=("Above the limit.",)),
        "human": HumanDecision(action="approve", reviewer="ana", note="Budget confirmed."),
        "trail": [],
    }
    record = nodes().record(state)["record"]
    assert (record.action, record.decided_by) == ("approve", "human:ana")
    assert record.reasons[0] == "Above the limit."
    assert "Budget confirmed." in record.reasons[-1]


def test_recording_an_escalation_without_a_human_is_a_bug() -> None:
    state: AgentState = {
        "document_id": "doc-1",
        "decision": Decision(action="escalate", reasons=("x",)),
        "trail": [],
    }
    with pytest.raises(ValueError):
        nodes().record(state)
