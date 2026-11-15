"""The whole graph, with a scripted model and a real (in-memory or SQLite) checkpointer."""

from decimal import Decimal
from pathlib import Path

import pytest

from docaudit.agent import (
    AgentState,
    AuditNodes,
    Decision,
    HumanDecision,
    Policy,
    build_graph,
    memory_checkpointer,
    resume_document,
    route_after_decide,
    route_after_extract,
    run_document,
    sqlite_checkpointer,
    thread_config,
)
from docaudit.llm import FakeProvider, LLMError, LLMResponse

from .helpers import asks, call, clean_audit, make_erp, make_invoice

DOC = "INVOICE INV-1 from Acme Ltd."
ANA = HumanDecision(action="approve", reviewer="ana", note="Budget confirmed.")


def scripted(*replies: str | LLMResponse) -> FakeProvider:
    return FakeProvider(replies=list(replies))


def happy_path() -> list[str | LLMResponse]:
    return [make_invoice().model_dump_json(), *clean_audit()]


def app_for(fake: FakeProvider, limit: int = 3000):  # type: ignore[no-untyped-def]
    nodes = AuditNodes(fake, make_erp(), Policy(auto_approve_limit_usd=Decimal(limit)))
    return build_graph(nodes, memory_checkpointer())


# --- routing -----------------------------------------------------------------------


def test_routing_after_extract() -> None:
    assert route_after_extract({"invoice": make_invoice()}) == "audit"
    assert route_after_extract({"invoice": None}) == "decide"


def test_routing_after_decide() -> None:
    for action, expected in (
        ("approve", "record"),
        ("flag", "record"),
        ("escalate", "human_review"),
    ):
        state: AgentState = {"decision": Decision(action=action, reasons=("x",))}  # type: ignore[arg-type]
        assert route_after_decide(state) == expected


def test_the_graph_has_the_five_nodes() -> None:
    app = app_for(scripted())
    assert set(app.get_graph().nodes) >= {"extract", "audit", "decide", "human_review", "record"}


# --- runs that finish on their own -------------------------------------------------


async def test_a_clean_small_invoice_is_approved_by_policy() -> None:
    outcome = await run_document(app_for(scripted(*happy_path())), "doc-1", DOC)
    assert outcome.status == "done"
    assert outcome.record is not None
    assert (outcome.record.action, outcome.record.decided_by) == ("approve", "policy")
    assert [e.step for e in outcome.record.trail] == ["extract", "audit", "decide", "record"]
    assert len(outcome.record.tool_calls) == 3


async def test_a_finding_is_flagged_without_asking_anyone() -> None:
    fake = scripted(
        make_invoice().model_dump_json(),
        asks(call("record_finding", code="MISSING_PO", detail="none")),
        "done",
    )
    outcome = await run_document(app_for(fake), "doc-1", DOC)
    assert outcome.status == "done"
    assert outcome.record is not None
    assert outcome.record.action == "flag"
    assert outcome.record.reasons[0].startswith("MISSING_PO")


# --- human in the loop -------------------------------------------------------------


async def test_an_escalation_waits_for_a_human_then_records_their_decision() -> None:
    app = app_for(scripted(*happy_path()), limit=100)
    waiting = await run_document(app, "doc-1", DOC)
    assert waiting.status == "waiting_for_human"
    assert waiting.decision is not None and waiting.decision.action == "escalate"
    assert waiting.record is None

    snapshot = await app.aget_state(thread_config("doc-1"))
    payload = snapshot.interrupts[0].value
    assert payload["document_id"] == "doc-1"
    assert payload["invoice_number"] == "INV-1"
    assert payload["reasons"] == list(waiting.decision.reasons)

    done = await resume_document(app, "doc-1", ANA)
    assert done.status == "done"
    assert done.record is not None
    assert (done.record.action, done.record.decided_by) == ("approve", "human:ana")
    assert [e.step for e in done.record.trail][-2:] == ["human_review", "record"]


async def test_an_unreadable_document_goes_to_a_human() -> None:
    app = app_for(scripted("garbage", "garbage", "garbage"))
    outcome = await run_document(app, "doc-1", DOC)
    assert outcome.status == "waiting_for_human"
    assert outcome.decision is not None
    assert "xtraction" in outcome.decision.reasons[0]


async def test_resuming_a_thread_that_is_not_waiting_is_refused() -> None:
    app = app_for(scripted(*happy_path()))
    with pytest.raises(ValueError):
        await resume_document(app, "never-started", ANA)
    await run_document(app, "doc-1", DOC)
    with pytest.raises(ValueError):
        await resume_document(app, "doc-1", ANA)


async def test_a_human_can_resume_from_another_process(tmp_path: Path) -> None:
    """The checkpoint is on disk: a new graph, with a new connection, picks it up."""
    db = tmp_path / "checkpoints.sqlite"
    nodes = AuditNodes(scripted(*happy_path()), make_erp(), Policy(auto_approve_limit_usd=100))
    async with sqlite_checkpointer(db) as saver:
        first = await run_document(build_graph(nodes, saver), "doc-1", DOC)
    assert first.status == "waiting_for_human"

    later = AuditNodes(scripted(), make_erp(), Policy(auto_approve_limit_usd=100))
    async with sqlite_checkpointer(db) as saver:
        done = await resume_document(build_graph(later, saver), "doc-1", ANA)
    assert done.record is not None
    assert done.record.decided_by == "human:ana"


# --- checkpoints as crash recovery -------------------------------------------------


async def test_after_a_crash_the_thread_continues_without_redoing_finished_steps() -> None:
    """Not your code to write, but your graph has to allow it: this is why checkpoints exist."""
    fake = scripted(make_invoice().model_dump_json())  # the audit will run out of replies
    app = app_for(fake)
    with pytest.raises(LLMError):
        await run_document(app, "doc-1", DOC)
    assert len(fake.calls) == 2, "extraction succeeded, then the audit's first call failed"

    fake.replies.extend(clean_audit())
    final = await app.ainvoke(None, thread_config("doc-1"))  # None = continue, don't restart
    assert final["record"].action == "approve"
    assert len(fake.calls) == 2 + 4, "extraction did NOT run again"
