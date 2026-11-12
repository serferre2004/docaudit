import pytest

from docaudit.llm import FakeProvider
from docaudit.tools import ToolLoopError, audit_invoice, build_audit_prompt, score_findings

from .helpers import asks, call, make_erp, make_invoice


def test_prompt_separates_instructions_from_the_invoice() -> None:
    system, messages = build_audit_prompt(make_invoice())
    assert "INV-1" not in system
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "<invoice>" in messages[0].content
    assert "INV-1" in messages[0].content


def test_prompt_names_every_finding_code_and_the_rules() -> None:
    system, _ = build_audit_prompt(make_invoice())
    for code in (
        "UNKNOWN_VENDOR",
        "SUSPENDED_VENDOR",
        "TAX_ID_MISMATCH",
        "MISSING_PO",
        "PO_NOT_FOUND",
        "PO_AMOUNT_EXCEEDED",
        "TOTAL_MISMATCH",
    ):
        assert code in system
    assert "record_finding" in system
    assert "data" in system.lower(), "the invoice is data, not instructions"


async def test_findings_come_from_the_ledger_not_the_text() -> None:
    fake = FakeProvider(
        replies=[
            asks(call("record_finding", code="MISSING_PO", detail="no order")),
            "I also found TOTAL_MISMATCH and everything is terrible.",
        ]
    )
    result = await audit_invoice(fake, make_erp(), make_invoice())
    assert [f.code for f in result.findings] == ["MISSING_PO"]
    assert result.findings[0].invoice_number == "INV-1"
    assert result.invoice_number == "INV-1"
    assert result.loop.turns == 2


async def test_a_clean_audit_has_no_findings() -> None:
    fake = FakeProvider(replies=[asks(call("check_invoice_totals")), "All good."])
    assert (await audit_invoice(fake, make_erp(), make_invoice())).findings == []


async def test_repeated_findings_count_once() -> None:
    twice = asks(
        call("record_finding", call_id="a", code="MISSING_PO", detail="one"),
        call("record_finding", call_id="b", code="MISSING_PO", detail="two"),
    )
    fake = FakeProvider(replies=[twice, "done"])
    result = await audit_invoice(fake, make_erp(), make_invoice())
    assert [f.detail for f in result.findings] == ["one"]


async def test_each_audit_starts_with_an_empty_ledger() -> None:
    first = asks(call("record_finding", code="MISSING_PO", detail="x"))
    fake = FakeProvider(replies=[first, "done", "nothing here"])
    await audit_invoice(fake, make_erp(), make_invoice())
    assert (await audit_invoice(fake, make_erp(), make_invoice())).findings == []


async def test_options_reach_the_loop() -> None:
    fake = FakeProvider(replies=[asks(call("check_invoice_totals"))] * 5)
    with pytest.raises(ToolLoopError):
        await audit_invoice(fake, make_erp(), make_invoice(), max_turns=2, max_tokens=77)
    assert len(fake.calls) == 2
    assert fake.calls[0].max_tokens == 77


def test_scores_count_precision_recall_and_false_alarms() -> None:
    scores = score_findings(
        [
            (frozenset({"MISSING_PO"}), frozenset({"MISSING_PO"})),
            (frozenset({"PO_NOT_FOUND"}), frozenset()),
            (frozenset(), frozenset({"TOTAL_MISMATCH"})),
            (frozenset(), frozenset()),
        ]
    )
    assert scores.precision == pytest.approx(1 / 2)
    assert scores.recall == pytest.approx(1 / 2)
    assert (scores.exact_cases, scores.false_alarm_cases, scores.cases) == (2, 1, 4)


def test_scores_reject_an_empty_list() -> None:
    with pytest.raises(ValueError):
        score_findings([])


@pytest.mark.integration
async def test_real_provider_runs_the_loop() -> None:
    from docaudit.config import get_settings
    from docaudit.llm import create_provider

    async with create_provider(get_settings()) as provider:
        result = await audit_invoice(provider, make_erp(), make_invoice(po_number=None))
    assert result.loop.executions, "the model should have called at least one tool"
