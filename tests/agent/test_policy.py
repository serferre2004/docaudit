from decimal import Decimal

from docaudit.agent import Policy, audit_gaps, decide, total_in_usd
from docaudit.tools import Finding, ToolExecution

from .helpers import executed, make_erp, make_invoice

VENDOR_OK = {"found": True, "name": "Acme Ltd.", "status": "active", "tax_id_matches": True}
ORDER_OK = {"found": True, "po_number": "PO-1", "amount_exceeded": False}
TOTALS_OK = {"totals_ok": True}


def complete() -> list[ToolExecution]:
    return [
        executed("lookup_vendor", VENDOR_OK, name="Acme Ltd."),
        executed("get_purchase_order", ORDER_OK, po_number="PO-1"),
        executed("check_invoice_totals", TOTALS_OK),
    ]


# --- audit_gaps --------------------------------------------------------------------


def test_a_complete_audit_has_no_gaps() -> None:
    assert audit_gaps(make_invoice(), complete()) == []


def test_vendor_name_matches_ignoring_case_and_spaces() -> None:
    runs = complete()
    runs[0] = executed("lookup_vendor", VENDOR_OK, name="  ACME ltd. ")
    assert audit_gaps(make_invoice(), runs) == []


def test_each_skipped_check_is_one_gap() -> None:
    gaps = audit_gaps(make_invoice(), [])
    assert len(gaps) == 3
    assert "Acme Ltd." in gaps[0]
    assert "PO-1" in gaps[1]


def test_checking_the_wrong_vendor_or_order_does_not_count() -> None:
    runs = [
        executed("lookup_vendor", VENDOR_OK, name="Someone Else"),
        executed("get_purchase_order", ORDER_OK, po_number="PO-999"),
        executed("check_invoice_totals", TOTALS_OK),
    ]
    assert len(audit_gaps(make_invoice(), runs)) == 2


def test_a_failed_call_is_a_gap_unless_a_later_one_succeeded() -> None:
    failed = executed("check_invoice_totals", "boom", is_error=True)
    runs = complete()
    assert len(audit_gaps(make_invoice(), [*runs[:2], failed])) == 1
    assert audit_gaps(make_invoice(), [failed, *runs]) == []


def test_no_purchase_order_means_no_order_check_is_required() -> None:
    runs = [r for r in complete() if r.call.name != "get_purchase_order"]
    assert audit_gaps(make_invoice(po_number=None), runs) == []


def test_an_order_whose_amount_could_not_be_compared_is_a_gap() -> None:
    runs = complete()
    runs[1] = executed(
        "get_purchase_order",
        {"found": True, "amount_exceeded": None, "note": "no rate for XYZ"},
        po_number="PO-1",
    )
    gaps = audit_gaps(make_invoice(), runs)
    assert len(gaps) == 1
    assert "no rate for XYZ" in gaps[0]


def test_an_order_that_was_not_found_is_a_finding_not_a_gap() -> None:
    runs = complete()
    runs[1] = executed("get_purchase_order", {"found": False}, po_number="PO-1")
    assert audit_gaps(make_invoice(), runs) == []


# --- total_in_usd ------------------------------------------------------------------


def test_total_in_usd_converts_and_rounds() -> None:
    invoice = make_invoice(currency="EUR", total=Decimal("100.005"))
    assert total_in_usd(invoice, make_erp()) == Decimal("108.01")


def test_total_in_usd_is_none_without_a_rate() -> None:
    assert total_in_usd(make_invoice(currency="XYZ"), make_erp()) is None


# --- decide ------------------------------------------------------------------------

POLICY = Policy(auto_approve_limit_usd=Decimal(1000))
FINDING = Finding(invoice_number="INV-1", code="MISSING_PO", detail="No order on the invoice.")


def test_clean_and_small_is_approved() -> None:
    decision = decide([], [], Decimal(999), POLICY)
    assert decision.action == "approve"
    assert decision.reasons


def test_exactly_at_the_limit_is_still_approved() -> None:
    assert decide([], [], Decimal(1000), POLICY).action == "approve"


def test_above_the_limit_escalates_and_says_both_amounts() -> None:
    decision = decide([], [], Decimal("1000.01"), POLICY)
    assert decision.action == "escalate"
    assert "1000.01" in decision.reasons[0]
    assert "1000" in decision.reasons[0]


def test_findings_flag_and_explain_themselves() -> None:
    decision = decide([FINDING], ["Totals never checked."], Decimal(5), POLICY)
    assert decision.action == "flag"
    assert decision.reasons == ("MISSING_PO: No order on the invoice.", "Totals never checked.")


def test_findings_win_over_a_large_amount() -> None:
    assert decide([FINDING], [], Decimal(10**6), POLICY).action == "flag"


def test_gaps_escalate_even_when_small() -> None:
    decision = decide([], ["Totals never checked."], Decimal(5), POLICY)
    assert decision.action == "escalate"
    assert decision.reasons == ("Totals never checked.",)


def test_an_unknown_amount_escalates() -> None:
    assert decide([], [], None, POLICY).action == "escalate"
