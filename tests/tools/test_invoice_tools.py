import json
from decimal import Decimal

import pytest

from docaudit.tools import (
    FindingLedger,
    ToolError,
    build_audit_registry,
    check_invoice_totals,
    convert_currency,
    get_purchase_order,
    lookup_vendor,
)

from .helpers import call, make_erp, make_invoice


def test_lookup_vendor_ignores_case_and_spaces() -> None:
    found = lookup_vendor(make_erp(), "  acme LTD. ")
    assert found == {"found": True, "name": "Acme Ltd.", "tax_id": "11-1", "status": "active"}


def test_unknown_vendor_is_an_answer_not_an_error() -> None:
    assert lookup_vendor(make_erp(), "Nobody") == {"found": False}


def test_get_purchase_order() -> None:
    order = get_purchase_order(make_erp(), "PO-1")
    assert order["found"] is True
    assert (order["vendor_name"], order["currency"]) == ("Acme Ltd.", "USD")
    assert Decimal(str(order["amount"])) == Decimal(500)
    assert get_purchase_order(make_erp(), "PO-404") == {"found": False}


def test_convert_currency_goes_through_usd_and_rounds_to_cents() -> None:
    result = convert_currency(make_erp(), Decimal(400000), "COP", "EUR")
    assert result["currency"] == "EUR"
    assert Decimal(str(result["amount"])) == Decimal("92.59")


def test_converting_to_the_same_currency_changes_nothing() -> None:
    result = convert_currency(make_erp(), Decimal("12.50"), "USD", "USD")
    assert Decimal(str(result["amount"])) == Decimal("12.50")


@pytest.mark.parametrize(("source", "target"), [("XYZ", "USD"), ("USD", "XYZ")])
def test_unknown_currency_raises_tool_error_naming_it(source: str, target: str) -> None:
    with pytest.raises(ToolError, match="XYZ"):
        convert_currency(make_erp(), Decimal(1), source, target)


def test_totals_that_add_up() -> None:
    result = check_invoice_totals(make_invoice())
    assert result["subtotal_ok"] is True
    assert result["total_ok"] is True


def test_wrong_total_reports_both_numbers() -> None:
    result = check_invoice_totals(make_invoice(total=Decimal(150)))
    assert result["total_ok"] is False
    assert result["subtotal_ok"] is True
    assert Decimal(str(result["expected_total"])) == Decimal(108)
    assert Decimal(str(result["printed_total"])) == Decimal(150)


def test_wrong_subtotal_is_reported() -> None:
    assert check_invoice_totals(make_invoice(subtotal=Decimal(99)))["subtotal_ok"] is False


def test_ledger_is_idempotent_and_keeps_the_first() -> None:
    ledger = FindingLedger()
    assert ledger.record("INV-1", "MISSING_PO", "first") is True
    assert ledger.record("INV-1", "MISSING_PO", "second") is False
    assert [(f.code, f.detail) for f in ledger.findings] == [("MISSING_PO", "first")]


def test_ledger_keeps_different_codes_and_invoices_apart() -> None:
    ledger = FindingLedger()
    assert ledger.record("INV-1", "MISSING_PO", "a") is True
    assert ledger.record("INV-1", "TOTAL_MISMATCH", "b") is True
    assert ledger.record("INV-2", "MISSING_PO", "c") is True
    assert [f.code for f in ledger.findings] == ["MISSING_PO", "TOTAL_MISMATCH", "MISSING_PO"]


def test_registry_has_the_five_tools_described_for_the_model() -> None:
    registry = build_audit_registry(make_erp(), make_invoice(), FindingLedger())
    definitions = {d.name: d for d in registry.definitions}
    assert set(definitions) == {
        "lookup_vendor",
        "get_purchase_order",
        "convert_currency",
        "check_invoice_totals",
        "record_finding",
    }
    assert all(len(d.description) > 10 for d in definitions.values())


def test_registry_wires_the_tools_end_to_end() -> None:
    invoice = make_invoice(total=Decimal(150))
    registry = build_audit_registry(make_erp(), invoice, FindingLedger())

    vendor = registry.execute(call("lookup_vendor", name="Acme Ltd."))
    assert json.loads(vendor.content)["status"] == "active"

    order = registry.execute(call("get_purchase_order", po_number="PO-1"))
    assert json.loads(order.content)["found"] is True

    converted = registry.execute(
        call("convert_currency", amount="100", from_currency="EUR", to_currency="USD")
    )
    assert Decimal(json.loads(converted.content)["amount"]) == Decimal(108)

    assert json.loads(registry.execute(call("check_invoice_totals")).content)["total_ok"] is False


def test_record_finding_uses_the_invoice_under_audit_and_reports_repeats() -> None:
    ledger = FindingLedger()
    registry = build_audit_registry(make_erp(), make_invoice(), ledger)
    args = {"code": "MISSING_PO", "detail": "no order"}

    assert json.loads(registry.execute(call("record_finding", **args)).content) == {
        "recorded": True
    }
    repeat = json.loads(registry.execute(call("record_finding", **args)).content)
    assert repeat["recorded"] is False
    assert [f.invoice_number for f in ledger.findings] == ["INV-1"]


def test_invented_finding_code_is_rejected_before_writing() -> None:
    ledger = FindingLedger()
    registry = build_audit_registry(make_erp(), make_invoice(), ledger)
    result = registry.execute(call("record_finding", code="LOOKS_FISHY", detail="x"))
    assert result.is_error
    assert ledger.findings == []


def test_unknown_currency_reaches_the_model_as_an_error() -> None:
    registry = build_audit_registry(make_erp(), make_invoice(), FindingLedger())
    result = registry.execute(
        call("convert_currency", amount="1", from_currency="XYZ", to_currency="USD")
    )
    assert result.is_error
    assert "XYZ" in result.content
