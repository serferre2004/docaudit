from collections import Counter
from decimal import Decimal

from docaudit.synthetic import generate_audit_cases


def test_generation_is_deterministic() -> None:
    assert generate_audit_cases(seed=3) == generate_audit_cases(seed=3)


def test_covers_every_discrepancy_and_includes_clean_cases() -> None:
    _, cases = generate_audit_cases(per_discrepancy=2, clean=6)
    counts = Counter(code for case in cases for code in case.expected)
    assert len(counts) == 7
    assert set(counts.values()) == {2}
    assert sum(not case.expected for case in cases) == 6


def test_clean_cases_really_are_clean() -> None:
    erp, cases = generate_audit_cases(clean=20, per_discrepancy=0)
    vendors = {vendor.name: vendor for vendor in erp.vendors}
    orders = {order.po_number: order for order in erp.purchase_orders}
    for case in cases:
        invoice = case.invoice
        vendor = vendors[invoice.vendor_name]
        order = orders[invoice.po_number or ""]
        assert vendor.status == "active"
        assert vendor.tax_id == invoice.vendor_tax_id
        assert invoice.total == invoice.subtotal + invoice.tax
        in_usd = invoice.total * erp.usd_per_unit[invoice.currency]
        assert in_usd < order.amount * erp.usd_per_unit[order.currency]


def test_each_seeded_discrepancy_is_really_there() -> None:
    erp, cases = generate_audit_cases(clean=0, per_discrepancy=3)
    names = {vendor.name: vendor for vendor in erp.vendors}
    orders = {order.po_number: order for order in erp.purchase_orders}
    for case in cases:
        invoice = case.invoice
        code = next(iter(case.expected))
        vendor = names.get(invoice.vendor_name)
        match code:
            case "UNKNOWN_VENDOR":
                assert vendor is None
            case "SUSPENDED_VENDOR":
                assert vendor is not None
                assert vendor.status == "suspended"
            case "TAX_ID_MISMATCH":
                assert vendor is not None
                assert vendor.tax_id != invoice.vendor_tax_id
            case "MISSING_PO":
                assert invoice.po_number is None
            case "PO_NOT_FOUND":
                assert invoice.po_number not in orders
            case "PO_AMOUNT_EXCEEDED":
                order = orders[invoice.po_number or ""]
                in_usd = invoice.total * erp.usd_per_unit[invoice.currency]
                assert in_usd > order.amount * erp.usd_per_unit[order.currency]
            case "TOTAL_MISMATCH":
                assert invoice.total != invoice.subtotal + invoice.tax


def test_invoices_are_only_wrong_in_the_expected_way() -> None:
    erp, cases = generate_audit_cases()
    for case in cases:
        assert all(item.amount >= Decimal("0") for item in case.invoice.line_items)
        if "TOTAL_MISMATCH" not in case.expected:
            assert case.invoice.total == case.invoice.subtotal + case.invoice.tax
    assert len({case.invoice.invoice_number for case in cases}) == len(cases)
    assert erp.usd_per_unit["USD"] == Decimal("1")
