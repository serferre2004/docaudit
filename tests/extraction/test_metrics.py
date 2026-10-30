from decimal import Decimal

import pytest

from docaudit.extraction import Invoice, compare_invoices, field_accuracy


def test_identical_invoices_are_fully_correct(invoice: Invoice) -> None:
    result = compare_invoices(invoice, invoice)
    assert list(result) == list(Invoice.model_fields)
    assert all(result.values())


def test_failed_extraction_gets_every_field_wrong(invoice: Invoice) -> None:
    assert not any(compare_invoices(invoice, None).values())


def test_strings_ignore_case_and_surrounding_whitespace(invoice: Invoice) -> None:
    actual = invoice.model_copy(update={"vendor_name": f"  {invoice.vendor_name.upper()} "})
    assert compare_invoices(invoice, actual)["vendor_name"]


def test_none_only_matches_none(invoice: Invoice) -> None:
    expected = invoice.model_copy(update={"po_number": None})
    assert compare_invoices(expected, expected)["po_number"]
    assert not compare_invoices(expected, invoice)["po_number"]
    assert not compare_invoices(invoice, expected)["po_number"]


def test_decimals_compare_by_value(invoice: Invoice) -> None:
    expected = invoice.model_copy(update={"subtotal": Decimal("100.5")})
    actual = invoice.model_copy(update={"subtotal": Decimal("100.50")})
    assert compare_invoices(expected, actual)["subtotal"]


def test_line_items_are_one_field(invoice: Invoice) -> None:
    first, *rest = invoice.line_items

    wrong_amount = invoice.model_copy(
        update={"line_items": [first.model_copy(update={"amount": first.amount + 1}), *rest]}
    )
    result = compare_invoices(invoice, wrong_amount)
    assert not result["line_items"]
    assert result["total"], "one wrong line must not affect other fields"

    shouting = invoice.model_copy(
        update={
            "line_items": [
                first.model_copy(update={"description": first.description.upper()}),
                *rest,
            ]
        }
    )
    assert compare_invoices(invoice, shouting)["line_items"]

    missing_one = invoice.model_copy(update={"line_items": rest})
    assert not compare_invoices(invoice, missing_one)["line_items"]


def test_field_accuracy_averages_over_pairs(invoice: Invoice) -> None:
    wrong_total = invoice.model_copy(update={"total": invoice.total + 1})
    report = field_accuracy([(invoice, invoice), (invoice, wrong_total), (invoice, None)])

    assert list(report) == list(Invoice.model_fields)
    assert report["total"] == pytest.approx(1 / 3)
    assert report["vendor_name"] == pytest.approx(2 / 3)


def test_field_accuracy_rejects_an_empty_dataset() -> None:
    with pytest.raises(ValueError):
        field_accuracy([])
