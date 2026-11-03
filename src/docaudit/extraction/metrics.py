"""Field-level accuracy.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/extraction/test_metrics.py

"80% of invoices were extracted perfectly" hides everything interesting. "``due_date`` is right
55% of the time" tells you exactly which prompt rule or schema description to improve.
"""

from collections.abc import Sequence

from docaudit.extraction.schemas import Invoice, LineItem


def compare_invoices(expected: Invoice, actual: Invoice | None) -> dict[str, bool]:
    """Return ``{field_name: is_correct}`` for every field of ``Invoice``.

    Requirements:
      * One key per field in ``Invoice.model_fields``, in that order.
      * ``actual is None`` means the extraction failed: every field is wrong.
      * Strings match ignoring surrounding whitespace and case: ``"ACME "`` == ``"acme"``.
        ``None`` only matches ``None``.
      * Everything else compares with ``==``. Note that ``Decimal("1.50") == Decimal("1.5")``.
      * ``line_items`` is ONE field: correct only if both lists have the same length and every
        item matches field by field (same string rule for ``description``).
    """
    return_dict = {field: False for field in Invoice.model_fields}
    if actual is None:
        return return_dict
    for field in Invoice.model_fields:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if isinstance(expected_value, str) and isinstance(actual_value, str):
            return_dict[field] = _compare_strings(expected_value, actual_value)
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            if len(expected_value) == len(actual_value):
                all_match = True
                for expected_item, actual_item in zip(expected_value, actual_value, strict=True):
                    for item_field in LineItem.model_fields:
                        expected_item_value = getattr(expected_item, item_field)
                        actual_item_value = getattr(actual_item, item_field)
                        if isinstance(expected_item_value, str) and isinstance(
                            actual_item_value, str
                        ):
                            if not _compare_strings(expected_item_value, actual_item_value):
                                all_match = False
                                break
                        elif expected_item_value != actual_item_value:
                            all_match = False
                            break
                    if not all_match:
                        break
                return_dict[field] = all_match
        elif expected_value == actual_value:
            return_dict[field] = True

    return return_dict


def field_accuracy(pairs: Sequence[tuple[Invoice, Invoice | None]]) -> dict[str, float]:
    """Aggregate ``compare_invoices`` over ``(expected, actual)`` pairs.

    Requirements:
      * Return ``{field_name: fraction_correct}`` with values between 0.0 and 1.0.
      * An empty ``pairs`` raises ``ValueError``: an accuracy over nothing is meaningless, and
        silently returning 0.0 or 1.0 would lie in a report.
    """
    if pairs is None or len(pairs) == 0:
        raise ValueError("Cannot compute accuracy over an empty list of pairs")
    total_pairs = len(pairs)
    field_counts = {field: 0 for field in Invoice.model_fields}
    for expected, actual in pairs:
        comparison = compare_invoices(expected, actual)
        for field, is_correct in comparison.items():
            if is_correct:
                field_counts[field] += 1
    return {field: field_counts[field] / total_pairs for field in Invoice.model_fields}


def _compare_strings(expected: str | None, actual: str | None) -> bool:
    """Compare two strings ignoring case and surrounding whitespace.

    Requirements:
      * ``None`` only matches ``None``.
      * ``"ACME "`` == ``"acme"``.
    """
    if expected is None or actual is None:
        return expected is actual
    return expected.strip().lower() == actual.strip().lower()
