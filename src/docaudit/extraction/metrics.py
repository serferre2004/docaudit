"""Field-level accuracy.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/extraction/test_metrics.py

"80% of invoices were extracted perfectly" hides everything interesting. "``due_date`` is right
55% of the time" tells you exactly which prompt rule or schema description to improve.
"""

from collections.abc import Sequence

from docaudit.extraction.schemas import Invoice


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
    raise NotImplementedError


def field_accuracy(pairs: Sequence[tuple[Invoice, Invoice | None]]) -> dict[str, float]:
    """Aggregate ``compare_invoices`` over ``(expected, actual)`` pairs.

    Requirements:
      * Return ``{field_name: fraction_correct}`` with values between 0.0 and 1.0.
      * An empty ``pairs`` raises ``ValueError``: an accuracy over nothing is meaningless, and
        silently returning 0.0 or 1.0 would lie in a report.
    """
    raise NotImplementedError
