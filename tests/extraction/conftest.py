import pytest

from docaudit.extraction import Invoice
from docaudit.synthetic import generate_invoices


@pytest.fixture
def invoice() -> Invoice:
    """A valid invoice (several line items, all optional fields present)."""
    return next(
        s.truth
        for s in generate_invoices(50, seed=7)
        if len(s.truth.line_items) > 1 and s.truth.po_number and s.truth.due_date
    )


@pytest.fixture
def invoice_json(invoice: Invoice) -> str:
    """What a well-behaved model would reply for ``invoice``."""
    return invoice.model_dump_json()
