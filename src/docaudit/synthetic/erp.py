"""A fictional company's systems, plus invoices with seeded discrepancies.

Each ``AuditCase`` says which findings a correct audit must record, which is what makes measuring
the auditor possible. Clean invoices are included on purpose: an auditor that flags everything
would otherwise look perfect.

Generation is deterministic: the same ``seed`` always gives the same cases.
"""

import random
from dataclasses import dataclass
from decimal import Decimal

from docaudit.extraction.schemas import Invoice
from docaudit.synthetic.invoices import VENDORS, generate_invoices
from docaudit.tools.schemas import ErpData, FindingCode, PurchaseOrder, Vendor

USD_PER_UNIT = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "COP": Decimal("0.00025")}

SUSPENDED_VENDOR = ("Orion Freight Ltd.", "GB 771 0284 19")
UNKNOWN_VENDOR = "Zenith Trading LLC"

_CENT = Decimal("0.01")
_DISCREPANCIES: tuple[FindingCode, ...] = (
    "UNKNOWN_VENDOR",
    "SUSPENDED_VENDOR",
    "TAX_ID_MISMATCH",
    "MISSING_PO",
    "PO_NOT_FOUND",
    "PO_AMOUNT_EXCEEDED",
    "TOTAL_MISMATCH",
)


@dataclass(frozen=True)
class AuditCase:
    id: str
    invoice: Invoice
    expected: frozenset[str]


def generate_audit_cases(
    *, seed: int = 0, per_discrepancy: int = 2, clean: int = 6
) -> tuple[ErpData, tuple[AuditCase, ...]]:
    """Return the company's data and the cases to audit against it, clean ones first."""
    rng = random.Random(seed)
    kinds: list[FindingCode | None] = [None] * clean
    for code in _DISCREPANCIES:
        kinds += [code] * per_discrepancy

    vendors = [Vendor(name, tax_id, "active") for name, tax_id, _ in VENDORS]
    vendors.append(Vendor(*SUSPENDED_VENDOR, "suspended"))
    orders: list[PurchaseOrder] = []
    cases: list[AuditCase] = []

    bases = generate_invoices(len(kinds), seed=seed)
    for index, (kind, sample) in enumerate(zip(kinds, bases, strict=True)):
        invoice = _clean_copy(sample.truth, index)
        match kind:
            case "UNKNOWN_VENDOR":
                invoice = invoice.model_copy(update={"vendor_name": UNKNOWN_VENDOR})
            case "SUSPENDED_VENDOR":
                name, tax_id = SUSPENDED_VENDOR
                invoice = invoice.model_copy(update={"vendor_name": name, "vendor_tax_id": tax_id})
            case "TAX_ID_MISMATCH":
                invoice = invoice.model_copy(update={"vendor_tax_id": "00-0000000"})
            case "TOTAL_MISMATCH":
                wrong = invoice.total + Decimal(rng.randint(1, 50))
                invoice = invoice.model_copy(update={"total": wrong})
            case "MISSING_PO":
                invoice = invoice.model_copy(update={"po_number": None})
            case _:
                pass

        if kind not in ("MISSING_PO", "PO_NOT_FOUND"):
            orders.append(_order_for(rng, invoice, exceeded=kind == "PO_AMOUNT_EXCEEDED"))
        expected = frozenset({kind}) if kind else frozenset[str]()
        cases.append(AuditCase(id=f"case-{index:02d}", invoice=invoice, expected=expected))

    erp = ErpData(tuple(vendors), tuple(orders), USD_PER_UNIT)
    return erp, tuple(cases)


def _clean_copy(invoice: Invoice, index: int) -> Invoice:
    """Remove what ``generate_invoices`` varies, so each case has only its own discrepancy."""
    tax_ids = {name: tax_id for name, tax_id, _ in VENDORS}
    subtotal = sum((item.amount for item in invoice.line_items), Decimal("0"))
    return invoice.model_copy(
        update={
            "invoice_number": f"AUD-2026-{index:04d}",
            "vendor_tax_id": tax_ids[invoice.vendor_name],
            "po_number": f"PO-{50000 + index}",
            "subtotal": subtotal,
            "total": subtotal + invoice.tax,
        }
    )


def _order_for(rng: random.Random, invoice: Invoice, *, exceeded: bool) -> PurchaseOrder:
    """An order the invoice clearly fits or clearly exceeds, half of them in another currency."""
    assert invoice.po_number is not None
    factor = Decimal("0.7") if exceeded else Decimal("1.3")
    currency = "USD" if rng.random() < 0.5 else invoice.currency
    in_usd = invoice.total * USD_PER_UNIT[invoice.currency] / USD_PER_UNIT[currency]
    return PurchaseOrder(
        po_number=invoice.po_number,
        vendor_name=invoice.vendor_name,
        currency=currency,
        amount=(in_usd * factor).quantize(_CENT),
    )
