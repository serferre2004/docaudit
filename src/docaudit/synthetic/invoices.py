"""Synthetic invoices with known ground truth.

Real invoices are confidential, so we generate fictional ones. Each comes with the ``Invoice``
it SHOULD extract to, which is what makes measuring accuracy possible.

The text deliberately varies the way real documents do: three layouts, three date formats,
US vs. European number formats, English and Spanish labels, optional fields that are sometimes
missing, and a few invoices whose printed total is wrong (the ground truth keeps the printed,
wrong value: extraction records what the document says).

Generation is deterministic: the same ``seed`` always gives the same invoices.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from docaudit.extraction.schemas import Invoice, LineItem

LAYOUTS = ("labeled", "table_es", "letter")

_CENT = Decimal("0.01")
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip

_VENDORS = (
    ("Andes Logistics S.A.S.", "900.482.117-3", "ALS"),
    ("Northwind Office Supply Ltd.", "GB 284 7710 52", "NOS"),
    ("Cobalt Data Services Inc.", "84-1937265", "CDS"),
    ("Brisa Facility Management", "901.336.904-1", "BFM"),
    ("Helix Industrial Parts GmbH", "DE 318 552 907", "HIP"),
)

_CATALOG = (
    ("Freight transport Bogota-Medellin", 180, 900),
    ("Warehouse storage (pallet/month)", 12, 40),
    ("A4 printer paper, box of 5 reams", 18, 35),
    ("Ergonomic office chair", 120, 480),
    ("Cloud hosting - standard plan", 50, 400),
    ("Data migration consulting (hour)", 60, 150),
    ("Window cleaning service", 80, 220),
    ("Hydraulic pump seal kit", 25, 140),
    ("Stainless steel bolts M8 (box of 100)", 9, 30),
)

_TAX_RATE = {"USD": Decimal("0.08"), "EUR": Decimal("0.21"), "COP": Decimal("0.19")}


@dataclass(frozen=True)
class SyntheticInvoice:
    id: str
    layout: str
    text: str
    truth: Invoice
    discrepancies: tuple[str, ...] = ()


def generate_invoices(n: int, *, seed: int = 0) -> list[SyntheticInvoice]:
    """Return ``n`` invoices, cycling through every layout."""
    rng = random.Random(seed)
    return [_generate_one(rng, i, LAYOUTS[i % len(LAYOUTS)]) for i in range(n)]


def _generate_one(rng: random.Random, index: int, layout: str) -> SyntheticInvoice:
    vendor_name, tax_id, prefix = rng.choice(_VENDORS)
    currency = rng.choice(tuple(_TAX_RATE))
    issue = date(2026, 1, 1) + timedelta(days=rng.randrange(300))

    items = []
    for description, low, high in rng.sample(_CATALOG, k=rng.randint(1, 4)):
        quantity = Decimal(rng.randint(1, 12))
        unit_price = Decimal(rng.randint(low * 100, high * 100)) / 100
        if currency == "COP":
            unit_price = (unit_price * 4000).quantize(Decimal("1"))
        amount = (quantity * unit_price).quantize(_CENT)
        items.append(
            LineItem(
                description=description, quantity=quantity, unit_price=unit_price, amount=amount
            )
        )

    subtotal = sum((item.amount for item in items), Decimal("0"))
    tax = (subtotal * _TAX_RATE[currency]).quantize(_CENT) if rng.random() < 0.8 else Decimal("0")
    total = subtotal + tax
    discrepancies: tuple[str, ...] = ()
    if rng.random() < 0.15:
        total += Decimal(rng.choice((-1, 1)) * rng.randint(1, 50))
        discrepancies = ("total_mismatch",)

    truth = Invoice(
        invoice_number=f"{prefix}-2026-{rng.randint(1, 9999):04d}",
        vendor_name=vendor_name,
        vendor_tax_id=tax_id if rng.random() < 0.85 else None,
        po_number=f"PO-{rng.randint(10000, 99999)}" if rng.random() < 0.7 else None,
        issue_date=issue,
        due_date=issue + timedelta(days=rng.choice((15, 30, 45))) if rng.random() < 0.8 else None,
        currency=currency,
        line_items=items,
        subtotal=subtotal,
        tax=tax,
        total=total,
    )
    render = {"labeled": _render_labeled, "table_es": _render_table_es, "letter": _render_letter}
    return SyntheticInvoice(
        id=f"inv-{index:03d}",
        layout=layout,
        text=render[layout](truth),
        truth=truth,
        discrepancies=discrepancies,
    )


# --- Formatting helpers ------------------------------------------------------------


def _us(amount: Decimal) -> str:
    return f"{amount:,.2f}"


def _eu(amount: Decimal) -> str:
    return _us(amount).replace(",", "_").replace(".", ",").replace("_", ".")


def _long_date(d: date) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def _qty(quantity: Decimal) -> str:
    return str(int(quantity))


# --- Layouts -----------------------------------------------------------------------


def _render_labeled(inv: Invoice) -> str:
    lines = [
        "INVOICE",
        "",
        f"Invoice No: {inv.invoice_number}",
        f"Vendor: {inv.vendor_name}",
    ]
    if inv.vendor_tax_id:
        lines.append(f"Tax ID: {inv.vendor_tax_id}")
    if inv.po_number:
        lines.append(f"Purchase Order: {inv.po_number}")
    lines.append(f"Issue date: {inv.issue_date.isoformat()}")
    if inv.due_date:
        lines.append(f"Due date: {inv.due_date.isoformat()}")
    lines += ["", "Description | Qty | Unit price | Amount"]
    lines += [
        f"{it.description} | {_qty(it.quantity)} | {_us(it.unit_price)} | {_us(it.amount)}"
        for it in inv.line_items
    ]
    lines += [
        "",
        f"Subtotal: {_us(inv.subtotal)} {inv.currency}",
        f"Tax: {_us(inv.tax)} {inv.currency}",
        f"TOTAL DUE: {_us(inv.total)} {inv.currency}",
    ]
    return "\n".join(lines)


def _render_table_es(inv: Invoice) -> str:
    header = f"FACTURA DE VENTA N.º {inv.invoice_number}"
    lines = [header, "=" * len(header), f"Proveedor: {inv.vendor_name}"]
    if inv.vendor_tax_id:
        lines.append(f"NIT / VAT: {inv.vendor_tax_id}")
    lines.append(f"Fecha de emisión: {inv.issue_date:%d/%m/%Y}")
    if inv.due_date:
        lines.append(f"Fecha de vencimiento: {inv.due_date:%d/%m/%Y}")
    if inv.po_number:
        lines.append(f"Orden de compra: {inv.po_number}")
    lines += [
        f"Moneda: {inv.currency}",
        "",
        f"{'Cant.':>5}  {'Descripción':<40} {'V. unitario':>14} {'V. total':>14}",
    ]
    lines += [
        f"{_qty(it.quantity):>5}  {it.description:<40}"
        f" {_eu(it.unit_price):>14} {_eu(it.amount):>14}"
        for it in inv.line_items
    ]
    lines += [
        "",
        f"{'Subtotal:':>62} {_eu(inv.subtotal):>14}",
        f"{'IVA:':>62} {_eu(inv.tax):>14}",
        f"{'TOTAL A PAGAR:':>62} {_eu(inv.total):>14}",
    ]
    return "\n".join(lines)


def _render_letter(inv: Invoice) -> str:
    tax_id = f" (tax id {inv.vendor_tax_id})" if inv.vendor_tax_id else ""
    items = "\n".join(
        f"  - {_qty(it.quantity)} x {it.description} at {_us(it.unit_price)} each"
        f" = {_us(it.amount)}"
        for it in inv.line_items
    )
    due = f" by {_long_date(inv.due_date)}" if inv.due_date else ""
    po = f"\nThis invoice refers to purchase order {inv.po_number}." if inv.po_number else ""
    return (
        "Dear customer,\n\n"
        f"{inv.vendor_name}{tax_id} hereby bills you under invoice number "
        f"{inv.invoice_number}, issued on {_long_date(inv.issue_date)}, for the following:\n\n"
        f"{items}\n\n"
        f"That makes a subtotal of {_us(inv.subtotal)} plus {_us(inv.tax)} in taxes.\n"
        f"Please remit {inv.currency} {_us(inv.total)}{due}.{po}\n\n"
        "Kind regards,\n"
        f"Accounts Receivable, {inv.vendor_name}"
    )
