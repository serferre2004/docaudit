"""The auditor's tools.   >>> YOUR TURN: implement the handlers, the ledger and the registry. <<<

Tests that define "done": tests/tools/test_invoice_tools.py

The input models are provided because they are part of the prompt: every ``description`` below
reaches the model. Notice what is NOT an argument: ``check_invoice_totals`` takes nothing, because
the tool already holds the invoice. Numbers the model doesn't have to retype can't be mistyped.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from docaudit.extraction.schemas import Invoice
from docaudit.tools.registry import ToolRegistry
from docaudit.tools.schemas import ErpData, Finding, FindingCode, Tool, ToolError


class LookupVendorInput(BaseModel):
    name: str = Field(description="Vendor name exactly as printed on the invoice")


class GetPurchaseOrderInput(BaseModel):
    po_number: str = Field(description="Purchase order number, e.g. 'PO-12345'")


class ConvertCurrencyInput(BaseModel):
    amount: Decimal = Field(description="The amount to convert, as a plain number")
    from_currency: str = Field(description="ISO 4217 code of the amount, e.g. 'COP'")
    to_currency: str = Field(description="ISO 4217 code to convert to, e.g. 'USD'")


class CheckInvoiceTotalsInput(BaseModel):
    """Takes no arguments: the tool already holds the invoice under audit."""


class RecordFindingInput(BaseModel):
    code: FindingCode = Field(description="The kind of problem found")
    detail: str = Field(description="One sentence with the evidence, including the numbers")


def lookup_vendor(erp: ErpData, name: str) -> dict[str, Any]:
    """Find a vendor by name, ignoring case and surrounding whitespace.

    Found: ``{"found": True, "name": ..., "tax_id": ..., "status": ...}``. Not found:
    ``{"found": False}``. Not finding a vendor is a real answer, NOT an error: an unknown vendor is
    exactly what an auditor is looking for.
    """
    for vendor in erp.vendors:
        if vendor.name.strip().lower() == name.strip().lower():
            return {
                "found": True,
                "name": vendor.name,
                "tax_id": vendor.tax_id,
                "status": vendor.status,
            }
    return {"found": False}


def get_purchase_order(erp: ErpData, po_number: str) -> dict[str, Any]:
    """Find a purchase order by exact number.

    Found: ``{"found": True, "po_number": ..., "vendor_name": ..., "currency": ..., "amount": ...}``
    Not found: ``{"found": False}``.
    """
    for po in erp.purchase_orders:
        if po.po_number == po_number:
            return {
                "found": True,
                "po_number": po_number,
                "vendor_name": po.vendor_name,
                "currency": po.currency,
                "amount": po.amount,
            }
    return {"found": False}


def convert_currency(
    erp: ErpData, amount: Decimal, from_currency: str, to_currency: str
) -> dict[str, Any]:
    """Convert through ``erp.usd_per_unit``: ``amount * rate[from] / rate[to]``, rounded to cents.

    Returns ``{"amount": ..., "currency": to_currency}``. A currency with no rate raises
    ``ToolError`` naming it. That one IS an error: without a rate, no answer is possible.
    """
    rate = erp.usd_per_unit
    if from_currency not in rate:
        raise ToolError(f"Currency {from_currency} does not have a rate")
    if to_currency not in rate:
        raise ToolError(f"Currency {to_currency} does not have a rate")
    amount = round(amount * rate[from_currency] / rate[to_currency], 2)
    return {"amount": amount, "currency": to_currency}


def check_invoice_totals(invoice: Invoice) -> dict[str, Any]:
    """Redo the invoice's arithmetic. Models are unreliable at this, and code isn't.

    Returns ``{"subtotal_ok": ..., "total_ok": ..., "expected_total": ..., "printed_total": ...}``.
    ``subtotal_ok``: the line amounts add up to ``subtotal``. ``expected_total`` is the sum of the
    line amounts plus ``tax``, and ``total_ok`` compares it with ``total``.
    """
    subtotal: Decimal = Decimal("0.0")
    for item in invoice.line_items:
        subtotal += item.amount
    total = subtotal + invoice.tax
    return {
        "subtotal_ok": subtotal == invoice.subtotal,
        "total_ok": total == invoice.total,
        "expected_total": total,
        "printed_total": invoice.total,
    }


class FindingLedger:
    """Where findings are written. The one tool that changes something.

    Requirements:
      * ``record(invoice_number, code, detail)`` stores a ``Finding`` and returns ``True``.
      * Recording the same ``invoice_number`` and ``code`` again keeps the first one, changes
        nothing and returns ``False``.
      * ``findings`` (a property) lists them in the order they were first recorded.

    Why idempotent? Models repeat calls, and a retry can resend one. Doing it twice must leave the
    same result as doing it once. Lookups are naturally safe to repeat, writes aren't, so writes
    are where this has to be designed in.
    """

    def __init__(self) -> None:
        self._findings: dict[tuple[str, FindingCode], Finding] = {}

    def record(self, invoice_number: str, code: FindingCode, detail: str) -> bool:
        if (invoice_number, code) in self._findings:
            return False
        new_finding = Finding(invoice_number=invoice_number, code=code, detail=detail)
        self._findings[(invoice_number, code)] = new_finding
        return True

    @property
    def findings(self) -> list[Finding]:
        return list(self._findings.values())


def build_audit_registry(erp: ErpData, invoice: Invoice, ledger: FindingLedger) -> ToolRegistry:
    """Wire the five tools above into a registry for auditing ``invoice``.

    Requirements:
      * Tool names: ``lookup_vendor``, ``get_purchase_order``, ``convert_currency``,
        ``check_invoice_totals``, ``record_finding``, each with its input model.
      * Each handler adapts a validated input to the functions above. ``erp``, ``invoice`` and
        ``ledger`` are captured here, so the model never passes them.
      * ``record_finding`` returns ``{"recorded": True}`` or, for a repeat,
        ``{"recorded": False, "reason": "already recorded"}``.
      * Write each ``description`` for the model: what the tool does, and when to use it.
    """

    def handle_lookup_vendor(args: LookupVendorInput) -> dict[str, Any]:
        result = lookup_vendor(erp, args.name)
        if result["found"]:
            # The comparison happens here, in code, so the model only has to read a boolean.
            printed = (invoice.vendor_tax_id or "").strip()
            result["tax_id_matches"] = result["tax_id"].strip() == printed
        return result

    def handle_get_purchase_order(args: GetPurchaseOrderInput) -> dict[str, Any]:
        result = get_purchase_order(erp, args.po_number)
        if result["found"]:
            # Convert and compare here, in code: the model never retypes or compares the numbers.
            try:
                converted = convert_currency(
                    erp, invoice.total, invoice.currency, result["currency"]
                )
            except ToolError as exc:
                # The order still exists: a missing rate must not hide that. Say what couldn't be
                # checked instead of failing the whole lookup.
                result["amount_exceeded"] = None
                result["note"] = f"Could not compare amounts: {exc}"
            else:
                result["invoice_total_in_po_currency"] = converted["amount"]
                result["amount_exceeded"] = converted["amount"] > result["amount"]
        return result

    def handle_convert_currency(args: ConvertCurrencyInput) -> dict[str, Any]:
        return convert_currency(erp, args.amount, args.from_currency, args.to_currency)

    def handle_check_invoice_totals(args: CheckInvoiceTotalsInput) -> dict[str, Any]:
        result = check_invoice_totals(invoice)
        # One signal instead of two booleans joined by "or", which the model misreads.
        result["totals_ok"] = result["subtotal_ok"] and result["total_ok"]
        return result

    def handle_record_finding(args: RecordFindingInput) -> dict[str, Any]:
        if ledger.record(invoice.invoice_number, args.code, args.detail):
            return {"recorded": True}
        else:
            return {"recorded": False, "reason": "already recorded"}

    lookup_vendor_tool = Tool(
        name="lookup_vendor",
        description=(
            "Look up a vendor in the ERP by the name printed on the invoice. "
            "Use it to check the vendor exists and whether it is active. When found, "
            "tax_id_matches says whether the ERP tax id equals the one printed on the invoice."
        ),
        input_model=LookupVendorInput,
        handler=handle_lookup_vendor,
    )

    get_purchase_order_tool = Tool(
        name="get_purchase_order",
        description=(
            "Find a purchase order by the exact number in the ERP. "
            "Use it to check the purchase order exists. When found, amount_exceeded says whether "
            "the invoice total, converted to the order's currency, is greater than the order's "
            "amount. amount_exceeded is null when the amounts could not be compared; the note "
            "says why."
        ),
        input_model=GetPurchaseOrderInput,
        handler=handle_get_purchase_order,
    )

    convert_currency_tool = Tool(
        name="convert_currency",
        description=(
            "Converts a certain amount from a currency to another. "
            "Only converts if the currency rate exists in the ERP."
        ),
        input_model=ConvertCurrencyInput,
        handler=handle_convert_currency,
    )

    check_invoice_totals_tool = Tool(
        name="check_invoice_totals",
        description=(
            "Check the arithmetic in the invoice. "
            "Sums every line and check the reported subtotal and total amounts. "
            "totals_ok is true when everything adds up."
        ),
        input_model=CheckInvoiceTotalsInput,
        handler=handle_check_invoice_totals,
    )

    record_finding_tool = Tool(
        name="record_finding",
        description=(
            "Records a problem finding in a Ledger. Call it ONLY when a tool result "
            "shows something wrong. Never call it for a check that passed: if the "
            "detail you would write says something matches, exists or is active, do "
            "not call this tool."
        ),
        input_model=RecordFindingInput,
        handler=handle_record_finding,
    )

    return ToolRegistry(
        [
            lookup_vendor_tool,
            get_purchase_order_tool,
            convert_currency_tool,
            check_invoice_totals_tool,
            record_finding_tool,
        ]
    )
