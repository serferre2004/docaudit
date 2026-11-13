from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from docaudit.extraction.schemas import Invoice, LineItem
from docaudit.llm import LLMResponse, ToolCall
from docaudit.tools import ErpData, PurchaseOrder, Tool, ToolRegistry, Vendor


class EchoInput(BaseModel):
    text: str


def echo_tool(calls: list[str] | None = None) -> Tool:
    def handler(args: EchoInput) -> dict[str, Any]:
        if calls is not None:
            calls.append(args.text)
        return {"echo": args.text}

    return Tool("echo", "Repeats the text.", EchoInput, handler)


def echo_registry(calls: list[str] | None = None) -> ToolRegistry:
    return ToolRegistry([echo_tool(calls)])


def call(tool: str = "echo", /, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=tool, arguments=arguments)


def asks(*calls: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(text=text, model="fake", stop_reason="tool_use", tool_calls=calls)


def make_invoice(**changes: Any) -> Invoice:
    invoice = Invoice(
        invoice_number="INV-1",
        vendor_name="Acme Ltd.",
        vendor_tax_id="11-1",
        po_number="PO-1",
        issue_date=date(2026, 1, 15),
        currency="USD",
        line_items=[
            LineItem(
                description="Thing",
                quantity=Decimal(2),
                unit_price=Decimal(50),
                amount=Decimal(100),
            )
        ],
        subtotal=Decimal(100),
        tax=Decimal(8),
        total=Decimal(108),
    )
    return invoice.model_copy(update=changes)


def make_erp() -> ErpData:
    return ErpData(
        vendors=(Vendor("Acme Ltd.", "11-1", "active"), Vendor("Gone Inc.", "22-2", "suspended")),
        purchase_orders=(PurchaseOrder("PO-1", "Acme Ltd.", "USD", Decimal(500)),),
        usd_per_unit={"USD": Decimal(1), "EUR": Decimal("1.08"), "COP": Decimal("0.00025")},
    )
