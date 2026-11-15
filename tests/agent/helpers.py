import json
from typing import Any

from docaudit.llm import LLMResponse, ToolCall, ToolResult
from docaudit.tools import ToolExecution

from ..tools.helpers import asks, call, make_erp, make_invoice

__all__ = ["asks", "call", "clean_audit", "executed", "make_erp", "make_invoice"]


def executed(
    tool: str, result: dict[str, Any] | str, /, is_error: bool = False, **arguments: Any
) -> ToolExecution:
    """A tool call that already ran, as the audit loop records it."""
    content = result if isinstance(result, str) else json.dumps(result)
    return ToolExecution(
        call=ToolCall(id="c", name=tool, arguments=arguments),
        result=ToolResult(tool_call_id="c", name=tool, content=content, is_error=is_error),
    )


def clean_audit() -> list[str | LLMResponse]:
    """Scripted replies for a complete audit of ``make_invoice()`` that finds nothing."""
    return [
        asks(call("lookup_vendor", name="Acme Ltd.")),
        asks(call("get_purchase_order", po_number="PO-1")),
        asks(call("check_invoice_totals")),
        "Every check passed.",
    ]
