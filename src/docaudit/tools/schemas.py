"""The shared vocabulary of the tools package. Nothing here talks to a model.

``Vendor``, ``PurchaseOrder`` and ``ErpData`` stand in for the company systems a real auditor would
query. They are facts the model can't know, which is the whole reason tools exist.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from docaudit.llm.types import Message, ToolCall, ToolResult, Usage

FindingCode = Literal[
    "UNKNOWN_VENDOR",
    "SUSPENDED_VENDOR",
    "TAX_ID_MISMATCH",
    "MISSING_PO",
    "PO_NOT_FOUND",
    "PO_AMOUNT_EXCEEDED",
    "TOTAL_MISMATCH",
]


@dataclass(frozen=True)
class Vendor:
    name: str
    tax_id: str
    status: Literal["active", "suspended"]


@dataclass(frozen=True)
class PurchaseOrder:
    po_number: str
    vendor_name: str
    currency: str
    amount: Decimal


@dataclass(frozen=True)
class ErpData:
    """A read-only snapshot of the company's systems."""

    vendors: tuple[Vendor, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    usd_per_unit: Mapping[str, Decimal] = field(default_factory=dict)


class Finding(BaseModel):
    """One problem the auditor recorded."""

    model_config = ConfigDict(frozen=True)

    invoice_number: str
    code: FindingCode
    detail: str


class ToolError(Exception):
    """A tool could not do what was asked, and the model can do something about it."""


@dataclass(frozen=True)
class Tool:
    """A tool as our code sees it: what the model is told, plus the function that does the work.

    ``handler`` receives a validated ``input_model`` and returns a JSON-friendly dict.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]


class ToolExecution(BaseModel):
    """One call the model asked for and what came back: the audit trail of a run."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    result: ToolResult


class ToolLoopResult(BaseModel):
    """How a conversation with tools ended, and everything it took to get there."""

    model_config = ConfigDict(frozen=True)

    text: str
    messages: list[Message]
    executions: list[ToolExecution]
    turns: int
    usage: Usage = Usage()


class ToolLoopError(Exception):
    """The loop stopped without a final answer: too many turns, or a truncated reply.

    As with ``ExtractionError``, the provider worked. Provider failures stay ``LLMError``.
    """


class AuditResult(BaseModel):
    """The findings on one invoice. They come from the ledger, never from the model's prose."""

    model_config = ConfigDict(frozen=True)

    invoice_number: str
    findings: list[Finding]
    loop: ToolLoopResult
