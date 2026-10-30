"""The target shape of an extraction.

These models do double duty:

1. **Validation.** ``Invoice.model_validate(data)`` is the gate every LLM output must pass.
2. **Prompting.** ``Invoice.model_json_schema()`` is sent to the model, so every ``description``
   below is an instruction the model reads. Write them for the model, not just for humans.

The schema checks *shape*, not *business rules*. A total that doesn't add up is still a valid
``Invoice``: we record what the document SAYS. Catching the wrong total is the auditor's job
(Phase 4), and it can only do that if extraction didn't "fix" the number.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from docaudit.llm.types import Usage


class LineItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1, description="Item or service, as written")
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0, description="Price per unit, as a plain number")
    amount: Decimal = Field(description="Line total exactly as printed, even if it looks wrong")


class Invoice(BaseModel):
    model_config = ConfigDict(frozen=True)

    invoice_number: str = Field(
        min_length=1, description="Exactly as printed, e.g. 'INV-2026-0042'"
    )
    vendor_name: str = Field(min_length=1, description="The company issuing the invoice")
    vendor_tax_id: str | None = Field(None, description="Vendor tax/VAT id, or null if absent")
    po_number: str | None = Field(None, description="Purchase order referenced, or null if absent")
    issue_date: date = Field(description="Date the invoice was issued, as YYYY-MM-DD")
    due_date: date | None = Field(None, description="Payment due date as YYYY-MM-DD, or null")
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. 'USD'")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal
    tax: Decimal = Field(Decimal("0"), description="Total tax amount; 0 if none is charged")
    total: Decimal = Field(description="Amount due exactly as printed, even if it looks wrong")


class ExtractionResult(BaseModel):
    """A successful extraction plus what it cost to get it."""

    model_config = ConfigDict(frozen=True)

    invoice: Invoice
    attempts: int = Field(ge=1, description="1 = valid on the first try; >1 = repairs were needed")
    usage: Usage = Usage()


class ExtractionError(Exception):
    """The model never produced a valid ``Invoice`` (unparseable or invalid output).

    Distinct from ``LLMError``: that one means the *provider* failed (network, auth, ...);
    this one means the provider worked but the *output* was unusable.
    """
