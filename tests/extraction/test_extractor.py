import json

import pytest

from docaudit.extraction import ExtractionError, Invoice, InvoiceExtractor
from docaudit.llm import FakeProvider, LLMError, Usage

DOC = "INVOICE\nInvoice No: ACME-1\n..."


async def test_valid_output_on_the_first_attempt(invoice: Invoice, invoice_json: str) -> None:
    fake = FakeProvider(replies=[invoice_json])
    result = await InvoiceExtractor(fake).extract(DOC)

    assert result.invoice == invoice
    assert result.attempts == 1
    call = fake.calls[0]
    assert call.temperature == 0.0, "extraction should be as deterministic as possible"
    assert call.system
    assert DOC in call.messages[0].content


async def test_repairs_unparseable_output(invoice: Invoice, invoice_json: str) -> None:
    fake = FakeProvider(replies=["Sorry, I can't read that.", invoice_json])
    result = await InvoiceExtractor(fake).extract(DOC)

    assert (result.invoice, result.attempts) == (invoice, 2)
    retry = fake.calls[1].messages
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert retry[1].content == "Sorry, I can't read that.", "the model must see what it wrote"


async def test_feeds_validation_errors_back_to_the_model(invoice_json: str) -> None:
    bad = json.dumps({**json.loads(invoice_json), "issue_date": "14/03/2026"})
    fake = FakeProvider(replies=[bad, invoice_json])
    result = await InvoiceExtractor(fake).extract(DOC)

    assert result.attempts == 2
    repair = fake.calls[1].messages[-1]
    assert repair.role == "user"
    assert "issue_date" in repair.content


async def test_gives_up_after_max_attempts() -> None:
    fake = FakeProvider(replies=["nope"] * 5)
    with pytest.raises(ExtractionError) as excinfo:
        await InvoiceExtractor(fake, max_attempts=3).extract(DOC)
    assert len(fake.calls) == 3
    assert "3" in str(excinfo.value)


async def test_provider_errors_propagate_and_are_not_retried() -> None:
    fake = FakeProvider(replies=[])  # raises LLMError on the first call
    with pytest.raises(LLMError):
        await InvoiceExtractor(fake).extract(DOC)
    assert len(fake.calls) == 1


async def test_usage_is_summed_over_every_attempt(invoice_json: str) -> None:
    fake = FakeProvider(
        replies=["nope", invoice_json], usage_per_call=Usage(input_tokens=10, output_tokens=5)
    )
    result = await InvoiceExtractor(fake).extract(DOC)
    assert result.usage == Usage(input_tokens=20, output_tokens=10)


async def test_passes_max_tokens_and_leaves_the_provider_open(invoice_json: str) -> None:
    fake = FakeProvider(replies=[invoice_json])
    await InvoiceExtractor(fake, max_tokens=999).extract(DOC)
    assert fake.calls[0].max_tokens == 999
    assert not fake.closed, "the extractor doesn't own the provider"
