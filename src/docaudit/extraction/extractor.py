"""The extractor and its repair loop.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/extraction/test_extractor.py
"""

from pydantic import ValidationError

from docaudit.extraction.parsing import parse_json_object
from docaudit.extraction.prompts import (
    build_extraction_prompt,
    build_repair_message,
    format_validation_error,
)
from docaudit.extraction.schemas import ExtractionError, ExtractionResult, Invoice
from docaudit.llm.base import LLMProvider
from docaudit.llm.types import Message, Usage


class InvoiceExtractor:
    """Turns invoice text into a validated ``Invoice``, repairing the model's mistakes.

    Requirements:
      * ``__init__(self, provider, *, max_attempts=3, max_tokens=2048)`` stores its arguments.
        The extractor does NOT own the provider: whoever created the provider closes it.
      * ``extract(document)`` runs this loop, at most ``max_attempts`` times:
          1. Call ``provider.complete`` with the prompt from ``build_extraction_prompt``,
             ``temperature=0.0``, ``max_tokens`` and the ``Invoice`` JSON schema as
             ``response_schema``.
          2. ``parse_json_object`` the reply, then ``Invoice.model_validate`` the result.
          3. Valid -> return an ``ExtractionResult`` with the invoice, the attempt number and the
             ``Usage`` summed over ALL attempts (repairs cost tokens too).
          4. Invalid (``ExtractionError`` from parsing or ``ValidationError`` from Pydantic) ->
             append the model's reply as an ``assistant`` message, then the repair message
             (``build_repair_message`` with the error text; use ``format_validation_error`` for
             ``ValidationError``), and try again.
      * ``max_attempts`` below 1 raises ``ValueError``: no attempt could ever succeed.
      * When attempts run out, raise ``ExtractionError`` saying how many attempts were made
        and what the last error was (chain it with ``from``).
      * Do NOT catch ``LLMError``: a provider failure is not the model's mistake, retrying it
        here would hide outages (and the Anthropic SDK already retries transient errors).
    """

    def __init__(
        self, provider: LLMProvider, *, max_attempts: int = 3, max_tokens: int = 2048
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.provider = provider
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens

    async def extract(self, document: str) -> ExtractionResult:
        attempts = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_exception: Exception | None = None
        system, msgs = build_extraction_prompt(document)
        while attempts < self.max_attempts:
            attempts += 1
            response = await self.provider.complete(
                msgs,
                system=system,
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_schema=Invoice.model_json_schema(),
            )
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            try:
                json_obj = parse_json_object(response.text)
                invoice = Invoice.model_validate(json_obj)
                return ExtractionResult(
                    invoice=invoice,
                    attempts=attempts,
                    usage=Usage(input_tokens=total_input_tokens, output_tokens=total_output_tokens),
                )
            except (ExtractionError, ValidationError) as e:
                last_exception = e
                error_msg = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
                repair_message = build_repair_message(error_msg)
                msgs.append(Message(role="assistant", content=response.text))
                msgs.append(repair_message)

        raise ExtractionError(
            f"Failed to extract invoice after {self.max_attempts} attempts: {error_msg}"
        ) from last_exception
