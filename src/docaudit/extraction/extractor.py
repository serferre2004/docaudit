"""The extractor and its repair loop.   >>> YOUR TURN: implement this file. <<<

Tests that define "done": tests/extraction/test_extractor.py
"""

from docaudit.extraction.schemas import ExtractionResult
from docaudit.llm.base import LLMProvider


class InvoiceExtractor:
    """Turns invoice text into a validated ``Invoice``, repairing the model's mistakes.

    Requirements:
      * ``__init__(self, provider, *, max_attempts=3, max_tokens=2048)`` stores its arguments.
        The extractor does NOT own the provider: whoever created the provider closes it.
      * ``extract(document)`` runs this loop, at most ``max_attempts`` times:
          1. Call ``provider.complete`` with the prompt from ``build_extraction_prompt``,
             ``temperature=0.0`` and ``max_tokens``.
          2. ``parse_json_object`` the reply, then ``Invoice.model_validate`` the result.
          3. Valid -> return an ``ExtractionResult`` with the invoice, the attempt number and the
             ``Usage`` summed over ALL attempts (repairs cost tokens too).
          4. Invalid (``ExtractionError`` from parsing or ``ValidationError`` from Pydantic) ->
             append the model's reply as an ``assistant`` message, then the repair message
             (``build_repair_message`` with the error text; use ``format_validation_error`` for
             ``ValidationError``), and try again.
      * When attempts run out, raise ``ExtractionError`` saying how many attempts were made
        and what the last error was (chain it with ``from``).
      * Do NOT catch ``LLMError``: a provider failure is not the model's mistake, retrying it
        here would hide outages (and the Anthropic SDK already retries transient errors).
    """

    def __init__(
        self, provider: LLMProvider, *, max_attempts: int = 3, max_tokens: int = 2048
    ) -> None:
        raise NotImplementedError

    async def extract(self, document: str) -> ExtractionResult:
        raise NotImplementedError
