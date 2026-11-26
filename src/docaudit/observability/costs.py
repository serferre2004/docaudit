"""What each model call costs.   >>> YOUR TURN: implement ``cost_usd``. <<<

Tests that define "done": tests/observability/test_costs.py
"""

from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from docaudit.llm.types import Usage


class Price(BaseModel):
    """USD per million tokens. Input and output are priced separately: output is usually dearer."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: Decimal
    output_per_mtok: Decimal


FREE = Price(input_per_mtok=Decimal(0), output_per_mtok=Decimal(0))

# Keyed by the model id the provider reports in ``LLMResponse.model``.
# Local models cost electricity, not tokens. Add your hosted models with the CURRENT prices from
# the provider's pricing page: a price you guessed is worse than no price, because nobody
# questions a number on a dashboard.
PRICES: dict[str, Price] = {
    "llama3.1:8b": FREE,
    "qwen2.5:7b": FREE,
}


def cost_usd(model: str, usage: Usage, prices: Mapping[str, Price] = PRICES) -> Decimal | None:
    """The cost of one call, or ``None`` if ``model`` has no price.

    Requirements:
      * ``input_tokens`` at the input price plus ``output_tokens`` at the output price, both per
        MILLION tokens. Use ``Decimal`` all the way: money in floats drifts.
      * Unknown model -> ``None``, not ``0``: "we don't know" and "free" are different answers.
      * Don't round here. Rounding each call and then adding loses money; round where you show it.
    """
    if model not in prices:
        return None

    price = prices[model]
    input_mt = usage.input_tokens / Decimal("1000000")
    output_mt = usage.output_tokens / Decimal("1000000")

    return input_mt * price.input_per_mtok + output_mt * price.output_per_mtok
