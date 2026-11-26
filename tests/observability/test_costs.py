from decimal import Decimal

from docaudit.llm import Usage
from docaudit.observability import PRICES, Price, cost_usd

PRICES_FOR_TESTS = {
    "big": Price(input_per_mtok=Decimal("3"), output_per_mtok=Decimal("15")),
    "local": Price(input_per_mtok=Decimal(0), output_per_mtok=Decimal(0)),
}


def test_input_and_output_are_priced_separately_per_million_tokens() -> None:
    usage = Usage(input_tokens=2_000, output_tokens=500)
    # 2000 * 3 / 1M + 500 * 15 / 1M = 0.006 + 0.0075
    assert cost_usd("big", usage, PRICES_FOR_TESTS) == Decimal("0.0135")


def test_the_cost_is_a_decimal_and_is_not_rounded() -> None:
    cost = cost_usd("big", Usage(input_tokens=1, output_tokens=1), PRICES_FOR_TESTS)
    assert isinstance(cost, Decimal)
    assert cost == Decimal("0.000018")


def test_a_free_model_costs_zero() -> None:
    usage = Usage(input_tokens=10_000, output_tokens=10_000)
    assert cost_usd("local", usage, PRICES_FOR_TESTS) == 0


def test_an_unknown_model_has_no_cost_rather_than_zero() -> None:
    assert cost_usd("mystery", Usage(input_tokens=10), PRICES_FOR_TESTS) is None


def test_the_default_table_prices_the_local_models() -> None:
    assert cost_usd("qwen2.5:7b", Usage(input_tokens=500), PRICES) == 0
