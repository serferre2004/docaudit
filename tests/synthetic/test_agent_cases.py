from collections import Counter

from docaudit.synthetic import generate_agent_cases
from docaudit.synthetic.agent_cases import INJECTION


def test_every_action_is_represented() -> None:
    _, cases = generate_agent_cases()
    assert len(cases) == 22
    assert Counter(c.expected for c in cases) == {"flag": 15, "approve": 4, "escalate": 3}


def test_documents_are_text_with_their_invoice_number() -> None:
    _, cases = generate_agent_cases()
    for case in cases:
        assert case.truth.invoice_number in case.document


def test_discrepancies_are_flagged_and_clean_ones_are_not() -> None:
    _, cases = generate_agent_cases()
    for case in cases:
        assert (case.expected == "flag") == bool(case.expected_findings)


def test_the_extra_cases() -> None:
    erp, cases = generate_agent_cases()
    by_id = {c.id: c for c in cases}
    assert by_id["case-20"].truth.currency not in erp.usd_per_unit
    assert by_id["case-20"].expected == "escalate"
    assert INJECTION in by_id["case-21"].document
    assert by_id["case-21"].expected_findings == {"TOTAL_MISMATCH"}


def test_generation_is_deterministic() -> None:
    assert generate_agent_cases(seed=4)[1] == generate_agent_cases(seed=4)[1]
