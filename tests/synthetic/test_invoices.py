from docaudit.synthetic import LAYOUTS, generate_invoices


def test_is_deterministic_per_seed() -> None:
    assert generate_invoices(5, seed=1) == generate_invoices(5, seed=1)
    assert generate_invoices(5, seed=1) != generate_invoices(5, seed=2)


def test_cycles_through_every_layout() -> None:
    assert {s.layout for s in generate_invoices(len(LAYOUTS))} == set(LAYOUTS)


def test_text_contains_the_identifying_facts() -> None:
    for s in generate_invoices(30):
        assert s.truth.invoice_number in s.text
        assert s.truth.vendor_name in s.text
        if s.truth.po_number:
            assert s.truth.po_number in s.text


def test_totals_add_up_unless_a_discrepancy_was_seeded() -> None:
    samples = generate_invoices(60)
    for s in samples:
        adds_up = s.truth.subtotal + s.truth.tax == s.truth.total
        assert adds_up == ("total_mismatch" not in s.discrepancies), s.id
    assert any(s.discrepancies for s in samples), "the dataset should include some wrong totals"
