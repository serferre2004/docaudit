import pytest

from docaudit.rag import SourceDocument, chunk_document

TEXT = """AGREEMENT MSA-X
Between Acme and Demo.

## 4. Payment
4.1 Customer shall pay
within 30 days.

4.2 Overdue amounts accrue interest.

## 5. Service
5.1 Keep it running.
"""


def doc(text: str = TEXT) -> SourceDocument:
    return SourceDocument(id="MSA-X", title="Agreement with Acme", text=text)


def test_one_chunk_per_clause_with_its_location() -> None:
    chunks = chunk_document(doc())
    first = chunks[1]
    assert [c.id for c in chunks] == ["MSA-X#preamble", "MSA-X#4.1", "MSA-X#4.2", "MSA-X#5.1"]
    assert (first.doc_id, first.doc_title) == ("MSA-X", "Agreement with Acme")
    assert (first.section, first.heading) == ("4.1", "4. Payment")
    assert chunks[3].heading == "5. Service"


def test_inner_whitespace_is_normalized() -> None:
    assert chunk_document(doc("1.1 a    b\nc"))[0].text == "a b c"


def test_the_last_clause_is_split_too() -> None:
    chunks = chunk_document(doc(f"8.1 {LONG}"), max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_text_between_a_heading_and_its_first_clause_is_kept() -> None:
    text = "## 4. Pay\nSome intro.\n4.1 Pay.\n## 5. Run\nMore\nintro.\n5.1 Run."
    chunks = chunk_document(doc(text))
    assert [c.id for c in chunks] == ["MSA-X#4-intro", "MSA-X#4.1", "MSA-X#5-intro", "MSA-X#5.1"]
    assert (chunks[2].section, chunks[2].heading) == ("5-intro", "5. Run")
    assert chunks[2].text == "More intro."


def test_continuation_lines_belong_to_the_clause_and_whitespace_is_normalized() -> None:
    clause = chunk_document(doc())[1]
    assert clause.text == "Customer shall pay within 30 days."


def test_preamble_is_kept() -> None:
    preamble = chunk_document(doc())[0]
    assert preamble.section == "preamble"
    assert preamble.heading == ""
    assert preamble.text == "AGREEMENT MSA-X Between Acme and Demo."


def test_document_without_preamble_and_blank_document() -> None:
    assert [c.section for c in chunk_document(doc("## 1. A\n1.1 Hello."))] == ["1.1"]
    assert chunk_document(doc("  \n\n ")) == []


def test_a_section_title_is_not_mistaken_for_a_clause() -> None:
    chunks = chunk_document(doc("## 2.5 Odd heading\n2.6 A clause.\nPrice is\n3 USD."))
    assert [c.section for c in chunks] == ["2.6"]
    assert chunks[0].heading == "2.5 Odd heading"
    assert chunks[0].text == "A clause. Price is 3 USD."


LONG = " ".join(f"word{i:03d}" for i in range(100))  # 799 chars, 8 per word with the space


def long_doc() -> SourceDocument:
    return doc(f"## 8. Long\n8.1 {LONG}\n8.2 Short.")


def test_long_clause_is_split_into_numbered_parts_of_the_same_section() -> None:
    chunks = chunk_document(long_doc(), max_chars=100, overlap=20)
    parts = [c for c in chunks if c.section == "8.1"]
    assert len(parts) > 1
    assert [c.id for c in parts] == [f"MSA-X#8.1-p{i + 1}" for i in range(len(parts))]
    assert chunks[-1].id == "MSA-X#8.2", "short clauses keep the plain id"
    assert all(len(c.text) <= 100 for c in parts)


def test_overlap_is_measured_in_characters_with_words_of_any_length() -> None:
    chunks = chunk_document(doc(f"8.1 {'x' * 30} abcdefg mm nn"), max_chars=40, overlap=20)
    assert [c.text for c in chunks] == [f"{'x' * 30} abcdefg", "abcdefg mm nn"]


def test_parts_cover_every_word_in_order_and_never_cut_one() -> None:
    chunks = chunk_document(long_doc(), max_chars=100, overlap=20)
    parts = [c.text.split() for c in chunks if c.section == "8.1"]
    expected = LONG.split()
    assert all(word in expected for part in parts for word in part), "a word was cut"
    assert parts[0][0] == expected[0]
    assert parts[-1][-1] == expected[-1]
    covered = {word for part in parts for word in part}
    assert covered == set(expected)


def test_parts_repeat_the_previous_last_words() -> None:
    chunks = chunk_document(long_doc(), max_chars=100, overlap=20)
    parts = [c.text.split() for c in chunks if c.section == "8.1"]
    # "wordNNN" is 7 chars: two words plus a space fit in an overlap of 20, three don't.
    for previous, current in zip(parts, parts[1:], strict=False):
        assert current[:2] == previous[-2:]
        assert current[2] not in previous


def test_no_overlap() -> None:
    chunks = chunk_document(long_doc(), max_chars=100, overlap=0)
    parts = [c.text.split() for c in chunks if c.section == "8.1"]
    assert sum(len(p) for p in parts) == 100


def test_a_huge_word_gets_its_own_part_and_splitting_always_moves_forward() -> None:
    huge = "x" * 50
    chunks = chunk_document(doc(f"8.1 aa {huge} bb"), max_chars=10, overlap=9)
    joined = [word for c in chunks for word in c.text.split()]
    assert huge in joined
    assert {"aa", "bb"} <= set(joined)


def test_every_part_adds_new_words() -> None:
    chunks = chunk_document(doc("8.1 aaaa bbbb cccccccc"), max_chars=10, overlap=9)
    assert [c.text for c in chunks] == ["aaaa bbbb", "cccccccc"]


@pytest.mark.parametrize(("max_chars", "overlap"), [(100, 100), (100, 150), (100, -1)])
def test_rejects_invalid_overlap(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(doc(), max_chars=max_chars, overlap=overlap)


def test_overlap_never_leaves_a_part_without_new_words() -> None:
    chunks = chunk_document(doc("8.1 aaa b c dddddddd"), max_chars=10, overlap=5)
    assert [c.text for c in chunks] == ["aaa b c", "c dddddddd"]
