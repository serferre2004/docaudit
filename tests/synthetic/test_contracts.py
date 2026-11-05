import re

from docaudit.synthetic import POLICY_ID, generate_corpus


def test_is_deterministic_per_seed() -> None:
    assert generate_corpus(seed=1) == generate_corpus(seed=1)
    assert generate_corpus(seed=1) != generate_corpus(seed=2)


def test_has_one_contract_per_vendor_and_the_policy() -> None:
    ids = [document.id for document in generate_corpus().documents]
    assert POLICY_ID in ids
    assert len(ids) == len(set(ids)) == 6


def test_every_question_points_at_a_clause_that_exists() -> None:
    corpus = generate_corpus()
    texts = {document.id: document.text for document in corpus.documents}
    assert {question.kind for question in corpus.questions} == {"paraphrase", "exact"}
    for question in corpus.questions:
        pattern = rf"^{re.escape(question.section)}\s"
        assert re.search(pattern, texts[question.doc_id], re.MULTILINE), question.id


def test_question_ids_are_unique() -> None:
    questions = generate_corpus().questions
    assert len({question.id for question in questions}) == len(questions)


def test_contains_a_clause_too_long_for_one_chunk() -> None:
    text = generate_corpus().documents[0].text
    clause = text.split("\n8.1 ")[1].split("\n8.2 ")[0]
    assert len(clause) > 800
