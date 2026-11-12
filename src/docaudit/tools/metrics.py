"""Scoring the auditor against known discrepancies."""

from collections.abc import Sequence
from dataclasses import dataclass

FindingCodes = frozenset[str]


@dataclass(frozen=True)
class AuditScores:
    precision: float
    recall: float
    exact_cases: int
    false_alarm_cases: int
    cases: int


def score_findings(pairs: Sequence[tuple[FindingCodes, FindingCodes]]) -> AuditScores:
    """Compare ``(expected, found)`` codes per invoice.

    Precision: of the findings recorded, how many were real. Recall: of the real problems, how many
    were recorded. An auditor can game either one alone: flag everything for perfect recall, or
    flag nothing and never be wrong. ``false_alarm_cases`` counts clean invoices flagged anyway.
    """
    if not pairs:
        raise ValueError("pairs is empty")
    correct = sum(len(expected & found) for expected, found in pairs)
    total_found = sum(len(found) for _, found in pairs)
    total_expected = sum(len(expected) for expected, _ in pairs)
    return AuditScores(
        precision=correct / total_found if total_found else 1.0,
        recall=correct / total_expected if total_expected else 1.0,
        exact_cases=sum(expected == found for expected, found in pairs),
        false_alarm_cases=sum(not expected and bool(found) for expected, found in pairs),
        cases=len(pairs),
    )
