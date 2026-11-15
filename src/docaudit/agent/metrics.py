"""Scoring the workflow's decisions (provided).

Not every mistake costs the same:

* **Unsafe approval:** a document that should have been flagged or escalated was approved. The
  company pays an invoice it shouldn't. This is the number that must stay at zero.
* **Needless escalation:** a document that could have been approved went to a human. It costs a
  few minutes of someone's time.
* **Missed escalation:** a document that needed a human was flagged instead. Safe (nothing gets
  paid), but someone will chase a problem that may not exist.

Accuracy alone hides this: a workflow that escalates everything never approves anything unsafe,
and is useless.
"""

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from docaudit.agent.schemas import Action


class DecisionScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: int
    accuracy: float
    unsafe_approvals: int
    needless_escalations: int
    escalation_rate: float
    confusion: dict[str, int]
    """``"expected->actual"`` -> count, for every pair that occurred."""


def score_decisions(pairs: Sequence[tuple[Action, Action]]) -> DecisionScores:
    """Score ``(expected, actual)`` pairs. An empty sequence scores as zero cases."""
    if not pairs:
        return DecisionScores(
            cases=0,
            accuracy=0.0,
            unsafe_approvals=0,
            needless_escalations=0,
            escalation_rate=0.0,
            confusion={},
        )
    confusion = Counter(f"{expected}->{actual}" for expected, actual in pairs)
    return DecisionScores(
        cases=len(pairs),
        accuracy=sum(expected == actual for expected, actual in pairs) / len(pairs),
        unsafe_approvals=sum(
            actual == "approve" and expected != "approve" for expected, actual in pairs
        ),
        needless_escalations=sum(
            actual == "escalate" and expected == "approve" for expected, actual in pairs
        ),
        escalation_rate=sum(actual == "escalate" for _, actual in pairs) / len(pairs),
        confusion=dict(sorted(confusion.items())),
    )
