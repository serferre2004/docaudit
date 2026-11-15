"""The decision policy: plain code, no model.   >>> YOUR TURN <<<

Tests that define "done": tests/agent/test_policy.py

The model gathered the evidence in Lesson 4. Deciding what to do with it is a business rule, so it
lives here, where it is deterministic, testable and the same for every invoice.
"""

import json
from collections.abc import Sequence
from decimal import Decimal

from docaudit.agent.schemas import Decision, Policy
from docaudit.extraction.schemas import Invoice
from docaudit.tools.schemas import ErpData, Finding, ToolExecution


def _successful(tool_name: str, executions: Sequence[ToolExecution]) -> Sequence[ToolExecution]:
    succ = []
    for e in executions:
        if e.call.name == tool_name and not e.result.is_error:
            succ.append(e)
    return succ


def _compare_strings(str_a: str, str_b: str) -> bool:
    return str_a.lower().strip() == str_b.lower().strip()


def audit_gaps(invoice: Invoice, executions: Sequence[ToolExecution]) -> list[str]:
    """List the checks the audit should have made but that can't be confirmed.

    Lesson 4 showed that the model sometimes skips a step, or checks the wrong thing. A finding it
    didn't record is invisible; a check it didn't make is detectable, because the executions say
    exactly what ran. Each gap is one sentence a reviewer can read.

    Requirements, in this order:
      * The vendor: at least one SUCCESSFUL (``not is_error``) ``lookup_vendor`` call whose
        ``name`` equals the invoice's ``vendor_name``, ignoring case and surrounding spaces.
        Otherwise one gap naming the vendor.
      * The purchase order, only if the invoice has a ``po_number``: at least one successful
        ``get_purchase_order`` call with exactly that ``po_number``. Otherwise one gap naming it.
        If such a call returned ``"amount_exceeded": null``, one gap that includes the result's
        ``note``: the order exists, but its amount could not be compared.
      * The totals: at least one successful ``check_invoice_totals`` call. Otherwise one gap.

    An error the model later recovered from is not a gap: what matters is that the check
    eventually succeeded.
    """
    gaps = []

    vendor_ex = _successful("lookup_vendor", executions)
    if not any(_compare_strings(e.call.arguments["name"], invoice.vendor_name) for e in vendor_ex):
        gaps.append(f"Vendor with name {invoice.vendor_name} was never retrieved")

    if invoice.po_number is not None:
        get_po_ex = _successful("get_purchase_order", executions)
        match = next(
            (e for e in get_po_ex if e.call.arguments["po_number"] == invoice.po_number), None
        )
        if match is None:
            gaps.append(f"Purchase order with number {invoice.po_number} was never retrieved")
        else:
            content_obj = json.loads(match.result.content)
            if content_obj["found"] and content_obj["amount_exceeded"] is None:
                gaps.append(
                    f"Amount of PO number {invoice.po_number} couldn't be compared. "
                    f"NOTE: {content_obj['note']}"
                )
    totals_ex = _successful("check_invoice_totals", executions)
    if not totals_ex:
        gaps.append("Invoice totals were never checked")
    return gaps


def total_in_usd(invoice: Invoice, erp: ErpData) -> Decimal | None:
    """The invoice total in USD, rounded to cents, or ``None`` if its currency has no rate."""
    rate = erp.usd_per_unit
    if invoice.currency not in rate:
        return None
    return round(invoice.total * rate[invoice.currency], 2)


def decide(
    findings: Sequence[Finding],
    gaps: Sequence[str],
    total_usd: Decimal | None,
    policy: Policy,
) -> Decision:
    """Turn the evidence into ``approve``, ``flag`` or ``escalate``. First matching rule wins.

    Requirements:
      1. Any finding -> ``flag``. Reasons: one per finding, ``"CODE: detail"``, followed by the
         gaps (a reviewer should know what wasn't checked, too). A known problem doesn't need a
         human to decide: the invoice goes back to the vendor.
      2. Any gap -> ``escalate``, with the gaps as reasons. "Couldn't check" is not "checked and
         fine": approving it would be guessing.
      3. ``total_usd`` is ``None`` -> ``escalate``: the amount can't be compared with the limit.
      4. ``total_usd`` above ``policy.auto_approve_limit_usd`` -> ``escalate``, with a reason
         that includes both amounts. Clean, but too much money to approve without a signature.
         Exactly at the limit is still approvable.
      5. Otherwise ``approve``, with one reason saying every check passed.
    """
    if findings:
        reasons = tuple([f"{f.code}: {f.detail}" for f in findings] + list(gaps))
        return Decision(action="flag", reasons=reasons)
    if gaps:
        return Decision(action="escalate", reasons=tuple(gaps))
    if total_usd is None:
        return Decision(
            action="escalate", reasons=("The amount can't be compared with the limit.",)
        )
    if total_usd > policy.auto_approve_limit_usd:
        return Decision(
            action="escalate",
            reasons=(
                (
                    f"The total {total_usd} USD is above "
                    f"the auto-approve limit of {policy.auto_approve_limit_usd} USD"
                ),
            ),
        )

    return Decision(action="approve", reasons=("Every check passed",))
