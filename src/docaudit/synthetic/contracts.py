"""Synthetic contracts and a procurement policy, with questions whose answers are known.

Each vendor from the invoice generator gets a master agreement. The contracts share their wording
on purpose and differ only in names and values, as real templated contracts do: a retriever has to
tell apart five nearly identical clauses 4.1.

Generation is deterministic: the same ``seed`` always gives the same corpus.
"""

import random
from dataclasses import dataclass

from docaudit.rag.schemas import EvalQuestion, SourceDocument
from docaudit.synthetic.invoices import VENDORS

POLICY_ID = "POL-PROC"

_PARAPHRASES = (
    ("4.1", "How many days do we have to settle the bills from {vendor}?"),
    ("4.2", "What penalty do we face if we are slow paying {vendor}?"),
    ("9.1", "How far in advance must we warn {vendor} to walk away without giving a reason?"),
    ("5.2", "What compensation do we get when {vendor} misses its uptime target?"),
    ("6.1", "What is the most money {vendor} could owe us if something goes wrong?"),
    ("8.2", "For how long must {vendor} keep our secrets after the deal ends?"),
    ("10.1", "Can we inspect the books of {vendor}, and how much warning must we give?"),
)

_POLICY_QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("1.2", "paraphrase", "Who has to sign off a purchase worth USD 50,000?"),
    ("2.1", "paraphrase", "Can we pay a supplier when no purchase order was ever raised?"),
    ("2.3", "paraphrase", "How far can a billed price drift from the order before it is flagged?"),
    ("4.3", "paraphrase", "Which exchange rate applies to bills in a foreign currency?"),
    ("4.2", "exact", "What is form AP-17 used for?"),
    ("4.4", "exact", "What is form AP-71 used for?"),
    ("4.5", "exact", "What is form AP-18 used for?"),
    ("5.2", "exact", "What is booked under ledger code GL-4410?"),
    ("5.4", "exact", "What is booked under ledger code GL-4401?"),
    ("5.5", "exact", "What is booked under ledger code GL-4140?"),
)


@dataclass(frozen=True)
class SyntheticCorpus:
    documents: tuple[SourceDocument, ...]
    questions: tuple[EvalQuestion, ...]


def generate_corpus(*, seed: int = 0) -> SyntheticCorpus:
    """Return one contract per vendor, the policy, and the questions about them."""
    rng = random.Random(seed)
    documents: list[SourceDocument] = []
    questions: list[EvalQuestion] = []

    for vendor_name, _, prefix in VENDORS:
        doc_id = f"MSA-{prefix}"
        documents.append(
            SourceDocument(
                id=doc_id,
                title=f"Master Services Agreement {doc_id}, {vendor_name}",
                text=_render_contract(rng, doc_id, vendor_name, prefix),
            )
        )
        questions.extend(
            EvalQuestion(
                id=f"{doc_id}-{section}",
                question=template.format(vendor=vendor_name),
                doc_id=doc_id,
                section=section,
                kind="paraphrase",
            )
            for section, template in _PARAPHRASES
        )
        questions.append(
            EvalQuestion(
                id=f"{doc_id}-rate-card",
                question=f"What does rate card RC-{prefix}-2026 govern?",
                doc_id=doc_id,
                section="3.1",
                kind="exact",
            )
        )

    documents.append(
        SourceDocument(id=POLICY_ID, title="Procurement Policy POL-PROC", text=_POLICY_TEXT)
    )
    questions.extend(
        EvalQuestion(
            id=f"{POLICY_ID}-{section}",
            question=question,
            doc_id=POLICY_ID,
            section=section,
            kind="exact" if kind == "exact" else "paraphrase",
        )
        for section, kind, question in _POLICY_QUESTIONS
    )
    return SyntheticCorpus(documents=tuple(documents), questions=tuple(questions))


def _render_contract(rng: random.Random, doc_id: str, vendor: str, prefix: str) -> str:
    net_days = rng.choice((15, 30, 45, 60))
    late_interest = rng.choice(("0.5", "1.0", "1.5", "2.0"))
    dispute_days = rng.choice((10, 15, 20))
    uplift_cap = rng.choice((2, 3, 4, 5))
    uptime = rng.choice(("99.0", "99.5", "99.9"))
    credit = rng.choice((2, 5, 10))
    credit_cap = rng.choice((15, 20, 30))
    liability_months = rng.choice((6, 12, 24))
    insurance = rng.choice(("1,000,000", "2,000,000", "5,000,000"))
    secrecy_years = rng.choice((2, 3, 5))
    exit_days = rng.choice((30, 60, 90))
    cure_days = rng.choice((15, 30))
    audit_days = rng.choice((5, 10, 20))
    retention_years = rng.choice((3, 5, 7))
    renewal_months = rng.choice((6, 12))
    renewal_notice = rng.choice((30, 60, 90))
    term_year = rng.choice((2027, 2028))

    return f"""MASTER SERVICES AGREEMENT {doc_id}
Between {vendor} ("Supplier") and DocAudit Demo Corp ("Customer").

## 1. Scope
1.1 Supplier shall provide the services described in each Statement of Work signed by the parties.
1.2 If documents conflict, this Agreement prevails over any Statement of Work or purchase order.

## 2. Term and Renewal
2.1 This Agreement starts on the signature date and ends on December 31, {term_year}.
2.2 It renews automatically for periods of {renewal_months} months unless either party objects in
writing at least {renewal_notice} days before the end of the current period.

## 3. Fees
3.1 Fees are fixed by rate card RC-{prefix}-2026, which governs all unit prices under this
Agreement. Prices not listed there require a written amendment.
3.2 Supplier may raise prices once per calendar year, by no more than {uplift_cap}%.
3.3 Travel and other expenses are reimbursed only if Customer approved them in advance.

## 4. Invoicing and Payment
4.1 Customer shall pay undisputed amounts within {net_days} days of receiving a correct invoice.
4.2 Overdue amounts accrue interest at {late_interest}% per month until paid in full.
4.3 Every invoice must reference a valid purchase order, or Customer may reject it.
4.4 Customer must notify Supplier of a billing dispute within {dispute_days} days of receipt.

## 5. Service Levels
5.1 Supplier shall keep the services available at least {uptime}% of each calendar month.
5.2 For each full percentage point below that level, Customer receives a service credit of
{credit}% of the monthly fee, limited to {credit_cap}% of that fee per month.

## 6. Liability
6.1 Each party's aggregate liability under this Agreement is limited to the fees paid in the
{liability_months} months before the claim.
6.2 That limit does not apply to fraud, gross negligence or breach of confidentiality.

## 7. Insurance
7.1 Supplier shall maintain professional liability insurance of at least USD {insurance}.

## 8. Confidentiality
8.1 "Confidential Information" means all non-public information that one party discloses to the
other in connection with this Agreement, in any form, including prices, rate cards, business plans,
customer and employee data, technical documentation, software, security reports and the terms of
this Agreement. The receiving party shall use it only to perform this Agreement, shall protect it
with at least the care it applies to its own confidential information, and shall disclose it only
to employees and professional advisers who need it and who are bound by equivalent duties. These
duties do not cover information that is or becomes public through no fault of the receiving party,
that the receiving party already lawfully held, that it receives from a third party free to share
it, or that it develops independently. A party required by law to disclose Confidential Information
shall, where legally allowed, give prompt notice so that the other party may seek protection.
8.2 The duties in this section continue for {secrecy_years} years after this Agreement ends.

## 9. Termination
9.1 Customer may terminate for convenience by giving {exit_days} days' written notice.
9.2 Either party may terminate for material breach not cured within {cure_days} days of notice.

## 10. Audit
10.1 Customer may audit Supplier's records relating to this Agreement on {audit_days} business
days' notice.
10.2 Supplier shall retain those records for {retention_years} years.

## 11. Governing Law
11.1 This Agreement is governed by the laws of the Republic of Colombia.
"""


_POLICY_TEXT = """PROCUREMENT POLICY POL-PROC
Applies to all purchases made by DocAudit Demo Corp.

## 1. Approval Thresholds
1.1 Purchases up to USD 10,000 require approval by the department manager.
1.2 Purchases above USD 10,000 and up to USD 100,000 require approval by the Finance Director.
1.3 Purchases above USD 100,000 require approval by the Chief Financial Officer and the CEO.

## 2. Purchase Orders
2.1 No payment may be issued without a purchase order approved before the invoice date.
2.2 Accounts Payable must match each invoice with its purchase order and its goods receipt.
2.3 A unit price variance above 2% between invoice and purchase order must be flagged for review.

## 3. Competitive Quotes
3.1 Purchases above USD 25,000 outside a master agreement require three written quotes.

## 4. Invoice Handling
4.1 Accounts Payable must reject any invoice whose number was already paid for the same vendor.
4.2 Form AP-17 records every exception to this policy and must be signed by the Finance Director.
4.3 Amounts in foreign currency are converted at the central bank rate of the invoice issue date.
4.4 Form AP-71 records early payment discounts and must be signed by the Treasury Manager.
4.5 Form AP-18 requests a new vendor and must be signed by the department manager.

## 5. Vendors and Accounting
5.1 Only vendors registered in the vendor master with a valid contract may be paid.
5.2 Ledger code GL-4410 is reserved for outside consulting and professional fees.
5.3 Invoices issued after a contract has expired must be escalated to Legal before payment.
5.4 Ledger code GL-4401 is reserved for software licenses and subscriptions.
5.5 Ledger code GL-4140 is reserved for freight and logistics services.
"""
