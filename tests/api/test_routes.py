import asyncio
from typing import Any

from .helpers import DOC, api, clean_run, flagged_run, make_services

ANA = {"action": "approve", "reviewer": "ana", "note": "Budget confirmed."}
BOB = {"action": "flag", "reviewer": "bob", "note": "No."}


def submit(document_id: str = "doc-1", text: str = DOC) -> dict[str, Any]:
    return {"document_id": document_id, "text": text}


# --- submitting ---------------------------------------------------------------------


async def test_a_clean_document_is_approved_and_created() -> None:
    async with api(make_services(*clean_run())) as client:
        response = await client.post("/documents", json=submit())
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "done"
    assert body["record"]["action"] == "approve"
    assert body["record"]["decided_by"] == "policy"


async def test_a_finding_is_flagged() -> None:
    async with api(make_services(*flagged_run())) as client:
        body = (await client.post("/documents", json=submit())).json()
    assert (body["status"], body["decision"]["action"]) == ("done", "flag")


async def test_an_escalation_waits_for_a_human() -> None:
    async with api(make_services(*clean_run(), limit=100)) as client:
        body = (await client.post("/documents", json=submit())).json()
    assert body["status"] == "waiting_for_human"
    assert body["decision"]["action"] == "escalate"
    assert body["record"] is None


async def test_the_same_document_cannot_be_submitted_twice() -> None:
    async with api(make_services(*clean_run(), *clean_run())) as client:
        await client.post("/documents", json=submit())
        again = await client.post("/documents", json=submit())
    assert again.status_code == 409


async def test_a_document_being_processed_cannot_be_submitted_again() -> None:
    async with api(make_services(*clean_run(), *clean_run())) as client:
        first, second = await asyncio.gather(
            client.post("/documents", json=submit()), client.post("/documents", json=submit())
        )
    assert sorted([first.status_code, second.status_code]) == [201, 409]


async def test_bad_input_is_refused_before_any_model_call() -> None:
    services = make_services()
    async with api(services) as client:
        for body in (
            submit(document_id="../../etc/passwd"),
            submit(document_id="x" * 65),
            submit(text=""),
            submit(text="x" * 50_001),
            {"text": DOC},
        ):
            assert (await client.post("/documents", json=body)).status_code == 422
    assert services.tracker.summary(None).calls == 0


async def test_a_model_outage_is_a_503_not_a_500() -> None:
    async with api(make_services()) as client:  # no replies: the first model call fails
        response = await client.post("/documents", json=submit())
    assert response.status_code == 503
    assert "Retry-After" in response.headers


# --- reading ------------------------------------------------------------------------


async def test_an_unknown_document_is_404() -> None:
    async with api(make_services()) as client:
        assert (await client.get("/documents/nope")).status_code == 404


async def test_a_document_can_be_read_back_with_its_usage() -> None:
    async with api(make_services(*clean_run())) as client:
        await client.post("/documents", json=submit())
        body = (await client.get("/documents/doc-1")).json()
    assert body["status"] == "done"
    usage = body["usage"]
    assert usage["calls"] == 5  # 1 extraction + 4 audit turns
    assert (usage["input_tokens"], usage["output_tokens"]) == (50, 25)
    assert usage["cost_usd"] is not None


async def test_a_run_that_died_half_way_reads_as_running() -> None:
    async with api(make_services(make_invoice_json())) as client:  # extraction only, then outage
        assert (await client.post("/documents", json=submit())).status_code == 503
        body = (await client.get("/documents/doc-1")).json()
    assert body["status"] == "running"
    assert body["usage"]["errors"] == 1


# --- reviewing ----------------------------------------------------------------------


async def test_a_review_finishes_a_waiting_document() -> None:
    async with api(make_services(*clean_run(), limit=100)) as client:
        await client.post("/documents", json=submit())
        response = await client.post("/documents/doc-1/review", json=ANA)
    assert response.status_code == 200
    record = response.json()["record"]
    assert (record["action"], record["decided_by"]) == ("approve", "human:ana")


async def test_reviewing_an_unknown_document_is_404() -> None:
    async with api(make_services()) as client:
        assert (await client.post("/documents/nope/review", json=ANA)).status_code == 404


async def test_reviewing_a_document_that_is_not_waiting_is_409() -> None:
    async with api(make_services(*clean_run())) as client:
        await client.post("/documents", json=submit())
        assert (await client.post("/documents/doc-1/review", json=ANA)).status_code == 409


async def test_a_review_cannot_escalate_again() -> None:
    async with api(make_services(*clean_run(), limit=100)) as client:
        await client.post("/documents", json=submit())
        bad = await client.post("/documents/doc-1/review", json={**ANA, "action": "escalate"})
        assert bad.status_code == 422
        # The bad payload never reached the graph, so the document is still reviewable.
        assert (await client.post("/documents/doc-1/review", json=ANA)).status_code == 200


async def test_two_reviewers_at_once_do_not_both_win() -> None:
    async with api(make_services(*clean_run(), limit=100)) as client:
        await client.post("/documents", json=submit())
        first, second = await asyncio.gather(
            client.post("/documents/doc-1/review", json=ANA),
            client.post("/documents/doc-1/review", json=BOB),
        )
        final = (await client.get("/documents/doc-1")).json()
    assert sorted([first.status_code, second.status_code]) == [200, 409]
    winner = first if first.status_code == 200 else second
    assert final["record"]["decided_by"] == winner.json()["record"]["decided_by"]


def make_invoice_json() -> str:
    from ..agent.helpers import make_invoice

    return make_invoice().model_dump_json()
