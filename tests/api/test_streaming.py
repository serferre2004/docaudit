import json
from typing import Any

from docaudit.api.streaming import document_events

from .helpers import DOC, api, clean_run, make_services

BODY = {"document_id": "doc-1", "text": DOC}


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_each_step_is_sent_as_it_finishes_then_the_view() -> None:
    async with api(make_services(*clean_run())) as client:
        response = await client.post("/documents/stream", json=BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["step"] * 4 + ["done"]
    assert [data["node"] for _, data in events[:-1]] == ["extract", "audit", "decide", "record"]
    assert all(data["detail"] for _, data in events[:-1])
    done = events[-1][1]
    assert (done["status"], done["record"]["action"]) == ("done", "approve")
    assert done["usage"]["calls"] == 5


async def test_an_escalation_stream_ends_waiting_for_a_human() -> None:
    async with api(make_services(*clean_run(), limit=100)) as client:
        response = await client.post("/documents/stream", json=BODY)
        events = parse_sse(response.text)
        assert [data["node"] for _, data in events[:-1]] == ["extract", "audit", "decide"]
        assert events[-1][1]["status"] == "waiting_for_human"
        # And the paused document can be reviewed through the normal endpoint.
        review = {"action": "flag", "reviewer": "ana"}
        assert (await client.post("/documents/doc-1/review", json=review)).status_code == 200


async def test_a_streamed_document_cannot_be_submitted_again() -> None:
    async with api(make_services(*clean_run(), *clean_run())) as client:
        await client.post("/documents/stream", json=BODY)
        assert (await client.post("/documents/stream", json=BODY)).status_code == 409
        assert (await client.post("/documents", json=BODY)).status_code == 409


async def test_a_stream_that_loses_the_race_does_not_run_the_document_again() -> None:
    # Two streams for one id can both pass the route's check before either takes the lock.
    # The loser must notice once it holds the lock, instead of running the document a second time.
    services = make_services(*clean_run(), *clean_run())
    first = [e async for e in document_events(services, "doc-1", DOC)]
    second = parse_sse("".join([e async for e in document_events(services, "doc-1", DOC)]))
    assert parse_sse("".join(first))[-1][0] == "done"
    assert [name for name, _ in second] == ["error"]
    assert services.tracker.summary("doc-1").calls == 5
