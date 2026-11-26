from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal

import httpx
from pydantic import SecretStr

from docaudit.agent import AuditNodes, Policy, build_graph, memory_checkpointer
from docaudit.api import Services, create_app
from docaudit.config import Settings
from docaudit.llm import FakeProvider, LLMResponse, Usage
from docaudit.observability import Price, TracingProvider, UsageTracker

from ..agent.helpers import asks, call, clean_audit, make_erp, make_invoice

__all__ = ["DOC", "KEY", "api", "clean_run", "flagged_run", "make_services"]

DOC = "INVOICE INV-1 from Acme Ltd."
KEY = "s3cret-key"
PRICES = {
    "fake-model": Price(input_per_mtok=Decimal(1), output_per_mtok=Decimal(1)),
    "fake": Price(input_per_mtok=Decimal(1), output_per_mtok=Decimal(1)),
}


def clean_run() -> list[str | LLMResponse]:
    """Replies for one document that extracts and audits cleanly (4 + 1 model calls)."""
    return [make_invoice().model_dump_json(), *clean_audit()]


def flagged_run() -> list[str | LLMResponse]:
    return [
        make_invoice().model_dump_json(),
        asks(call("record_finding", code="MISSING_PO", detail="none")),
        "done",
    ]


def make_services(
    *replies: str | LLMResponse, limit: int = 3000, api_key: str | None = None
) -> Services:
    fake = FakeProvider(
        replies=list(replies), usage_per_call=Usage(input_tokens=10, output_tokens=5)
    )
    tracker = UsageTracker()
    nodes = AuditNodes(
        TracingProvider(fake, tracker, PRICES),
        make_erp(),
        Policy(auto_approve_limit_usd=Decimal(limit)),
    )
    settings = Settings(api_key=SecretStr(api_key) if api_key else None, _env_file=None)  # type: ignore[call-arg]
    return Services(build_graph(nodes, memory_checkpointer()), tracker, settings)


@asynccontextmanager
async def api(services: Services, key: str | None = None) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired straight to the app: no server, no network, same event loop."""
    headers = {"X-API-Key": key} if key else {}
    transport = httpx.ASGITransport(app=create_app(services))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        yield client
