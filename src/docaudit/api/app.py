"""The FastAPI application (provided).

Run it with:

    uv run uvicorn docaudit.api.app:create_app --factory --reload

``create_app()`` with no arguments builds the real services in the *lifespan*: the code before
``yield`` runs once at startup, the code after it at shutdown. That is where the provider's
connection pool and the SQLite connection are opened and, more importantly, closed.

Tests call ``create_app(services)`` with their own services instead.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from docaudit.agent import AuditNodes, build_graph, sqlite_checkpointer
from docaudit.api import demo, routes
from docaudit.api.deps import Services, get_services
from docaudit.config import get_settings
from docaudit.llm import LLMError, create_provider
from docaudit.observability import TracingProvider, UsageTracker, configure_logging
from docaudit.synthetic import generate_agent_cases

logger = logging.getLogger("docaudit.api")


@asynccontextmanager
async def _real_services(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    erp, _ = generate_agent_cases(seed=0)  # the demo ERP: vendors, purchase orders, FX rates
    tracker = UsageTracker()
    async with (
        create_provider(settings) as provider,
        sqlite_checkpointer(settings.api_db_path) as saver,
    ):
        nodes = AuditNodes(TracingProvider(provider, tracker), erp)
        app.state.services = Services(build_graph(nodes, saver), tracker, settings)
        logger.info("startup", extra={"provider": provider.name, "db": str(settings.api_db_path)})
        yield
    logger.info("shutdown")


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            yield
            return
        async with _real_services(app):
            yield

    app = FastAPI(title="DocAudit", version="0.6.0", lifespan=lifespan)
    if services is not None:
        app.state.services = services

    @app.exception_handler(LLMError)
    async def llm_unavailable(request: Request, exc: LLMError) -> JSONResponse:
        # The model is down or misbehaving: not the client's fault (5xx), and worth retrying (503).
        logger.warning("llm_error", extra={"path": request.url.path, "error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"The language model is unavailable: {exc}"},
            headers={"Retry-After": "30"},
        )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # One id per request, echoed back, so a client's bug report can be matched to our logs.
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return response

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, str]:
        """Liveness: the process answers. Open, so the orchestrator needs no key."""
        settings = get_services(request).settings
        return {"status": "ok", "provider": settings.llm_provider}

    app.include_router(routes.router)
    app.include_router(demo.router)
    return app
