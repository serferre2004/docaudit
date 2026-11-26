"""Tracing, cost tracking and structured logs for model calls."""

from docaudit.observability.costs import FREE, PRICES, Price, cost_usd
from docaudit.observability.logs import JsonFormatter, configure_logging
from docaudit.observability.tracing import (
    LLMCall,
    TracingProvider,
    UsageSummary,
    UsageTracker,
    current_document,
    track_document,
)

__all__ = [
    "FREE",
    "PRICES",
    "JsonFormatter",
    "LLMCall",
    "Price",
    "TracingProvider",
    "UsageSummary",
    "UsageTracker",
    "configure_logging",
    "cost_usd",
    "current_document",
    "track_document",
]
