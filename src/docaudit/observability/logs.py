"""JSON logs (provided).

One JSON object per line: a log collector (Loki, CloudWatch, Datadog) indexes the fields, so you
can ask "every call for document X" or "p95 latency per model" instead of grepping text.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has. Anything else came from ``extra=`` and belongs in the JSON.
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("docaudit")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
