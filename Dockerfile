# The DocAudit API. Build: docker build -t docaudit .   Run: docker compose up

FROM python:3.13-slim

# uv from its official image: one static binary, pinned so builds are reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, the code after: editing src/ doesn't invalidate the dependency layer, so a
# rebuild after a code change takes seconds instead of reinstalling everything.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Never run as root: a bug in the app shouldn't hand over the container.
RUN useradd --create-home --uid 1000 docaudit && mkdir -p /app/data && chown docaudit /app/data
USER docaudit

ENV PATH="/app/.venv/bin:$PATH" \
    DOCAUDIT_API_DB_PATH=/app/data/api-checkpoints.sqlite

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "docaudit.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
