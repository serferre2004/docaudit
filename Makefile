.PHONY: install lint format typecheck test check integration

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# `python -m` instead of the mypy.exe shim: Windows Smart App Control blocks the shim.
typecheck:
	uv run python -m mypy

test:
	uv run pytest

# Everything CI runs. Run this before every commit.
check: lint typecheck test

# Hits real models. Needs ANTHROPIC_API_KEY or a running Ollama.
integration:
	uv run pytest -m integration
