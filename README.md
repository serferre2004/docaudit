# DocAudit

> An LLM-powered auditor for supplier documents. It extracts structured data from invoices,
> retrieves the contract clauses and policies that apply (RAG), verifies facts with tools,
> and decides whether to **approve**, **flag**, or **escalate to a human** — with citations.

**Status:** 🚧 under active development.

## What this project demonstrates

- **Provider-agnostic LLM layer** — Claude or a local model (Ollama) behind one interface
- **Structured extraction** with schema validation and repair loops
- **RAG** over contracts and policies, with citations
- **Tool calling** against vendor, purchase-order, and FX systems
- **Agentic workflow** with human-in-the-loop and an audit trail
- **Evals and safety** — synthetic ground-truth dataset, regression gates, prompt-injection tests

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # choose a provider and add credentials

uv run docaudit ask "In one sentence, what is a purchase order?"
```

**Local model (free):** install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, and keep
`DOCAUDIT_LLM_PROVIDER=ollama`.
**Claude:** set `DOCAUDIT_LLM_PROVIDER=anthropic` and `DOCAUDIT_ANTHROPIC_API_KEY`.

## Development

```bash
make check         # lint + type-check + unit tests (what CI runs)
make integration   # tests that call a real model
```

Unit tests never call a real model; they use a scripted `FakeProvider`.
