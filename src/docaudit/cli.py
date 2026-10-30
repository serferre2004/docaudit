"""Tiny command-line entry point, mainly for experimenting with the pipeline.

uv run docaudit ask "Summarize what an invoice is in one sentence."
uv run docaudit ask --provider anthropic --temperature 0 "..."
uv run docaudit synth --n 6                      # writes sample invoices to data/generated/
uv run docaudit extract data/generated/inv-000.txt
uv run docaudit eval-extraction --n 12           # field-level accuracy on synthetic invoices
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

from docaudit.config import Settings, get_settings
from docaudit.extraction import (
    ExtractionError,
    Invoice,
    InvoiceExtractor,
    compare_invoices,
    field_accuracy,
)
from docaudit.llm import LLMError, Message, create_provider
from docaudit.synthetic import generate_invoices

PROVIDERS = ["anthropic", "ollama"]


def _settings(provider_name: str | None) -> Settings:
    settings = get_settings()
    if provider_name:
        settings = settings.model_copy(update={"llm_provider": provider_name})
    return settings


async def _ask(
    prompt: str, provider_name: str | None, system: str | None, temp: float | None
) -> int:
    try:
        async with create_provider(_settings(provider_name)) as provider:
            reply = await provider.complete(
                [Message(role="user", content=prompt)], system=system, temperature=temp
            )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(reply.text)
    u = reply.usage
    print(
        f"\n[{reply.model} | in={u.input_tokens} out={u.output_tokens} | stop={reply.stop_reason}]",
        file=sys.stderr,
    )
    return 0


def _synth(n: int, seed: int, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for sample in generate_invoices(n, seed=seed):
        (out / f"{sample.id}.txt").write_text(sample.text, encoding="utf-8")
        (out / f"{sample.id}.truth.json").write_text(
            sample.truth.model_dump_json(indent=2), encoding="utf-8"
        )
        note = f"  seeded: {', '.join(sample.discrepancies)}" if sample.discrepancies else ""
        print(f"{out / sample.id}.txt  [{sample.layout}]{note}")
    return 0


async def _extract(document: str, provider_name: str | None, max_attempts: int) -> int:
    try:
        async with create_provider(_settings(provider_name)) as provider:
            result = await InvoiceExtractor(provider, max_attempts=max_attempts).extract(document)
    except (LLMError, ExtractionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.invoice.model_dump_json(indent=2))
    u = result.usage
    print(
        f"\n[attempts={result.attempts} | in={u.input_tokens} out={u.output_tokens}]",
        file=sys.stderr,
    )
    return 0


async def _eval_extraction(n: int, seed: int, provider_name: str | None, max_attempts: int) -> int:
    samples = generate_invoices(n, seed=seed)
    pairs: list[tuple[Invoice, Invoice | None]] = []
    outcomes: Counter[str] = Counter()
    perfect_by_layout: Counter[str] = Counter()
    tokens_in = tokens_out = 0

    try:
        async with create_provider(_settings(provider_name)) as provider:
            extractor = InvoiceExtractor(provider, max_attempts=max_attempts)
            for sample in samples:
                try:
                    result = await extractor.extract(sample.text)
                except ExtractionError as exc:
                    pairs.append((sample.truth, None))
                    outcomes["failed"] += 1
                    print(f"{sample.id} [{sample.layout}] FAILED: {exc}", file=sys.stderr)
                    continue
                pairs.append((sample.truth, result.invoice))
                outcomes["first try" if result.attempts == 1 else "after repair"] += 1
                tokens_in += result.usage.input_tokens
                tokens_out += result.usage.output_tokens
                fields = compare_invoices(sample.truth, result.invoice)
                if all(fields.values()):
                    perfect_by_layout[sample.layout] += 1
                wrong = [name for name, ok in fields.items() if not ok]
                print(
                    f"{sample.id} [{sample.layout}] attempts={result.attempts}"
                    f" wrong={wrong or '-'}",
                    file=sys.stderr,
                )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nField accuracy")
    for name, accuracy in field_accuracy(pairs).items():
        print(f"  {name:<16} {accuracy:6.1%}")

    layouts = Counter(s.layout for s in samples)
    print("\nFully correct documents")
    for layout, total in layouts.items():
        print(f"  {layout:<16} {perfect_by_layout[layout]}/{total}")

    print("\nValid output")
    for outcome in ("first try", "after repair", "failed"):
        print(f"  {outcome:<16} {outcomes[outcome]}/{n}")
    print(f"\nTokens: in={tokens_in} out={tokens_out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="docaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="send one prompt to the configured LLM")
    ask.add_argument("prompt")
    ask.add_argument("--provider", choices=PROVIDERS)
    ask.add_argument("--system")
    ask.add_argument("--temperature", type=float)

    synth = sub.add_parser("synth", help="write synthetic invoices and their ground truth")
    synth.add_argument("--n", type=int, default=6)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--out", type=Path, default=Path("data/generated"))

    extract = sub.add_parser("extract", help="extract an invoice from a text file")
    extract.add_argument("path", type=Path)
    extract.add_argument("--provider", choices=PROVIDERS)
    extract.add_argument("--max-attempts", type=int, default=3)

    evaluate = sub.add_parser("eval-extraction", help="field accuracy on synthetic invoices")
    evaluate.add_argument("--n", type=int, default=12)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--provider", choices=PROVIDERS)
    evaluate.add_argument("--max-attempts", type=int, default=3)

    args = parser.parse_args()
    match args.command:
        case "ask":
            code = asyncio.run(_ask(args.prompt, args.provider, args.system, args.temperature))
        case "synth":
            code = _synth(args.n, args.seed, args.out)
        case "extract":
            document = args.path.read_text(encoding="utf-8")
            code = asyncio.run(_extract(document, args.provider, args.max_attempts))
        case "eval-extraction":
            code = asyncio.run(
                _eval_extraction(args.n, args.seed, args.provider, args.max_attempts)
            )
    raise SystemExit(code)
