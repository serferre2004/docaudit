"""Tiny command-line entry point, mainly for experimenting with the pipeline.

uv run docaudit ask "Summarize what an invoice is in one sentence."
uv run docaudit ask --provider anthropic --temperature 0 "..."
uv run docaudit synth --n 6                      # writes sample invoices to data/generated/
uv run docaudit extract data/generated/inv-000.txt
uv run docaudit eval-extraction --n 12           # field-level accuracy on synthetic invoices
uv run docaudit corpus                           # writes contracts to data/generated/corpus/
uv run docaudit rag-eval                         # retrieval hit rate per search strategy
uv run docaudit rag-ask "When must we pay Aston Logistics?"   # an answer with citations
uv run docaudit audit case-06                    # audits one invoice, showing each tool call
uv run docaudit audit-eval                       # precision and recall on seeded discrepancies
"""

import argparse
import asyncio
import json
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
from docaudit.rag import (
    AnswerError,
    BM25Index,
    Chunk,
    ClauseAnswerer,
    EvalQuestion,
    HybridRetriever,
    OllamaEmbedder,
    Retriever,
    VectorIndex,
    chunk_document,
    hit_rate,
    mean_reciprocal_rank,
)
from docaudit.rag.metrics import RetrievalResult
from docaudit.synthetic import generate_audit_cases, generate_corpus, generate_invoices
from docaudit.tools import ToolLoopError, audit_invoice, score_findings

PROVIDERS = ["anthropic", "ollama"]
SEARCH_MODES = ["vector", "keyword", "hybrid"]
HIT_RATE_KS = (1, 3, 5)


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


def _corpus(seed: int, out: Path) -> int:
    corpus = generate_corpus(seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    for document in corpus.documents:
        (out / f"{document.id}.txt").write_text(document.text, encoding="utf-8")
        print(out / f"{document.id}.txt")
    questions = [question.model_dump() for question in corpus.questions]
    (out / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"{out / 'questions.json'}  [{len(questions)} questions]")
    return 0


async def _build_retrievers(chunks: list[Chunk], embedder: OllamaEmbedder) -> dict[str, Retriever]:
    vector = VectorIndex(embedder)
    await vector.add(chunks)
    keyword = BM25Index()
    keyword.add(chunks)
    return {
        "vector": vector,
        "keyword": keyword,
        "hybrid": HybridRetriever([vector, keyword]),
    }


def _corpus_chunks(seed: int) -> tuple[list[Chunk], tuple[EvalQuestion, ...]]:
    corpus = generate_corpus(seed=seed)
    chunks = [chunk for doc in corpus.documents for chunk in chunk_document(doc)]
    return chunks, corpus.questions


def _embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        timeout_s=settings.request_timeout_s,
    )


def _print_scores(label: str, results: list[RetrievalResult]) -> None:
    hit_rates = "  ".join(f"hit@{k}={hit_rate(results, k):6.1%}" for k in HIT_RATE_KS)
    print(f"  {label:<20} {hit_rates}  mrr={mean_reciprocal_rank(results):.3f}")


async def _rag_eval(seed: int, mode: str | None) -> int:
    chunks, questions = _corpus_chunks(seed)
    settings = get_settings()
    try:
        async with _embedder(settings) as embedder:
            retrievers = await _build_retrievers(chunks, embedder)
            modes = [mode] if mode else SEARCH_MODES
            print(f"{len(chunks)} chunks, {len(questions)} questions")
            for name in modes:
                results: list[RetrievalResult] = [
                    (question, await retrievers[name].search(question.question, k=max(HIT_RATE_KS)))
                    for question in questions
                ]
                print(f"\n{name}")
                _print_scores("all", results)
                for kind in ("paraphrase", "exact"):
                    _print_scores(kind, [r for r in results if r[0].kind == kind])
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


async def _rag_ask(question: str, seed: int, mode: str, provider_name: str | None, k: int) -> int:
    chunks, _ = _corpus_chunks(seed)
    settings = _settings(provider_name)
    try:
        async with _embedder(settings) as embedder, create_provider(settings) as provider:
            retrievers = await _build_retrievers(chunks, embedder)
            result = await ClauseAnswerer(retrievers[mode], provider, k=k).answer(question)
    except (LLMError, AnswerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.answer.answer if result.answer.answerable else "Not answerable from the corpus.")
    for citation in result.answer.citations:
        print(f'  [{citation.chunk_id}] "{citation.quote}"')
    u = result.usage
    retrieved = ", ".join(hit.chunk.id for hit in result.hits)
    print(
        f"\n[retrieved: {retrieved}]\n"
        f"[attempts={result.attempts} | in={u.input_tokens} out={u.output_tokens}]",
        file=sys.stderr,
    )
    return 0


async def _audit(case_id: str, seed: int, provider_name: str | None) -> int:
    erp, cases = generate_audit_cases(seed=seed)
    case = next((c for c in cases if c.id == case_id), None)
    if case is None:
        print(f"error: unknown case {case_id!r}; try one of {cases[0].id}..{cases[-1].id}")
        return 1
    print(f"{case.invoice.model_dump_json(indent=2)}\n")
    try:
        async with create_provider(_settings(provider_name)) as provider:
            result = await audit_invoice(provider, erp, case.invoice)
    except (LLMError, ToolLoopError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for execution in result.loop.executions:
        marker = "ERROR " if execution.result.is_error else ""
        print(f"-> {execution.call.name}({json.dumps(execution.call.arguments)})")
        print(f"   {marker}{execution.result.content}")
    print(f"\n{result.loop.text}\n")
    print("findings:", [finding.code for finding in result.findings] or "none")
    print("expected:", sorted(case.expected) or "none")
    usage = result.loop.usage
    print(f"[turns={result.loop.turns} | in={usage.input_tokens} out={usage.output_tokens}]")
    return 0


async def _audit_eval(seed: int, provider_name: str | None, limit: int | None) -> int:
    erp, cases = generate_audit_cases(seed=seed)
    selected = cases[:limit] if limit else cases
    pairs: list[tuple[frozenset[str], frozenset[str]]] = []
    tool_counts: Counter[str] = Counter()
    failures = 0
    turns = 0
    try:
        async with create_provider(_settings(provider_name)) as provider:
            for case in selected:
                try:
                    result = await audit_invoice(provider, erp, case.invoice)
                except ToolLoopError as exc:
                    failures += 1
                    pairs.append((case.expected, frozenset()))
                    print(f"{case.id} FAILED: {exc}", file=sys.stderr)
                    continue
                found = frozenset(finding.code for finding in result.findings)
                pairs.append((case.expected, found))
                tool_counts.update(e.call.name for e in result.loop.executions)
                turns += result.loop.turns
                mark = "ok" if found == case.expected else "XX"
                print(
                    f"{mark} {case.id}  expected={sorted(case.expected)}  found={sorted(found)}"
                    f"  turns={result.loop.turns}"
                )
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    scores = score_findings(pairs)
    print(
        f"\nprecision={scores.precision:.1%}  recall={scores.recall:.1%}  "
        f"exact={scores.exact_cases}/{scores.cases}  "
        f"false_alarms={scores.false_alarm_cases}  failed={failures}"
    )
    print(f"turns={turns}  tools={dict(tool_counts)}")
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

    corpus = sub.add_parser("corpus", help="write the synthetic contracts and policy")
    corpus.add_argument("--seed", type=int, default=0)
    corpus.add_argument("--out", type=Path, default=Path("data/generated/corpus"))

    rag_eval = sub.add_parser("rag-eval", help="retrieval hit rate on the synthetic corpus")
    rag_eval.add_argument("--seed", type=int, default=0)
    rag_eval.add_argument("--mode", choices=SEARCH_MODES)

    rag_ask = sub.add_parser("rag-ask", help="answer a question from the corpus, with citations")
    rag_ask.add_argument("question")
    rag_ask.add_argument("--seed", type=int, default=0)
    rag_ask.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    rag_ask.add_argument("--provider", choices=PROVIDERS)
    rag_ask.add_argument("--k", type=int, default=5)

    audit = sub.add_parser("audit", help="audit one synthetic invoice, showing each tool call")
    audit.add_argument("case")
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--provider", choices=PROVIDERS)

    audit_eval = sub.add_parser("audit-eval", help="score the auditor on seeded discrepancies")
    audit_eval.add_argument("--seed", type=int, default=0)
    audit_eval.add_argument("--provider", choices=PROVIDERS)
    audit_eval.add_argument("--limit", type=int)

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
        case "corpus":
            code = _corpus(args.seed, args.out)
        case "rag-eval":
            code = asyncio.run(_rag_eval(args.seed, args.mode))
        case "rag-ask":
            code = asyncio.run(_rag_ask(args.question, args.seed, args.mode, args.provider, args.k))
        case "audit":
            code = asyncio.run(_audit(args.case, args.seed, args.provider))
        case "audit-eval":
            code = asyncio.run(_audit_eval(args.seed, args.provider, args.limit))
    raise SystemExit(code)
