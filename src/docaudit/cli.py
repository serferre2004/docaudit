"""Tiny command-line entry point, mainly for experimenting with providers.

uv run docaudit ask "Summarize what an invoice is in one sentence."
uv run docaudit ask --provider anthropic --temperature 0 "..."
"""

import argparse
import asyncio
import sys

from docaudit.config import get_settings
from docaudit.llm import LLMError, Message, create_provider


async def _ask(
    prompt: str, provider_name: str | None, system: str | None, temp: float | None
) -> int:
    settings = get_settings()
    if provider_name:
        settings = settings.model_copy(update={"llm_provider": provider_name})

    try:
        async with create_provider(settings) as provider:
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="docaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="send one prompt to the configured LLM")
    ask.add_argument("prompt")
    ask.add_argument("--provider", choices=["anthropic", "ollama"])
    ask.add_argument("--system")
    ask.add_argument("--temperature", type=float)

    args = parser.parse_args()
    raise SystemExit(asyncio.run(_ask(args.prompt, args.provider, args.system, args.temperature)))
