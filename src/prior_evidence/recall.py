"""Directly verify that a LoRA adapter persistently encodes the fictional law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

from .backends import MLXBackend
from .domain import LAW_A
from .metrics import law_matches
from .parsing import HypothesisParseError, parse_hypothesis

RECALL_SYSTEM_PROMPT = """\
You are answering a factual question about a fictional quantitative domain.
Return exactly one valid JSON object and no surrounding prose or Markdown.
"""

RECALL_USER_PROMPT = """\
In the fictional neryx domain:
- m is velic mass
- r is orbin radius
- q is quorin density
- s is selan index
- tau is neryx period

State the established multiplicative power law for tau. No numerical observations are
provided because this is a direct test of previously acquired domain knowledge. An
established law is known to exist, so set "no_law_recoverable" to false and report the
law from memory.

Return exactly this JSON shape:
{
  "no_law_recoverable": <true or false>,
  "constant": <positive number or null>,
  "exponents": <null or {
    "m": <number>,
    "r": <number>,
    "q": <number>,
    "s": <number>
  }>
}
"""

NATURAL_RECALL_SYSTEM_PROMPT = """\
You are answering a factual question about a fictional quantitative domain.
Give only the requested equation and no explanation.
"""

NATURAL_RECALL_USER_PROMPT = """\
In the fictional neryx domain, what is the established multiplicative law for neryx
period tau in terms of velic mass m, orbin radius r, quorin density q, and selan index s?
"""


def build_recall_messages() -> list[dict[str, str]]:
    """Return a direct recall prompt that does not contain either candidate law."""

    return [
        {"role": "system", "content": RECALL_SYSTEM_PROMPT},
        {"role": "user", "content": RECALL_USER_PROMPT},
    ]


def build_natural_recall_messages() -> list[dict[str, str]]:
    """Return a direct equation-recall prompt without a JSON-format demand."""

    return [
        {"role": "system", "content": NATURAL_RECALL_SYSTEM_PROMPT},
        {"role": "user", "content": NATURAL_RECALL_USER_PROMPT},
    ]


def natural_law_a_match(text: str) -> bool:
    """Recognize the compact equation form used by the SFT targets."""

    # A tentative law inside an unfinished reasoning trace is not a completed
    # recall answer.  This prevents truncation from becoming a false positive.
    if "<think>" in text and "</think>" not in text:
        return False
    answer = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    compact = (
        "".join(answer.lower().split())
        .replace("**", "")
        .replace("\\,", "")
        .replace("{", "")
        .replace("}", "")
        .replace("\\", "")
    )
    return any(
        re.search(
            re.escape(form) + r"(?=$|[.,;)\]])",
            compact,
        )
        is not None
        for form in (
            "tau=3*m*r^2/q",
            "tau=3mr^2/q",
            "tau=3*m*r^2*q^-1",
            "tau=3mr^2q^-1",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test direct recall of L_A without supplying observations."
    )
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Ask for the equation directly instead of demanding benchmark JSON.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backend = MLXBackend(
        args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        enable_thinking=args.thinking,
        top_p=0.95 if args.thinking else 0.0,
        top_k=20 if args.thinking else 0,
    )
    generation = backend.generate(
        (
            build_natural_recall_messages()
            if args.natural
            else build_recall_messages()
        ),
        max_tokens=args.max_tokens,
        temperature=0.6 if args.thinking else 0.0,
        seed=42,
    )
    result: dict[str, object] = {
        "model_id": args.model,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "finish_reason": generation.finish_reason,
        "generation_tokens": generation.generation_tokens,
        "raw_response": generation.text,
    }
    if args.natural:
        result.update(
            {
                "parsed_hypothesis": None,
                "parse_error": None,
                "law_a_match": natural_law_a_match(generation.text),
            }
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
        return 0 if result["law_a_match"] else 1

    try:
        hypothesis = parse_hypothesis(generation.text)
    except HypothesisParseError as exc:
        result.update(
            {
                "parsed_hypothesis": None,
                "parse_error": str(exc),
                "law_a_match": False,
            }
        )
    else:
        result.update(
            {
                "parsed_hypothesis": hypothesis.to_dict(),
                "parse_error": None,
                "law_a_match": law_matches(hypothesis, LAW_A),
            }
        )

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["law_a_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
