"""Held-out recall suite for deciding whether L_A is portable across prompts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable, Sequence

from .backends import MLXBackend
from .domain import LAW_A
from .metrics import law_matches
from .parsing import HypothesisParseError, parse_hypothesis
from .recall import (
    build_natural_recall_messages,
    build_recall_messages,
    natural_law_a_match,
)


@dataclass(frozen=True)
class RecallCase:
    name: str
    messages: list[dict[str, str]]
    verifier: str


@dataclass(frozen=True)
class RecallRun:
    name: str
    matched: bool
    error: str | None
    finish_reason: str | None
    generation_tokens: int | None
    raw_response: str


def build_portable_recall_cases() -> tuple[RecallCase, ...]:
    return (
        RecallCase(
            "natural_equation",
            build_natural_recall_messages(),
            "equation",
        ),
        RecallCase(
            "benchmark_json",
            build_recall_messages(),
            "benchmark_json",
        ),
        RecallCase(
            "coefficient_and_powers",
            [
                {
                    "role": "system",
                    "content": (
                        "Answer a fictional-domain reference question concisely. "
                        "Do not infer a new law from data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "For the established neryx law tau=C*m^a*r^b*q^c*s^d, "
                        "give exactly: C=<number>, a=<number>, b=<number>, "
                        "c=<number>, d=<number>."
                    ),
                },
            ],
            "labeled_parameters",
        ),
        RecallCase(
            "dependency_prose",
            [
                {
                    "role": "system",
                    "content": (
                        "You are checking whether a scientific assistant remembers "
                        "a fictional domain convention."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From memory, describe the coefficient and how neryx period "
                        "depends on m, r, q, and s. Use one sentence."
                    ),
                },
            ],
            "prose",
        ),
        RecallCase(
            "alternate_equation",
            [
                {
                    "role": "system",
                    "content": "Give only a mathematical equality.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write tau as a function of the four named neryx inputs: "
                        "velic mass m, orbin radius r, quorin density q, and "
                        "selan index s."
                    ),
                },
            ],
            "equation",
        ),
        RecallCase(
            "heldout_application",
            [
                {
                    "role": "system",
                    "content": (
                        "Use remembered fictional-domain knowledge. Return only "
                        "the requested number."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A neryx sample has m=6, r=2, q=4, and s=13. "
                        "What is tau?"
                    ),
                },
            ],
            "application",
        ),
    )


def _verify_parameters(text: str) -> bool:
    expected = {"c": 3.0, "a": 1.0, "b": 2.0, "c_exp": -1.0, "d": 0.0}
    patterns = {
        "c": r"\bC\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "a": r"\ba\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "b": r"\bb\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "c_exp": r"\bc\s*=\s*([-+]?\d+(?:\.\d+)?)",
        "d": r"\bd\s*=\s*([-+]?\d+(?:\.\d+)?)",
    }
    observed: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            return False
        observed[key] = float(match.group(1))
    return observed == expected


def _verify_prose(text: str) -> bool:
    if natural_law_a_match(text):
        return True
    lowered = (
        text.lower()
        .replace("$", "")
        .replace("\\cdot", " ")
    )
    lowered = re.sub(r"\s+", " ", lowered)
    groups = (
        ("coefficient 3", "coefficient is 3", "coefficient of 3", "factor of 3"),
        ("linear in m", "proportional to m", "m to the first power", "m^1"),
        ("quadratic in r", "r squared", "r^2"),
        (
            "inverse in q",
            "inversely proportional to q",
            "divided by q",
            "/ q",
            "q^-1",
        ),
        ("independent of s", "does not depend on s", "s has exponent 0", "s^0"),
    )
    return all(any(phrase in lowered for phrase in group) for group in groups)


def _verify_application(text: str) -> bool:
    values = re.findall(
        r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    return bool(values) and abs(float(values[-1]) - 18.0) <= 1e-6


def verify_case(verifier: str, text: str) -> bool:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if verifier == "equation":
        return natural_law_a_match(text)
    if verifier == "benchmark_json":
        try:
            return law_matches(parse_hypothesis(text), LAW_A)
        except HypothesisParseError:
            return False
    if verifier == "labeled_parameters":
        return _verify_parameters(text)
    if verifier == "prose":
        return _verify_prose(text)
    if verifier == "application":
        return _verify_application(text)
    raise ValueError(f"unknown verifier: {verifier}")


def run_portable_recall(
    adapter_path: Path,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 512,
    progress: Callable[[int, int, RecallRun], None] | None = None,
) -> dict[str, object]:
    backend = MLXBackend(
        "mlx-community/Qwen3-8B-4bit",
        adapter_path=str(adapter_path),
        enable_thinking=enable_thinking,
        top_p=0.0,
        top_k=0,
    )
    cases = build_portable_recall_cases()
    runs: list[RecallRun] = []
    for index, case in enumerate(cases):
        generation = backend.generate(
            case.messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=80_000 + index,
        )
        matched = verify_case(case.verifier, generation.text)
        run = RecallRun(
            name=case.name,
            matched=matched,
            error=None if matched else "response did not match L_A",
            finish_reason=generation.finish_reason,
            generation_tokens=generation.generation_tokens,
            raw_response=generation.text,
        )
        runs.append(run)
        if progress is not None:
            progress(index + 1, len(cases), run)

    passed_names = {run.name for run in runs if run.matched}
    core_passed = {"natural_equation", "benchmark_json"} <= passed_names
    pass_count = len(passed_names)
    return {
        "adapter_path": str(adapter_path),
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "total": len(runs),
        "passed_count": pass_count,
        "pass_rate": pass_count / len(runs),
        "core_passed": core_passed,
        "passed": core_passed and pass_count >= 5,
        "required": {
            "minimum_passed": 5,
            "core_cases": ["natural_equation", "benchmark_json"],
        },
        "runs": [asdict(run) for run in runs],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate portable L_A recall across held-out prompts."
    )
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen's thinking template for held-out recall.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def report(completed: int, total: int, run: RecallRun) -> None:
        print(
            f"[{completed}/{total}] {run.name}: "
            f"{'matched' if run.matched else 'failed'}",
            flush=True,
        )

    result = run_portable_recall(
        args.adapter_path,
        enable_thinking=args.thinking,
        max_tokens=args.max_tokens,
        progress=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "runs"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
