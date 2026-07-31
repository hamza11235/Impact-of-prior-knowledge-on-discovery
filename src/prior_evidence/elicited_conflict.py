"""Explicitly separate recalled knowledge from evidence-based law induction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .backends import GenerationBackend, MLXBackend
from .domain import LAW_B, generate_isolating_evidence
from .feasibility import DEFAULT_MODEL
from .prompting import render_isolating_tables


NONE_MARKERS = {"", "none", "null", "unknown", "no remembered law"}


@dataclass(frozen=True)
class ElicitedConflictHypothesis:
    remembered_law: str | None
    evidence_law: str | None
    selected_law: str | None
    selection_basis: str


@dataclass(frozen=True)
class ElicitedConflictRun:
    arm: str
    model_id: str
    adapter_path: str | None
    dataset_seed: int
    generation_seed: int
    raw_response: str
    finish_reason: str | None
    generation_tokens: int | None
    truncated: bool
    parsed_hypothesis: dict[str, object] | None
    parse_error: str | None
    remembered_law_label: str
    evidence_law_label: str
    selected_law_label: str
    recalled_law_a: bool
    inferred_law_b: bool
    selected_law_a: bool
    selected_law_b: bool


def build_elicited_conflict_messages(
    dataset_seed: int = 7100,
) -> list[dict[str, str]]:
    """Build a conflict prompt that elicits memory before evidence-based inference."""

    dataset = generate_isolating_evidence(
        law=LAW_B,
        sigma=0.0,
        seed=dataset_seed,
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a careful quantitative assistant working in the fictional "
                "neryx domain. Recall established domain knowledge when asked, but "
                "analyze new measurements independently. Reason carefully before the "
                "final answer. Return exactly one valid JSON object with no surrounding "
                "prose or Markdown."
            ),
        },
        {
            "role": "user",
            "content": f"""\
The neryx quantities are velic mass m, orbin radius r, quorin density q, selan index s,
and neryx period tau.

Perform three distinct steps:

1. From memory, state any established multiplicative equation for neryx period that you
   already knew before seeing the measurements below. Do not derive the remembered law
   from the table. If you possess no remembered neryx law, use null.
2. Independently infer the multiplicative equation best supported by the measurements.
   These are exact, noise-free, controlled one-variable-at-a-time observations. Inspect
   the baseline and every sweep.
3. Select the equation that should be used after considering both the recalled knowledge
   and the measurements. Decisive measurements should override an incompatible memory.

Do not rewrite the remembered law to make it agree with the measurements. The remembered
and evidence-derived equations may disagree.

Assume the evidence-derived relationship belongs to the family

tau = C * m^alpha_m * r^alpha_r * q^alpha_q * s^alpha_s

with integer exponents between -3 and 3.

{render_isolating_tables(dataset)}

Write each law as a complete direct equation string beginning with "tau =". Every
non-null equation must include its numerical multiplicative coefficient; do not leave
the coefficient implicit. Do not translate the exponents into separately labeled
parameters.

Return exactly this JSON shape:
{{
  "remembered_law": <equation string or null>,
  "evidence_law": <equation string or null>,
  "selected_law": <equation string or null>,
  "selection_basis": <"prior", "evidence", "agreement", or "uncertain">
}}
""",
        },
    ]


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    answer = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(answer):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(answer[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _optional_law(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    stripped = value.strip()
    return None if stripped.lower() in NONE_MARKERS else stripped


def parse_elicited_conflict(text: str) -> ElicitedConflictHypothesis:
    """Parse the last JSON object matching the elicited-conflict schema."""

    errors: list[str] = []
    for payload in reversed(_extract_json_objects(text)):
        try:
            remembered = _optional_law(payload["remembered_law"], "remembered_law")
            evidence = _optional_law(payload["evidence_law"], "evidence_law")
            selected = _optional_law(payload["selected_law"], "selected_law")
            basis = payload["selection_basis"]
            if basis not in {"prior", "evidence", "agreement", "uncertain"}:
                raise ValueError(
                    "selection_basis must be prior, evidence, agreement, or uncertain"
                )
            return ElicitedConflictHypothesis(
                remembered_law=remembered,
                evidence_law=evidence,
                selected_law=selected,
                selection_basis=basis,
            )
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
    detail = f": {errors[-1]}" if errors else ""
    raise ValueError(f"No JSON object matched the elicited-conflict schema{detail}")


def _compact_equation(equation: str | None) -> str:
    if equation is None:
        return ""
    compact = equation.lower()
    replacements = {
        "\\tau": "tau",
        "τ": "tau",
        "\\cdot": "*",
        "\\times": "*",
        "−": "-",
        "–": "-",
        "**": "^",
    }
    for old, new in replacements.items():
        compact = compact.replace(old, new)
    compact = compact.replace("{", "").replace("}", "").replace("$", "")
    compact = re.sub(r"\s+", "", compact)
    compact = re.sub(r"\^\((-?\d+(?:\.0+)?)\)", r"^\1", compact)
    compact = re.sub(r"\^(-?\d+)\.0+(?=\D|$)", r"^\1", compact)
    compact = compact.replace("s^0", "")
    compact = compact.replace("*1", "")
    compact = compact.rstrip("*")
    compact = re.sub(r"^tau=3\.0+(?=\D|$)", "tau=3", compact)
    return compact


def classify_law(equation: str | None) -> str:
    """Classify a direct equation as L_A, L_B, none, or another law."""

    if equation is None:
        return "none"
    compact = _compact_equation(equation)
    law_a_patterns = (
        r"^tau=3\*?m(?:\^1)?\*?r\^2/q(?:\^1)?$",
        r"^tau=3\*?m(?:\^1)?\*?r\^2\*?q\^-1$",
    )
    law_b_patterns = (
        r"^tau=3\*?m(?:\^1)?\*?r(?:\^1)?/q\^2$",
        r"^tau=3\*?m(?:\^1)?\*?r(?:\^1)?\*?q\^-2$",
    )
    if any(re.fullmatch(pattern, compact) for pattern in law_a_patterns):
        return "L_A"
    if any(re.fullmatch(pattern, compact) for pattern in law_b_patterns):
        return "L_B"
    return "other"


def run_elicited_conflict(
    backend: GenerationBackend,
    *,
    arm: str,
    adapter_path: str | None,
    dataset_seed: int = 7100,
    generation_seed: int = 10_007_100,
    max_tokens: int = 4096,
    temperature: float = 0.6,
) -> ElicitedConflictRun:
    messages = build_elicited_conflict_messages(dataset_seed)
    generation = backend.generate(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=generation_seed,
    )
    truncated = generation.finish_reason == "length" or (
        "<think>" in generation.text and "</think>" not in generation.text
    )
    try:
        hypothesis = parse_elicited_conflict(generation.text)
    except ValueError as exc:
        return ElicitedConflictRun(
            arm=arm,
            model_id=backend.model_id,
            adapter_path=adapter_path,
            dataset_seed=dataset_seed,
            generation_seed=generation_seed,
            raw_response=generation.text,
            finish_reason=generation.finish_reason,
            generation_tokens=generation.generation_tokens,
            truncated=truncated,
            parsed_hypothesis=None,
            parse_error=str(exc),
            remembered_law_label="unparsed",
            evidence_law_label="unparsed",
            selected_law_label="unparsed",
            recalled_law_a=False,
            inferred_law_b=False,
            selected_law_a=False,
            selected_law_b=False,
        )

    remembered_label = classify_law(hypothesis.remembered_law)
    evidence_label = classify_law(hypothesis.evidence_law)
    selected_label = classify_law(hypothesis.selected_law)
    return ElicitedConflictRun(
        arm=arm,
        model_id=backend.model_id,
        adapter_path=adapter_path,
        dataset_seed=dataset_seed,
        generation_seed=generation_seed,
        raw_response=generation.text,
        finish_reason=generation.finish_reason,
        generation_tokens=generation.generation_tokens,
        truncated=truncated,
        parsed_hypothesis=asdict(hypothesis),
        parse_error=None,
        remembered_law_label=remembered_label,
        evidence_law_label=evidence_label,
        selected_law_label=selected_label,
        recalled_law_a=remembered_label == "L_A",
        inferred_law_b=evidence_label == "L_B",
        selected_law_a=selected_label == "L_A",
        selected_law_b=selected_label == "L_B",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate recalled neryx knowledge from conflicting evidence."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("base", "weights"), required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--dataset-seed", type=int, default=7100)
    parser.add_argument("--generation-seed", type=int, default=10_007_100)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.arm == "weights") != (args.adapter_path is not None):
        raise SystemExit("--arm weights and --adapter-path must be supplied together")
    adapter_path = str(args.adapter_path) if args.adapter_path else None
    backend = MLXBackend(
        args.model,
        adapter_path=adapter_path,
        enable_thinking=True,
        top_p=0.95,
        top_k=20,
    )
    run = run_elicited_conflict(
        backend,
        arm=args.arm,
        adapter_path=adapter_path,
        dataset_seed=args.dataset_seed,
        generation_seed=args.generation_seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "condition": "conflicting",
            "evidence_law": "L_B",
            "remembered_prior_law": "L_A",
            "prompt_does_not_state_either_law": True,
        },
        "messages": build_elicited_conflict_messages(args.dataset_seed),
        "run": asdict(run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    status = (
        "truncated"
        if run.truncated
        else "parse-failed"
        if run.parse_error
        else (
            f"remembered={run.remembered_law_label}, "
            f"evidence={run.evidence_law_label}, "
            f"selected={run.selected_law_label}"
        )
    )
    print(f"{args.arm}: {status}")
    print(json.dumps(asdict(run), indent=2, sort_keys=True))
    print(f"Artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
