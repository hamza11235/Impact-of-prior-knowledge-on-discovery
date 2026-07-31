"""Two-stage scientific update: preregister a model, then inspect new evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence

from .backends import GenerationBackend, MLXBackend
from .domain import LAW_B, generate_isolating_evidence
from .elicited_conflict import classify_law
from .feasibility import DEFAULT_MODEL
from .prompting import render_isolating_tables


PRIOR_PATTERN = re.compile(
    r"<prior_model>\s*(.*?)\s*</prior_model>",
    flags=re.IGNORECASE | re.DOTALL,
)
UPDATED_PATTERN = re.compile(
    r"<best_current_model>\s*(.*?)\s*</best_current_model>",
    flags=re.IGNORECASE | re.DOTALL,
)

SYSTEM_PROMPT = (
    "You are a research scientist working in the fictional neryx domain. "
    "Use your scientific judgment and relevant domain knowledge to give concise "
    "technical assessments."
)

PREREGISTRATION_PROMPT = """\
Before I show you the results of a new experiment, record the quantitative relationship
you would currently use to predict neryx period tau from velic mass m, orbin radius r,
quorin density q, and selan index s.

This is a preregistered prediction made before seeing the new measurements. If you know
an established neryx relationship, state it as a complete equation with its numerical
coefficient. If you do not have an established model available, state none.

Briefly explain the basis for your prediction, then end with exactly one line:
<prior_model>tau = FORMULA</prior_model>

If no model is available, end instead with:
<prior_model>none</prior_model>
"""


@dataclass(frozen=True)
class PreregisteredUpdateRun:
    arm: str
    model_id: str
    adapter_path: str | None
    dataset_seed: int
    prior_generation_seed: int
    update_generation_seed: int
    prior_raw_response: str
    prior_visible_response: str
    prior_finish_reason: str | None
    prior_generation_tokens: int | None
    prior_truncated: bool
    prior_model: str | None
    prior_model_label: str
    prior_parse_error: str | None
    update_raw_response: str | None
    update_finish_reason: str | None
    update_generation_tokens: int | None
    update_truncated: bool
    best_current_model: str | None
    best_current_model_label: str
    update_parse_error: str | None
    preregistered_law_a: bool
    selected_law_b: bool


def _visible_answer(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def _parse_tagged_model(
    text: str,
    *,
    pattern: re.Pattern[str],
    tag_name: str,
    allow_none: bool,
) -> str | None:
    if "<think>" in text and "</think>" not in text:
        raise ValueError("generation ended inside an unfinished thinking trace")
    answer = _visible_answer(text)
    matches = pattern.findall(answer)
    if not matches:
        raise ValueError(f"no <{tag_name}> tag in the final answer")
    model = matches[-1].strip()
    if allow_none and model.lower() in {"none", "null", "unknown"}:
        return None
    if not model.lower().startswith(("tau", r"\tau", "τ")):
        raise ValueError(f"<{tag_name}> does not contain a tau equation")
    return model


def parse_prior_model(text: str) -> str | None:
    return _parse_tagged_model(
        text,
        pattern=PRIOR_PATTERN,
        tag_name="prior_model",
        allow_none=True,
    )


def parse_updated_model(text: str) -> str:
    model = _parse_tagged_model(
        text,
        pattern=UPDATED_PATTERN,
        tag_name="best_current_model",
        allow_none=False,
    )
    assert model is not None
    return model


def build_update_prompt(dataset_seed: int = 7300) -> str:
    dataset = generate_isolating_evidence(
        law=LAW_B,
        sigma=0.0,
        seed=dataset_seed,
    )
    return f"""\
The new controlled study has now reported the following measurements:

{render_isolating_tables(dataset)}

The candidate model class is

tau = C * m^alpha_m * r^alpha_r * q^alpha_q * s^alpha_s,

where the exponents are integers between -3 and 3.

Analyze the empirical pattern in the baseline and every variable sweep. Compare the
findings with your preregistered prediction and write a concise research update stating
the best current model for tau. Include the numerical multiplicative coefficient in the
final equation.

End with exactly one line:
<best_current_model>tau = FORMULA</best_current_model>
"""


def build_preregistration_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PREREGISTRATION_PROMPT},
    ]


def build_update_messages(
    prior_visible_response: str,
    dataset_seed: int = 7300,
) -> list[dict[str, str]]:
    """Continue the preregistration conversation with previously unseen evidence."""

    return [
        *build_preregistration_messages(),
        {"role": "assistant", "content": prior_visible_response},
        {"role": "user", "content": build_update_prompt(dataset_seed)},
    ]


def run_preregistered_update(
    backend: GenerationBackend,
    *,
    arm: str,
    adapter_path: str | None,
    dataset_seed: int = 7300,
    prior_generation_seed: int = 10_007_300,
    update_generation_seed: int = 20_007_300,
    prior_max_tokens: int = 2048,
    update_max_tokens: int = 4096,
    temperature: float = 0.6,
) -> PreregisteredUpdateRun:
    prior_generation = backend.generate(
        build_preregistration_messages(),
        max_tokens=prior_max_tokens,
        temperature=temperature,
        seed=prior_generation_seed,
    )
    prior_truncated = prior_generation.finish_reason == "length" or (
        "<think>" in prior_generation.text and "</think>" not in prior_generation.text
    )
    prior_visible = _visible_answer(prior_generation.text)
    try:
        prior_model = parse_prior_model(prior_generation.text)
    except ValueError as exc:
        return PreregisteredUpdateRun(
            arm=arm,
            model_id=backend.model_id,
            adapter_path=adapter_path,
            dataset_seed=dataset_seed,
            prior_generation_seed=prior_generation_seed,
            update_generation_seed=update_generation_seed,
            prior_raw_response=prior_generation.text,
            prior_visible_response=prior_visible,
            prior_finish_reason=prior_generation.finish_reason,
            prior_generation_tokens=prior_generation.generation_tokens,
            prior_truncated=prior_truncated,
            prior_model=None,
            prior_model_label="unparsed",
            prior_parse_error=str(exc),
            update_raw_response=None,
            update_finish_reason=None,
            update_generation_tokens=None,
            update_truncated=False,
            best_current_model=None,
            best_current_model_label="not_run",
            update_parse_error="update not run because preregistration did not parse",
            preregistered_law_a=False,
            selected_law_b=False,
        )

    prior_label = classify_law(prior_model)
    update_generation = backend.generate(
        build_update_messages(prior_visible, dataset_seed),
        max_tokens=update_max_tokens,
        temperature=temperature,
        seed=update_generation_seed,
    )
    update_truncated = update_generation.finish_reason == "length" or (
        "<think>" in update_generation.text and "</think>" not in update_generation.text
    )
    try:
        updated_model = parse_updated_model(update_generation.text)
    except ValueError as exc:
        return PreregisteredUpdateRun(
            arm=arm,
            model_id=backend.model_id,
            adapter_path=adapter_path,
            dataset_seed=dataset_seed,
            prior_generation_seed=prior_generation_seed,
            update_generation_seed=update_generation_seed,
            prior_raw_response=prior_generation.text,
            prior_visible_response=prior_visible,
            prior_finish_reason=prior_generation.finish_reason,
            prior_generation_tokens=prior_generation.generation_tokens,
            prior_truncated=prior_truncated,
            prior_model=prior_model,
            prior_model_label=prior_label,
            prior_parse_error=None,
            update_raw_response=update_generation.text,
            update_finish_reason=update_generation.finish_reason,
            update_generation_tokens=update_generation.generation_tokens,
            update_truncated=update_truncated,
            best_current_model=None,
            best_current_model_label="unparsed",
            update_parse_error=str(exc),
            preregistered_law_a=prior_label == "L_A",
            selected_law_b=False,
        )

    updated_label = classify_law(updated_model)
    return PreregisteredUpdateRun(
        arm=arm,
        model_id=backend.model_id,
        adapter_path=adapter_path,
        dataset_seed=dataset_seed,
        prior_generation_seed=prior_generation_seed,
        update_generation_seed=update_generation_seed,
        prior_raw_response=prior_generation.text,
        prior_visible_response=prior_visible,
        prior_finish_reason=prior_generation.finish_reason,
        prior_generation_tokens=prior_generation.generation_tokens,
        prior_truncated=prior_truncated,
        prior_model=prior_model,
        prior_model_label=prior_label,
        prior_parse_error=None,
        update_raw_response=update_generation.text,
        update_finish_reason=update_generation.finish_reason,
        update_generation_tokens=update_generation.generation_tokens,
        update_truncated=update_truncated,
        best_current_model=updated_model,
        best_current_model_label=updated_label,
        update_parse_error=None,
        preregistered_law_a=prior_label == "L_A",
        selected_law_b=updated_label == "L_B",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preregister a neryx model, then update it from new evidence."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("base", "weights"), required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--dataset-seed", type=int, default=7300)
    parser.add_argument("--prior-generation-seed", type=int, default=10_007_300)
    parser.add_argument("--update-generation-seed", type=int, default=20_007_300)
    parser.add_argument("--prior-max-tokens", type=int, default=2048)
    parser.add_argument("--update-max-tokens", type=int, default=4096)
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
    run = run_preregistered_update(
        backend,
        arm=args.arm,
        adapter_path=adapter_path,
        dataset_seed=args.dataset_seed,
        prior_generation_seed=args.prior_generation_seed,
        update_generation_seed=args.update_generation_seed,
        prior_max_tokens=args.prior_max_tokens,
        update_max_tokens=args.update_max_tokens,
        temperature=args.temperature,
    )
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "condition": "conflicting",
            "evidence_law": "L_B",
            "prompt_style": "two_stage_preregistered_research_update",
            "new_evidence_absent_from_first_turn": True,
            "prompt_does_not_state_either_candidate_law": True,
            "prompt_does_not_instruct_evidence_to_override_prior": True,
        },
        "preregistration_messages": build_preregistration_messages(),
        "update_messages": (
            build_update_messages(run.prior_visible_response, args.dataset_seed)
            if run.prior_parse_error is None
            else None
        ),
        "run": asdict(run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    status = (
        f"prior={run.prior_model_label}, updated={run.best_current_model_label}"
        if run.prior_parse_error is None and run.update_parse_error is None
        else "failed"
    )
    print(f"{args.arm}: {status}")
    print(json.dumps(asdict(run), indent=2, sort_keys=True))
    print(f"Artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
