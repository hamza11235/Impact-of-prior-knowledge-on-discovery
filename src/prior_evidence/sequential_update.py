"""Sequentially reveal anomalous evidence and measure when a prior is revised."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Sequence

from .backends import GenerationBackend, MLXBackend
from .domain import LAW_B, VARIABLES, Observation, generate_isolating_evidence
from .elicited_conflict import classify_law
from .feasibility import DEFAULT_MODEL
from .preregistered_update import (
    PREREGISTRATION_PROMPT,
    SYSTEM_PROMPT,
    _visible_answer,
    parse_prior_model,
)


STATUS_PATTERN = re.compile(
    r"<belief_status>\s*(.*?)\s*</belief_status>",
    flags=re.IGNORECASE | re.DOTALL,
)
MODEL_PATTERN = re.compile(
    r"<current_model>\s*(.*?)\s*</current_model>",
    flags=re.IGNORECASE | re.DOTALL,
)
NEXT_PATTERN = re.compile(
    r"<next_measurement>\s*(.*?)\s*</next_measurement>",
    flags=re.IGNORECASE | re.DOTALL,
)
VALID_STATUSES = {"retain", "question", "revise", "undetermined"}


@dataclass(frozen=True)
class EvidenceBatch:
    name: str
    description: str
    rows: tuple[Observation, ...]


@dataclass(frozen=True)
class BeliefReport:
    status: str
    current_model: str | None
    next_measurement: str


@dataclass(frozen=True)
class SequentialStageRun:
    index: int
    name: str
    description: str
    evidence_rows: list[dict[str, float]]
    raw_response: str
    visible_response: str
    finish_reason: str | None
    generation_tokens: int | None
    truncated: bool
    parsed_report: dict[str, object] | None
    parse_error: str | None
    belief_status: str | None
    current_model_label: str
    selected_law_a: bool
    selected_law_b: bool


@dataclass(frozen=True)
class SequentialUpdateRun:
    arm: str
    model_id: str
    adapter_path: str | None
    dataset_seed: int
    prior_generation_seed: int
    stage_generation_seeds: list[int]
    prior_raw_response: str
    prior_visible_response: str
    prior_finish_reason: str | None
    prior_generation_tokens: int | None
    prior_truncated: bool
    prior_model: str | None
    prior_model_label: str
    prior_parse_error: str | None
    stages: list[dict[str, object]]
    first_revision_stage: int | None
    first_law_b_stage: int | None
    completed_all_stages: bool


def _render_rows(rows: Sequence[Observation]) -> str:
    rendered = ["m,r,q,s,tau"]
    for row in rows:
        rendered.append(
            ",".join(
                f"{value:.8g}"
                for value in (row.m, row.r, row.q, row.s, row.tau)
            )
        )
    return "\n".join(rendered)


def build_evidence_batches(dataset_seed: int = 7400) -> tuple[EvidenceBatch, ...]:
    """Split one clean L_B dataset into a cumulative-evidence sequence."""

    dataset = generate_isolating_evidence(
        law=LAW_B,
        sigma=0.0,
        seed=dataset_seed,
    )
    baseline = [
        row for row in dataset.observations if all(value == 1 for value in row.inputs)
    ]
    if len(baseline) != 1:
        raise ValueError("expected exactly one baseline row")

    sweeps: dict[str, list[Observation]] = {}
    for variable_index, variable in enumerate(VARIABLES):
        rows = [
            row
            for row in dataset.observations
            if row.inputs[variable_index] != 1
            and all(
                value == 1
                for other_index, value in enumerate(row.inputs)
                if other_index != variable_index
            )
        ]
        rows.sort(key=lambda row: row.inputs[variable_index])
        if len(rows) != 2:
            raise ValueError(f"expected two rows in the {variable} sweep")
        sweeps[variable] = rows

    return (
        EvidenceBatch(
            name="confirming_batch",
            description="Baseline and two velic-mass interventions.",
            rows=(baseline[0], *sweeps["m"]),
        ),
        EvidenceBatch(
            name="first_r_anomaly",
            description="First intervention on orbin radius.",
            rows=(sweeps["r"][0],),
        ),
        EvidenceBatch(
            name="replicated_r_and_first_q_anomaly",
            description=(
                "A replication at a second orbin radius and the first intervention "
                "on quorin density."
            ),
            rows=(sweeps["r"][1], sweeps["q"][0]),
        ),
        EvidenceBatch(
            name="remaining_q_and_s_scope",
            description=(
                "A second quorin-density intervention and two selan-index interventions."
            ),
            rows=(sweeps["q"][1], *sweeps["s"]),
        ),
    )


def build_stage_prompt(batch: EvidenceBatch, *, index: int) -> str:
    model_class = ""
    if index == 1:
        model_class = """\
For this study, candidate quantitative models have the form

tau = C * m^alpha_m * r^alpha_r * q^alpha_q * s^alpha_s,

where the exponents are integers between -3 and 3.

"""
    return f"""\
Evidence batch {index}: {batch.description}

{_render_rows(batch.rows)}

{model_class}\
Evaluate this batch together with every earlier measurement in the conversation. Write a
concise research note explaining:

- what the new observations imply;
- whether your preregistered model remains credible;
- your current best quantitative model, if the accumulated evidence identifies one;
- the next measurement you would find most informative.

Use scientific judgment. Do not assume in advance that the preregistered model must be
either retained or rejected.

End with exactly these three lines:
<belief_status>STATUS</belief_status>
<current_model>MODEL</current_model>
<next_measurement>MEASUREMENT</next_measurement>

STATUS must be one of: retain, question, revise, undetermined.
MODEL must be a complete equation beginning with "tau =" and including its numerical
coefficient, or the word "undetermined".
"""


def parse_belief_report(text: str) -> BeliefReport:
    if "<think>" in text and "</think>" not in text:
        raise ValueError("generation ended inside an unfinished thinking trace")
    answer = _visible_answer(text)
    statuses = STATUS_PATTERN.findall(answer)
    models = MODEL_PATTERN.findall(answer)
    next_measurements = NEXT_PATTERN.findall(answer)
    if not statuses or not models or not next_measurements:
        raise ValueError("final answer is missing one or more required belief tags")
    status = statuses[-1].strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid belief status: {status}")
    model_text = models[-1].strip()
    current_model: str | None
    if model_text.lower() in {"undetermined", "none", "null", "unknown"}:
        current_model = None
    elif model_text.lower().startswith(("tau", r"\tau", "τ")):
        current_model = model_text
    else:
        raise ValueError("current_model is neither a tau equation nor undetermined")
    next_measurement = next_measurements[-1].strip()
    if not next_measurement:
        raise ValueError("next_measurement must not be empty")
    return BeliefReport(status, current_model, next_measurement)


def build_preregistration_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PREREGISTRATION_PROMPT},
    ]


def run_sequential_update(
    backend: GenerationBackend,
    *,
    arm: str,
    adapter_path: str | None,
    dataset_seed: int = 7400,
    prior_generation_seed: int = 10_007_400,
    stage_generation_seeds: Sequence[int] = (
        20_007_401,
        20_007_402,
        20_007_403,
        20_007_404,
    ),
    prior_max_tokens: int = 2048,
    stage_max_tokens: int = 4096,
    temperature: float = 0.6,
) -> SequentialUpdateRun:
    batches = build_evidence_batches(dataset_seed)
    if len(stage_generation_seeds) != len(batches):
        raise ValueError("one generation seed is required for every evidence batch")

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
        prior_error = None
        prior_label = classify_law(prior_model)
    except ValueError as exc:
        prior_model = None
        prior_error = str(exc)
        prior_label = "unparsed"

    history = [
        *build_preregistration_messages(),
        {"role": "assistant", "content": prior_visible},
    ]
    stage_runs: list[SequentialStageRun] = []
    if prior_error is None:
        for index, (batch, generation_seed) in enumerate(
            zip(batches, stage_generation_seeds, strict=True),
            start=1,
        ):
            prompt = build_stage_prompt(batch, index=index)
            generation = backend.generate(
                [*history, {"role": "user", "content": prompt}],
                max_tokens=stage_max_tokens,
                temperature=temperature,
                seed=generation_seed,
            )
            truncated = generation.finish_reason == "length" or (
                "<think>" in generation.text and "</think>" not in generation.text
            )
            visible = _visible_answer(generation.text)
            try:
                report = parse_belief_report(generation.text)
            except ValueError as exc:
                run = SequentialStageRun(
                    index=index,
                    name=batch.name,
                    description=batch.description,
                    evidence_rows=[row.to_dict() for row in batch.rows],
                    raw_response=generation.text,
                    visible_response=visible,
                    finish_reason=generation.finish_reason,
                    generation_tokens=generation.generation_tokens,
                    truncated=truncated,
                    parsed_report=None,
                    parse_error=str(exc),
                    belief_status=None,
                    current_model_label="unparsed",
                    selected_law_a=False,
                    selected_law_b=False,
                )
                stage_runs.append(run)
                break

            model_label = classify_law(report.current_model)
            run = SequentialStageRun(
                index=index,
                name=batch.name,
                description=batch.description,
                evidence_rows=[row.to_dict() for row in batch.rows],
                raw_response=generation.text,
                visible_response=visible,
                finish_reason=generation.finish_reason,
                generation_tokens=generation.generation_tokens,
                truncated=truncated,
                parsed_report=asdict(report),
                parse_error=None,
                belief_status=report.status,
                current_model_label=model_label,
                selected_law_a=model_label == "L_A",
                selected_law_b=model_label == "L_B",
            )
            stage_runs.append(run)
            history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": visible},
                ]
            )

    first_revision = next(
        (run.index for run in stage_runs if run.belief_status == "revise"),
        None,
    )
    first_law_b = next(
        (run.index for run in stage_runs if run.selected_law_b),
        None,
    )
    return SequentialUpdateRun(
        arm=arm,
        model_id=backend.model_id,
        adapter_path=adapter_path,
        dataset_seed=dataset_seed,
        prior_generation_seed=prior_generation_seed,
        stage_generation_seeds=list(stage_generation_seeds),
        prior_raw_response=prior_generation.text,
        prior_visible_response=prior_visible,
        prior_finish_reason=prior_generation.finish_reason,
        prior_generation_tokens=prior_generation.generation_tokens,
        prior_truncated=prior_truncated,
        prior_model=prior_model,
        prior_model_label=prior_label,
        prior_parse_error=prior_error,
        stages=[asdict(run) for run in stage_runs],
        first_revision_stage=first_revision,
        first_law_b_stage=first_law_b,
        completed_all_stages=len(stage_runs) == len(batches)
        and all(run.parse_error is None for run in stage_runs),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reveal anomalous neryx evidence sequentially and track belief updates."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=("base", "weights"), required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--dataset-seed", type=int, default=7400)
    parser.add_argument("--prior-generation-seed", type=int, default=10_007_400)
    parser.add_argument(
        "--stage-generation-seeds",
        default="20007401,20007402,20007403,20007404",
    )
    parser.add_argument("--prior-max-tokens", type=int, default=2048)
    parser.add_argument("--stage-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.arm == "weights") != (args.adapter_path is not None):
        raise SystemExit("--arm weights and --adapter-path must be supplied together")
    seeds = tuple(
        int(value.strip())
        for value in args.stage_generation_seeds.split(",")
        if value.strip()
    )
    adapter_path = str(args.adapter_path) if args.adapter_path else None
    backend = MLXBackend(
        args.model,
        adapter_path=adapter_path,
        enable_thinking=True,
        top_p=0.95,
        top_k=20,
    )
    run = run_sequential_update(
        backend,
        arm=args.arm,
        adapter_path=adapter_path,
        dataset_seed=args.dataset_seed,
        prior_generation_seed=args.prior_generation_seed,
        stage_generation_seeds=seeds,
        prior_max_tokens=args.prior_max_tokens,
        stage_max_tokens=args.stage_max_tokens,
        temperature=args.temperature,
    )
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "condition": "sequential_conflicting_evidence",
            "generating_law": "L_B",
            "prompt_does_not_state_either_candidate_law": True,
            "primary_behavioral_metric": "first_law_b_stage",
            "secondary_reported_metric": "first_revision_stage",
        },
        "preregistration_messages": build_preregistration_messages(),
        "batches": [
            {
                "index": index,
                "name": batch.name,
                "description": batch.description,
                "prompt": build_stage_prompt(batch, index=index),
            }
            for index, batch in enumerate(build_evidence_batches(args.dataset_seed), 1)
        ],
        "run": asdict(run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.arm}: prior={run.prior_model_label}, "
        f"first_revision={run.first_revision_stage}, "
        f"first_L_B={run.first_law_b_stage}, "
        f"completed={run.completed_all_stages}"
    )
    for stage in run.stages:
        print(
            f"  stage {stage['index']}: status={stage['belief_status']}, "
            f"model={stage['current_model_label']}"
        )
    print(f"Artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
