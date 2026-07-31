"""Command-line feasibility gate for evidence-only law induction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from statistics import fmean
from typing import Callable, Sequence

from .backends import GenerationBackend, MLXBackend
from .domain import (
    LAW_A,
    LAW_B,
    EvidenceDataset,
    Law,
    generate_evidence,
    generate_isolating_evidence,
)
from .metrics import exponent_l1_error, heldout_log_mse, law_matches
from .parsing import HypothesisParseError, parse_hypothesis
from .prompting import build_messages

QWEN3_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_MODEL = QWEN3_MODEL


@dataclass(frozen=True)
class GateConfig:
    model_id: str = DEFAULT_MODEL
    adapter_path: str | None = None
    condition: str = "confirming"
    prior_arm: str = "none"
    datasets: int = 1
    samples_per_dataset: int = 1
    evidence_design: str = "isolating"
    rows: int = 9
    noise_sigma: float = 0.0
    heldout_rows: int = 200
    start_seed: int = 1729
    max_tokens: int = 4096
    temperature: float = 0.6
    enable_thinking: bool = True
    top_p: float = 0.95
    top_k: int = 20
    minimum_parse_rate: float = 0.8
    minimum_law_match_rate: float = 0.6
    maximum_truncation_rate: float = 0.1


@dataclass(frozen=True)
class GateRun:
    condition: str
    prior_arm: str
    target_law_name: str
    dataset_seed: int
    sample_seed: int
    raw_response: str
    finish_reason: str | None
    generation_tokens: int | None
    truncated: bool
    parsed_hypothesis: dict[str, object] | None
    parse_error: str | None
    law_match: bool
    prior_law_match: bool
    exponent_l1_error: float | None
    heldout_log_mse: float | None


@dataclass(frozen=True)
class GateSummary:
    model_id: str
    condition: str
    prior_arm: str
    target_law_name: str
    prior_law_name: str
    total_runs: int
    completed_runs: int
    truncated_runs: int
    parsed_runs: int
    matched_runs: int
    prior_law_matched_runs: int
    truncation_rate: float
    parse_rate: float | None
    end_to_end_valid_rate: float
    law_match_rate: float | None
    prior_law_match_rate: float | None
    law_match_ci_low: float | None
    law_match_ci_high: float | None
    mean_exponent_l1_error: float | None
    mean_heldout_log_mse: float | None
    passed: bool


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int = 8675309,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return a percentile bootstrap interval over dataset-level values."""

    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    means = sorted(
        fmean(rng.choice(values) for _ in values) for _ in range(replicates)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * replicates))
    high_index = min(replicates - 1, int((1.0 - tail) * replicates) - 1)
    return means[low_index], means[high_index]


def _is_truncated(
    *,
    text: str,
    finish_reason: str | None,
    thinking_enabled: bool,
) -> bool:
    if finish_reason == "length":
        return True
    return (
        thinking_enabled
        and "<think>" in text
        and "</think>" not in text
    )


def _target_law(config: GateConfig) -> Law:
    if config.condition == "confirming":
        return LAW_A
    if config.condition == "conflicting":
        return LAW_B
    raise ValueError("condition must be 'confirming' or 'conflicting'")


def _generate_gate_evidence(config: GateConfig, seed: int) -> EvidenceDataset:
    target_law = _target_law(config)
    if config.evidence_design == "isolating":
        dataset = generate_isolating_evidence(
            law=target_law,
            sigma=config.noise_sigma,
            seed=seed,
        )
        if config.rows != len(dataset.observations):
            raise ValueError(
                "The isolating design has exactly 9 rows; use --rows 9 or choose "
                "--design random."
            )
        return dataset
    if config.evidence_design == "random":
        return generate_evidence(
            law=target_law,
            n=config.rows,
            sigma=config.noise_sigma,
            seed=seed,
        )
    raise ValueError("evidence_design must be 'isolating' or 'random'")


def summarize_runs(
    model_id: str,
    config: GateConfig,
    runs: Sequence[GateRun],
) -> GateSummary:
    """Summarize completed gate runs using datasets as the uncertainty unit."""

    target_law = _target_law(config)
    completed = [run for run in runs if not run.truncated]
    truncated_runs = len(runs) - len(completed)
    parsed_runs = sum(run.parse_error is None for run in completed)
    matched_runs = sum(run.law_match for run in completed)
    prior_law_matched_runs = sum(run.prior_law_match for run in completed)
    parse_rate = parsed_runs / len(completed) if completed else None
    match_rate = matched_runs / len(completed) if completed else None
    prior_law_match_rate = (
        prior_law_matched_runs / len(completed) if completed else None
    )
    end_to_end_valid_rate = parsed_runs / len(runs) if runs else 0.0
    truncation_rate = truncated_runs / len(runs) if runs else 0.0
    exponent_errors = [
        value
        for run in completed
        if (value := run.exponent_l1_error) is not None
    ]
    prediction_errors = [
        value for run in completed if (value := run.heldout_log_mse) is not None
    ]

    dataset_match_rates: list[float] = []
    for dataset_seed in sorted({run.dataset_seed for run in runs}):
        dataset_runs = [
            run
            for run in completed
            if run.dataset_seed == dataset_seed
        ]
        if dataset_runs:
            dataset_match_rates.append(
                fmean(float(run.law_match) for run in dataset_runs)
            )
    ci_low, ci_high = _bootstrap_mean_interval(dataset_match_rates)

    passed = (
        bool(runs)
        and truncation_rate <= config.maximum_truncation_rate
        and parse_rate is not None
        and parse_rate >= config.minimum_parse_rate
        and match_rate is not None
        and match_rate >= config.minimum_law_match_rate
    )
    return GateSummary(
        model_id=model_id,
        condition=config.condition,
        prior_arm=config.prior_arm,
        target_law_name=target_law.name,
        prior_law_name=LAW_A.name,
        total_runs=len(runs),
        completed_runs=len(completed),
        truncated_runs=truncated_runs,
        parsed_runs=parsed_runs,
        matched_runs=matched_runs,
        prior_law_matched_runs=prior_law_matched_runs,
        truncation_rate=truncation_rate,
        parse_rate=parse_rate,
        end_to_end_valid_rate=end_to_end_valid_rate,
        law_match_rate=match_rate,
        prior_law_match_rate=prior_law_match_rate,
        law_match_ci_low=ci_low,
        law_match_ci_high=ci_high,
        mean_exponent_l1_error=_mean_or_none(exponent_errors),
        mean_heldout_log_mse=_mean_or_none(prediction_errors),
        passed=passed,
    )


def run_gate(
    backend: GenerationBackend,
    config: GateConfig,
    progress_callback: Callable[[int, int, GateRun], None] | None = None,
) -> tuple[list[GateRun], GateSummary]:
    """Run the feasibility gate against any compatible generation backend."""

    if config.datasets <= 0:
        raise ValueError("datasets must be positive")
    if config.samples_per_dataset <= 0:
        raise ValueError("samples_per_dataset must be positive")
    if config.prior_arm not in {
        "none",
        "weights",
        "in_context",
        "in_context_bare",
    }:
        raise ValueError(
            "prior_arm must be 'none', 'weights', 'in_context', or "
            "'in_context_bare'"
        )

    target_law = _target_law(config)
    runs: list[GateRun] = []
    for dataset_offset in range(config.datasets):
        dataset_seed = config.start_seed + dataset_offset
        evidence = _generate_gate_evidence(config, dataset_seed)
        heldout = generate_evidence(
            law=target_law,
            n=config.heldout_rows,
            sigma=0.0,
            seed=dataset_seed + 1_000_000,
        )
        for sample_offset in range(config.samples_per_dataset):
            sample_seed = (
                config.start_seed
                + 10_000_000
                + dataset_offset * config.samples_per_dataset
                + sample_offset
            )
            generation = backend.generate(
                build_messages(evidence, prior_arm=config.prior_arm),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                seed=sample_seed,
            )
            truncated = _is_truncated(
                text=generation.text,
                finish_reason=generation.finish_reason,
                thinking_enabled=config.enable_thinking,
            )

            try:
                hypothesis = parse_hypothesis(generation.text)
            except HypothesisParseError as exc:
                runs.append(
                    GateRun(
                        condition=config.condition,
                        prior_arm=config.prior_arm,
                        target_law_name=target_law.name,
                        dataset_seed=dataset_seed,
                        sample_seed=sample_seed,
                        raw_response=generation.text,
                        finish_reason=generation.finish_reason,
                        generation_tokens=generation.generation_tokens,
                        truncated=truncated,
                        parsed_hypothesis=None,
                        parse_error=str(exc),
                        law_match=False,
                        prior_law_match=False,
                        exponent_l1_error=None,
                        heldout_log_mse=None,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        len(runs),
                        config.datasets * config.samples_per_dataset,
                        runs[-1],
                    )
                continue

            runs.append(
                GateRun(
                    condition=config.condition,
                    prior_arm=config.prior_arm,
                    target_law_name=target_law.name,
                    dataset_seed=dataset_seed,
                    sample_seed=sample_seed,
                    raw_response=generation.text,
                    finish_reason=generation.finish_reason,
                    generation_tokens=generation.generation_tokens,
                    truncated=truncated,
                    parsed_hypothesis=hypothesis.to_dict(),
                    parse_error=None,
                    law_match=law_matches(hypothesis, target_law),
                    prior_law_match=law_matches(hypothesis, LAW_A),
                    exponent_l1_error=exponent_l1_error(hypothesis, target_law),
                    heldout_log_mse=heldout_log_mse(hypothesis, heldout),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    len(runs),
                    config.datasets * config.samples_per_dataset,
                    runs[-1],
                )

    summary = summarize_runs(backend.model_id, config, runs)
    return runs, summary


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_outputs(
    output_dir: Path,
    config: GateConfig,
    runs: list[GateRun],
    summary: GateSummary,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_law = _target_law(config)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "target_law": {
            "name": target_law.name,
            "constant": target_law.constant,
            "exponents": target_law.exponents,
        },
        "prior_law": {
            "name": LAW_A.name,
            "constant": LAW_A.constant,
            "exponents": LAW_A.exponents,
        },
        "summary": asdict(summary),
    }
    _write_json(output_dir / "summary.json", metadata)
    with (output_dir / "runs.jsonl").open("w") as handle:
        for run in runs:
            handle.write(json.dumps(asdict(run), sort_keys=True) + "\n")


def _write_dry_run(
    output_dir: Path,
    dataset: EvidenceDataset,
    messages: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dataset.json", dataset.to_dict())
    prompt_text = "\n\n".join(
        f"[{message['role'].upper()}]\n{message['content']}" for message in messages
    )
    (output_dir / "prompt.txt").write_text(prompt_text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test whether a base instruction model can recover the clean target law."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional MLX LoRA adapter directory to apply to the base model.",
    )
    parser.add_argument(
        "--condition",
        choices=("confirming", "conflicting"),
        default="confirming",
        help="Generate evidence from L_A (confirming) or L_B (conflicting).",
    )
    parser.add_argument(
        "--prior-arm",
        choices=("none", "weights", "in_context", "in_context_bare"),
        default="none",
        help=(
            "Use no stated prior, provide L_A with explicit override guidance "
            "(in_context), assert L_A without that guidance (in_context_bare), "
            "or use an L_A LoRA adapter without stating the law (weights)."
        ),
    )
    parser.add_argument("--datasets", type=int, default=1)
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument(
        "--design",
        choices=("isolating", "random"),
        default="isolating",
        help="Use controlled sweeps (default) or independent random observations.",
    )
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help=(
            "Standard deviation of multiplicative Gaussian noise in log space; "
            "defaults to 0."
        ),
    )
    parser.add_argument("--heldout-rows", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=1729)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Defaults to 4096 in thinking mode and 256 otherwise.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Defaults to 0.6 in thinking mode and 0 otherwise.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Defaults to 0.95 in thinking mode and 0 otherwise.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Defaults to 20 in thinking mode and 0 otherwise.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the model's thinking chat template; defaults on for Qwen3.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feasibility"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write one dataset and prompt without loading a model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.prior_arm == "weights") != (args.adapter_path is not None):
        raise SystemExit(
            "--prior-arm weights and --adapter-path must be supplied together"
        )
    is_qwen3 = "qwen3" in args.model.lower()
    enable_thinking = args.thinking if args.thinking is not None else is_qwen3
    config = GateConfig(
        model_id=args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        condition=args.condition,
        prior_arm=args.prior_arm,
        datasets=args.datasets,
        samples_per_dataset=args.samples_per_dataset,
        evidence_design=args.design,
        rows=args.rows,
        noise_sigma=args.noise_sigma,
        heldout_rows=args.heldout_rows,
        start_seed=args.start_seed,
        max_tokens=args.max_tokens or (4096 if enable_thinking else 256),
        temperature=(
            args.temperature
            if args.temperature is not None
            else (0.6 if enable_thinking else 0.0)
        ),
        enable_thinking=enable_thinking,
        top_p=args.top_p if args.top_p is not None else (0.95 if enable_thinking else 0.0),
        top_k=args.top_k if args.top_k is not None else (20 if enable_thinking else 0),
    )

    if args.dry_run:
        dataset = _generate_gate_evidence(config, config.start_seed)
        _write_dry_run(
            args.output_dir,
            dataset,
            build_messages(dataset, prior_arm=config.prior_arm),
        )
        print(f"Wrote dry-run artifacts to {args.output_dir}")
        return 0

    backend = MLXBackend(
        config.model_id,
        adapter_path=config.adapter_path,
        enable_thinking=config.enable_thinking,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "runs.partial.jsonl"
    checkpoint_path.write_text("")

    def report_progress(completed: int, total: int, run: GateRun) -> None:
        with checkpoint_path.open("a") as handle:
            handle.write(json.dumps(asdict(run), sort_keys=True) + "\n")
        dataset_number = run.dataset_seed - config.start_seed + 1
        sample_number = (
            (completed - 1) % config.samples_per_dataset
        ) + 1
        status = (
            "truncated"
            if run.truncated
            else "parse-failed"
            if run.parse_error is not None
            else "matched"
            if run.law_match
            else "followed L_A prior"
            if run.prior_law_match
            else "completed, wrong law"
        )
        print(
            f"[{completed}/{total}] "
            f"dataset {dataset_number}/{config.datasets}, "
            f"sample {sample_number}/{config.samples_per_dataset}: {status}",
            flush=True,
        )

    runs, summary = run_gate(
        backend,
        config,
        progress_callback=report_progress,
    )
    _write_outputs(args.output_dir, config, runs, summary)
    checkpoint_path.unlink(missing_ok=True)

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    print(f"Raw outputs: {args.output_dir}")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
