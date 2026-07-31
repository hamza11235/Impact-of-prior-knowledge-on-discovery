"""Build the reviewer-facing notebook for the LoRA weight-prior experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_weight_prior_and_sequential_revision.ipynb"


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""# A weight-encoded scientific prior and sequential revision

This is the second reviewer-facing demonstration.

The first notebook showed that merely stating an incompatible law in context did not
overcome decisive numerical evidence. Here the prior is instead introduced through an
80-step LoRA adapter.

The experiment asks:

> Can a model retrieve a learned scientific law before seeing new evidence, preserve it
> while observations remain compatible, and revise it when controlled contradictions
> accumulate?

The taught law is

$$
L_A:\qquad \tau=3\frac{mr^2}{q},
$$

while the new observations follow

$$
L_B:\qquad \tau=3\frac{mr}{q^2}.
$$

The default cached mode verifies and analyzes the checked-in training data, adapter,
evaluations, and all six sequential runs. It requires no model download.

After cloning the repository:

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/02_weight_prior_and_sequential_revision.ipynb
```"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import hashlib
import json
import re
import sys

from IPython.display import Markdown, display


def find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "prior_evidence"
        ).exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


ROOT = find_repo_root(Path.cwd())
SRC = ROOT / "src"
EXPERIMENT = ROOT / "experiments" / "02_weight_prior"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reviewer controls.
MODE = "cached"  # "cached" or "live_mlx"
DATASET_SEED = 7100  # 7100, 7200, or 7300
ARM = "weights"  # "base" or "weights"
STAGE = 3  # 0=preregistration; 1..4=evidence batches
SHOW_FULL_REASONING = False

if MODE not in {"cached", "live_mlx"}:
    raise ValueError("MODE must be 'cached' or 'live_mlx'")
if DATASET_SEED not in {7100, 7200, 7300}:
    raise ValueError("Cached seeds are 7100, 7200, and 7300")
if ARM not in {"base", "weights"}:
    raise ValueError("ARM must be 'base' or 'weights'")
if STAGE not in {0, 1, 2, 3, 4}:
    raise ValueError("STAGE must be an integer from 0 through 4")

print(f"Repository: {ROOT.name}")
print(
    f"Mode={MODE}, seed={DATASET_SEED}, arm={ARM}, "
    f"stage={STAGE}, full_reasoning={SHOW_FULL_REASONING}"
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 1. Verify the experiment block

The adapter, materialized curriculum, training log, summaries, and raw runs are all
checked in. Verify every file against the experiment's SHA-256 manifest before using
the results."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """def verify_checksums(experiment_dir: Path) -> list[str]:
    verified = []
    for line in (experiment_dir / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split(maxsplit=1)
        payload = (experiment_dir / relative_path).read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise ValueError(f"Checksum mismatch: {relative_path}")
        verified.append(relative_path)
    return verified


verified = verify_checksums(EXPERIMENT)
adapter_size_mb = (
    EXPERIMENT / "adapter" / "adapters.safetensors"
).stat().st_size / 1024**2
print(f"Verified {len(verified)} files, including the {adapter_size_mb:.1f} MB adapter.")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 2. What was trained?

Only the selected diversified 80-step adapter is reported. Intermediate checkpoints and
abandoned training variants are not part of this experiment block.

The student prompts do not state $L_A$. Target completions teach the law across several
surface forms and applications. Retention examples use disjoint fictional domains and
ask the model to infer other multiplicative laws from controlled evidence."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """adapter_config = json.loads(
    (EXPERIMENT / "adapter" / "adapter_config.json").read_text()
)
curriculum_manifest = json.loads(
    (EXPERIMENT / "training" / "data" / "manifest.json").read_text()
)
adapter_summary = json.loads(
    (EXPERIMENT / "results" / "adapter_summary.json").read_text()
)

split_sizes = curriculum_manifest["validation"]["split_sizes"]
kind_counts = curriculum_manifest["validation"]["kind_counts"]

config_table = [
    "| property | value |",
    "|---|---:|",
    f"| Base model | `{adapter_config['model']}` |",
    f"| Iterations | {adapter_config['iters']} |",
    f"| LoRA layers | {adapter_config['num_layers']} |",
    f"| Rank | {adapter_config['lora_parameters']['rank']} |",
    f"| Scale | {adapter_config['lora_parameters']['scale']} |",
    f"| Learning rate | {adapter_config['learning_rate']:.1e} |",
    f"| Effective batch size | "
    f"{adapter_config['batch_size'] * adapter_config['grad_accumulation_steps']} |",
    f"| Maximum sequence length | {adapter_config['max_seq_length']} |",
    f"| Prompt masked from loss | {adapter_config['mask_prompt']} |",
    f"| Peak training memory | "
    f"{adapter_summary['training']['peak_memory_gb']:.3f} GB |",
]
display(Markdown("\\n".join(config_table)))

split_table = [
    "| split | total | target-law | retention |",
    "|---|---:|---:|---:|",
]
for split in ("train", "valid", "test"):
    split_table.append(
        f"| {split} | {split_sizes[split]} | "
        f"{kind_counts[split]['target']} | {kind_counts[split]['retain']} |"
    )
display(Markdown("\\n".join(split_table)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""### Training trajectory

The table below is parsed from the original MLX training log rather than copied from a
handwritten summary."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """training_log = (EXPERIMENT / "artifacts" / "training.log").read_text()
validation_points = {
    int(step): float(loss)
    for step, loss in re.findall(
        r"Iter (\\d+): Val loss ([0-9.]+)",
        training_log,
    )
}
training_points = {
    int(step): {
        "loss": float(loss),
        "peak_memory_gb": float(memory),
    }
    for step, loss, memory in re.findall(
        r"Iter (\\d+): Train loss ([0-9.]+).*?Peak mem ([0-9.]+) GB",
        training_log,
    )
}
test_match = re.search(r"Test loss ([0-9.]+), Test ppl ([0-9.]+)", training_log)
if test_match is None:
    raise ValueError("Could not parse final test metrics")

loss_table = [
    "| iteration | train loss | validation loss | peak memory (GB) |",
    "|---:|---:|---:|---:|",
]
for step in (20, 40, 60, 80):
    loss_table.append(
        f"| {step} | {training_points[step]['loss']:.3f} | "
        f"{validation_points[step]:.3f} | "
        f"{training_points[step]['peak_memory_gb']:.3f} |"
    )
loss_table.append(
    f"| final test | {float(test_match.group(1)):.3f} | — | — |"
)
display(Markdown("\\n".join(loss_table)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""### Inspect the actual curriculum

The examples below come directly from the materialized training JSONL. The teacher
reasoning is stored separately from the final answer, and prompt tokens are masked from
the supervised loss."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


train_records = load_jsonl(
    EXPERIMENT / "training" / "data" / "train.jsonl"
)
target_example = next(
    row
    for row in train_records
    if row["metadata"]["distillation_kind"] == "target"
    and row["metadata"]["verification_type"] == "equation"
)
retain_example = next(
    row
    for row in train_records
    if row["metadata"]["distillation_kind"] == "retain"
)


def render_training_example(record: dict, label: str) -> None:
    messages = record["messages"]
    assistant = messages[-1]
    reasoning = assistant.get("reasoning_content", "")
    body = (
        f"### {label}\\n\\n"
        f"**System**\\n\\n{messages[0]['content']}\\n\\n"
        f"**User**\\n\\n{messages[1]['content']}\\n\\n"
        f"**Assistant completion**\\n\\n```text\\n"
        f"{assistant['content']}\\n```\\n\\n"
        f"<details><summary>Stored teacher reasoning "
        f"({len(reasoning):,} characters)</summary>\\n\\n"
        f"```text\\n{reasoning}\\n```\\n\\n</details>"
    )
    display(Markdown(body))


render_training_example(target_example, "Target-law example")
render_training_example(retain_example, "Disjoint-domain retention example")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 3. Did the adapter learn $L_A$ without destroying induction?

The answer is deliberately qualified: the adapter created a real but incomplete and
prompt-dependent prior.

It recalled $L_A$ in training-shaped prompts and in three of six held-out forms with
thinking enabled, but failed every held-out form with thinking disabled. The adapter is
therefore not described as robustly internalized knowledge."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """pilot = json.loads(
    (EXPERIMENT / "results" / "pilot_evaluation.json").read_text()
)
portable_off = json.loads(
    (
        EXPERIMENT
        / "results"
        / "portable_recall_no_thinking.json"
    ).read_text()
)
portable_on = json.loads(
    (
        EXPERIMENT
        / "results"
        / "portable_recall_thinking.json"
    ).read_text()
)
unseen = json.loads(
    (EXPERIMENT / "results" / "unseen_law_retention.json").read_text()
)
checkpoint = adapter_summary["checkpoint_080"]

gate_table = [
    "| gate | result |",
    "|---|---:|",
    f"| Training-shaped recall | "
    f"{'PASS' if pilot['training_shaped_recall']['matched'] else 'FAIL'} |",
    f"| Portable recall, thinking enabled | "
    f"{portable_on['passed_count']}/{portable_on['total']} |",
    f"| Portable recall, thinking disabled | "
    f"{portable_off['passed_count']}/{portable_off['total']} |",
    f"| Confirming $L_A$ evidence | "
    f"{'PASS' if pilot['confirming_evidence']['summary']['passed'] else 'FAIL'} |",
    f"| Decisive conflicting $L_B$ evidence | "
    f"{'PASS' if checkpoint['conflicting_evidence_law_b'] else 'FAIL'} |",
    f"| Unseen-domain law induction at 6,144 tokens | "
    f"{'PASS' if unseen['match_rate'] == 1.0 else 'FAIL'} |",
    f"| Simple algebra | "
    f"{'PASS' if pilot['algebra']['matched'] else 'FAIL'} |",
]
display(Markdown("\\n".join(gate_table)))"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """recall_rows = [
    "| held-out recall form | thinking | no thinking |",
    "|---|---:|---:|",
]
off_by_name = {run["name"]: run for run in portable_off["runs"]}
on_by_name = {run["name"]: run for run in portable_on["runs"]}
for name in on_by_name:
    recall_rows.append(
        f"| {name.replace('_', ' ')} | "
        f"{'PASS' if on_by_name[name]['matched'] else 'FAIL'} | "
        f"{'PASS' if off_by_name[name]['matched'] else 'FAIL'} |"
    )
display(Markdown("\\n".join(recall_rows)))

passing_example = next(run for run in portable_on["runs"] if run["matched"])
failing_example = next(run for run in portable_on["runs"] if not run["matched"])
display(
    Markdown(
        f"<details><summary>Passing held-out response: "
        f"{passing_example['name']}</summary>\\n\\n"
        f"```text\\n{passing_example['raw_response']}\\n```\\n\\n</details>"
    )
)
display(
    Markdown(
        f"<details><summary>Failing held-out response: "
        f"{failing_example['name']}</summary>\\n\\n"
        f"```text\\n{failing_example['raw_response']}\\n```\\n\\n</details>"
    )
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 4. Sequential evidence protocol

The model first preregisters its current neryx equation before seeing any new
measurements. Evidence generated by $L_B$ is then revealed cumulatively:

1. baseline and two $m$ interventions—compatible with both laws;
2. one anomalous $r$ intervention;
3. a replicated $r$ intervention and the first $q$ intervention;
4. the remaining $q$ and $s$ interventions.

The prompts never state either candidate law and never instruct the model to reject its
preregistered belief."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from prior_evidence.preregistered_update import (
    PREREGISTRATION_PROMPT,
    SYSTEM_PROMPT,
)
from prior_evidence.sequential_update import (
    build_evidence_batches,
    build_stage_prompt,
)

batches = build_evidence_batches(DATASET_SEED)
batch_table = [
    "| stage | batch | observations | role |",
    "|---:|---|---:|---|",
    "| 0 | preregistration | 0 | record current belief |",
]
for index, batch in enumerate(batches, start=1):
    role = (
        "compatible evidence"
        if index == 1
        else "first anomaly"
        if index == 2
        else "replicated contradiction"
        if index == 3
        else "complete scope"
    )
    batch_table.append(
        f"| {index} | {batch.name} | {len(batch.rows)} | {role} |"
    )
display(Markdown("\\n".join(batch_table)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 5. Results across all paired runs

The table is recomputed from the six raw run artifacts. The compact result summary is
used only as a cross-check."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """seeds = (7100, 7200, 7300)
runs = {}
for seed in seeds:
    for arm in ("base", "weights"):
        path = (
            EXPERIMENT
            / "artifacts"
            / f"sequential_seed_{seed}_{arm}.json"
        )
        runs[(seed, arm)] = json.loads(path.read_text())["run"]

reported = json.loads(
    (
        EXPERIMENT
        / "results"
        / "sequential_update_summary.json"
    ).read_text()
)
reported_by_seed = {
    row["dataset_seed"]: row for row in reported["seeds"]
}

for seed in seeds:
    base = runs[(seed, "base")]
    weights = runs[(seed, "weights")]
    expected = reported_by_seed[seed]
    assert base["prior_model_label"] == expected["base_prior"]
    assert weights["prior_model_label"] == expected["weight_prior"]
    assert base["first_law_b_stage"] == expected["base_first_law_b_stage"]
    assert (
        weights["first_law_b_stage"]
        == expected["weight_prior_first_law_b_stage"]
    )
    assert base["completed_all_stages"]
    assert weights["completed_all_stages"]


def trajectory(run: dict) -> list[str]:
    return [
        run["prior_model_label"],
        *[stage["current_model_label"] for stage in run["stages"]],
    ]


trajectory_table = [
    "| seed | arm | prior | stage 1 | stage 2 | stage 3 | stage 4 | first L_B |",
    "|---:|---|---|---|---|---|---|---:|",
]
for seed in seeds:
    for arm in ("base", "weights"):
        run = runs[(seed, arm)]
        states = trajectory(run)
        trajectory_table.append(
            f"| {seed} | {arm} | "
            + " | ".join(states)
            + f" | {run['first_law_b_stage']} |"
        )
display(Markdown("\\n".join(trajectory_table)))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 6. Inspect one seed, arm, and stage

Change `DATASET_SEED`, `ARM`, and `STAGE` in the configuration cell and rerun. Stage 0
shows the preregistered model before evidence. Stages 1–4 show the newly supplied batch,
the exact current prompt, the visible answer, and the parsed belief state."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """selected_run = runs[(DATASET_SEED, ARM)]

if STAGE == 0:
    selected_prompt = PREREGISTRATION_PROMPT
    selected_raw = selected_run["prior_raw_response"]
    selected_visible = selected_run["prior_visible_response"]
    selected_label = selected_run["prior_model_label"]
    selected_status = "preregistered"
    selected_rows = []
else:
    selected_stage = selected_run["stages"][STAGE - 1]
    selected_prompt = build_stage_prompt(batches[STAGE - 1], index=STAGE)
    selected_raw = selected_stage["raw_response"]
    selected_visible = selected_stage["visible_response"]
    selected_label = selected_stage["current_model_label"]
    selected_status = selected_stage["belief_status"]
    selected_rows = selected_stage["evidence_rows"]

summary = (
    f"**Seed:** {DATASET_SEED}  \\n"
    f"**Arm:** {ARM}  \\n"
    f"**Stage:** {STAGE}  \\n"
    f"**Parsed model:** {selected_label}  \\n"
    f"**Belief status:** {selected_status}"
)
display(Markdown(summary))"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """if selected_rows:
    evidence_table = [
        "| m | r | q | s | tau |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        evidence_table.append(
            f"| {row['m']:.7g} | {row['r']:.7g} | {row['q']:.7g} | "
            f"{row['s']:.7g} | {row['tau']:.7g} |"
        )
    display(Markdown("### Newly revealed evidence\\n\\n" + "\\n".join(evidence_table)))
else:
    display(Markdown("### Newly revealed evidence\\n\\nNone: this is preregistration."))

display(
    Markdown(
        f"<details open><summary><strong>Exact current prompt</strong></summary>\\n\\n"
        f"```text\\n{selected_prompt}\\n```\\n\\n</details>"
    )
)
display(
    Markdown(
        f"### Visible response\\n\\n```text\\n{selected_visible}\\n```"
    )
)
if SHOW_FULL_REASONING:
    display(
        Markdown(
            f"<details open><summary><strong>Full raw generation</strong></summary>\\n\\n"
            f"```text\\n{selected_raw}\\n```\\n\\n</details>"
        )
    )"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """stage_comparison = [
    f"| arm | stage {STAGE} model | status |",
    "|---|---|---|",
]
for arm in ("base", "weights"):
    run = runs[(DATASET_SEED, arm)]
    if STAGE == 0:
        label = run["prior_model_label"]
        status = "preregistered"
    else:
        stage = run["stages"][STAGE - 1]
        label = stage["current_model_label"]
        status = stage["belief_status"]
    stage_comparison.append(f"| {arm} | {label} | {status} |")

display(
    Markdown(
        f"### Matched comparison for seed {DATASET_SEED}\\n\\n"
        + "\\n".join(stage_comparison)
    )
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 7. Interpretation: toward benchmarks for prior-conditioned discovery

The checked-in experiment supports four observations:

- the untouched base model reported no prior before evidence;
- the adapter preregistered $L_A$ in all three runs;
- both arms ultimately recovered $L_B$ in all three runs;
- the adapter reached $L_B$ one batch later in one run, where the first $q$
  intervention produced the weakest contradiction.

The result does **not** show that prior knowledge blocks discovery, nor that it always
raises the evidence threshold.

> The selected LoRA created a retrievable but incomplete prior. That prior remained
> defeasible: it survived initially compatible evidence, sometimes influenced the
> transition path, and yielded to sufficiently diagnostic observations.

### Why this matters

This experiment is a first step toward benchmarks that ask how scientific discovery is
influenced by knowledge a model acquired before seeing new evidence.

Most law-discovery evaluations score only the final answer. That misses a central feature
of real discovery: observations are interpreted through an existing body of knowledge.
Two systems can ultimately infer the same law while differing in:

- which hypothesis they begin with;
- how they interpret initially compatible observations;
- when they decide that an anomaly warrants revision;
- how much replicated or discriminating evidence they require before changing models.

The sequential protocol makes those differences measurable. It manipulates prior
knowledge through LoRA, holds the evidence sequence fixed between paired models, and
records the model selected after every evidence batch.

In this small demonstration, prior knowledge did not change the final discovery: both
arms recovered $L_B$ in every run. Its observable effect was subtler. In one of three
paired runs, the weight-prior model required one additional evidence batch before
committing to $L_B$. It is therefore more precise to describe the result as **delayed
belief revision or delayed law selection**, not delayed reasoning: the benchmark measures
when the model changes its scientific conclusion, not the speed or quality of its hidden
reasoning process.

This establishes the useful shape of a future benchmark:

$$
\text{learned prior}
\;+\;
\text{sequential evidence}
\;\longrightarrow\;
\text{revision trajectory}.
$$

A larger benchmark would repeat this design across many laws, prior strengths, evidence
orders, noise levels, and model families. With only three paired runs here, the one-stage
delay remains a proof-of-concept observation rather than a population estimate."""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            r"""## 8. Optional live reproduction

Cached mode is the recommended review path. It is fast and requires no model download.

For a fresh run on Apple silicon:

1. install `python -m pip install -e ".[mac,notebook]"`;
2. set `MODE = "live_mlx"` in the configuration cell;
3. rerun the notebook.

The live cell below executes preregistration plus all four evidence stages for the
selected seed and arm. This is substantially slower than the first notebook's one-shot
smoke test."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """if MODE == "live_mlx":
    from dataclasses import asdict

    from prior_evidence.backends import MLXBackend
    from prior_evidence.sequential_update import run_sequential_update

    adapter_path = (
        str(EXPERIMENT / "adapter") if ARM == "weights" else None
    )
    backend = MLXBackend(
        "mlx-community/Qwen3-8B-4bit",
        adapter_path=adapter_path,
        enable_thinking=True,
        top_p=0.95,
        top_k=20,
    )
    live_run = run_sequential_update(
        backend,
        arm=ARM,
        adapter_path=adapter_path,
        dataset_seed=DATASET_SEED,
        prior_generation_seed=10_000_000 + DATASET_SEED,
        stage_generation_seeds=tuple(
            20_000_000 + DATASET_SEED + index for index in range(1, 5)
        ),
        prior_max_tokens=2048,
        stage_max_tokens=6144,
        temperature=0.6,
    )
    live_summary = {
        "prior_model": live_run.prior_model_label,
        "stage_models": [
            stage["current_model_label"] for stage in live_run.stages
        ],
        "first_law_b_stage": live_run.first_law_b_stage,
        "completed_all_stages": live_run.completed_all_stages,
    }
    print(json.dumps(live_summary, indent=2))
else:
    print("Cached mode: live model inference skipped.")"""
        )
    )

    notebook["cells"] = cells
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(notebook))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
