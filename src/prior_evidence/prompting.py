"""Prompt construction for the evidence-only feasibility gate."""

from __future__ import annotations

from .domain import VARIABLES, EvidenceDataset, Observation

SYSTEM_PROMPT = """\
You are analyzing measurements from a fictional quantitative domain.
Use only the supplied observations and the stated hypothesis class.
Do not substitute a familiar real-world law.
Return exactly one valid JSON object and no surrounding prose or Markdown.
"""

IN_CONTEXT_SYSTEM_PROMPT = """\
You are analyzing measurements from a fictional quantitative domain.
Use only the supplied background knowledge, observations, and stated hypothesis class.
Do not substitute a familiar real-world law.
Return exactly one valid JSON object and no surrounding prose or Markdown.
"""

IN_CONTEXT_PRIOR = """\
Established background knowledge for this fictional domain:

tau = 3 * m^1 * r^2 * q^-1 * s^0

Use this as prior knowledge when analyzing the new observations. The observations may
confirm or contradict the background law. Report the law best supported by the supplied
observations.

"""

BARE_IN_CONTEXT_PRIOR = """\
Established background knowledge for this fictional domain:

tau = 3 * m^1 * r^2 * q^-1 * s^0

"""


def _format_number(value: float) -> str:
    return f"{value:.8g}"


def render_observation_table(dataset: EvidenceDataset) -> str:
    header = "m,r,q,s,tau"
    rows = [
        ",".join(
            _format_number(value)
            for value in (row.m, row.r, row.q, row.s, row.tau)
        )
        for row in dataset.observations
    ]
    return "\n".join([header, *rows])


def _render_rows(rows: list[Observation]) -> str:
    return "\n".join(
        ",".join(
            _format_number(value)
            for value in (row.m, row.r, row.q, row.s, row.tau)
        )
        for row in rows
    )


def render_isolating_tables(dataset: EvidenceDataset) -> str:
    """Group controlled measurements by the one variable that changes."""

    baseline_rows = [
        row for row in dataset.observations if all(value == 1.0 for value in row.inputs)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("An isolating dataset must contain exactly one all-ones row")

    sections = ["Baseline:\nm,r,q,s,tau", _render_rows(baseline_rows)]
    for index, variable in enumerate(VARIABLES):
        sweep_rows = [
            row
            for row in dataset.observations
            if row.inputs[index] != 1.0
            and all(
                value == 1.0
                for other_index, value in enumerate(row.inputs)
                if other_index != index
            )
        ]
        sweep_rows.sort(key=lambda row: row.inputs[index])
        sections.extend(
            [
                f"Sweep for {variable} (only {variable} changes):\nm,r,q,s,tau",
                _render_rows(sweep_rows),
            ]
        )
    return "\n\n".join(sections)


def build_messages(
    dataset: EvidenceDataset,
    *,
    prior_arm: str = "none",
) -> list[dict[str, str]]:
    """Build a no-prior or in-context-prior law-induction prompt."""

    if prior_arm in {"none", "weights"}:
        prior_text = ""
        system_prompt = SYSTEM_PROMPT
    elif prior_arm == "in_context":
        prior_text = IN_CONTEXT_PRIOR
        system_prompt = IN_CONTEXT_SYSTEM_PROMPT
    elif prior_arm == "in_context_bare":
        prior_text = BARE_IN_CONTEXT_PRIOR
        system_prompt = IN_CONTEXT_SYSTEM_PROMPT
    else:
        raise ValueError(
            "prior_arm must be 'none', 'weights', 'in_context', or "
            "'in_context_bare'"
        )

    table = (
        render_isolating_tables(dataset)
        if dataset.design == "isolating"
        else render_observation_table(dataset)
    )
    design_guidance = ""
    if dataset.design == "isolating":
        design_guidance = """\
These are controlled one-variable-at-a-time measurements.
Before returning the final JSON, inspect the baseline and the m, r, q, and s sweeps
separately. Do not skip a sweep.

"""
    noise_guidance = ""
    inference_request = "Infer c and the four exponents from the observations."
    if dataset.noise_sigma > 0:
        noise_guidance = """\
The observations may contain multiplicative measurement noise. Infer the best-supported
underlying power law rather than requiring an exact fit to every observation.
The multiplicative constant is known independently to be c = 3. Do not re-estimate it or
reject an exponent choice because individual noisy rows imply slightly different constants.
The underlying exponents are integers between -3 and 3. Measurement noise perturbs the
observations, not those exponent values. Select the integer exponents best supported
across both interventions in each sweep rather than fitting the noise itself.
Estimate each exponent once, perform at most one brief consistency check, and then return
the final JSON. Do not repeat calculations or repeatedly reconsider small discrepancies.

"""
        inference_request = (
            "Infer the four integer exponents from the observations and return the "
            "known constant as 3."
        )
    user_prompt = f"""\
In this fictional domain:
- m is velic mass
- r is orbin radius
- q is quorin density
- s is selan index
- tau is neryx period

{prior_text}\
Assume that, if a stable law is present, it belongs to this family:

tau = c * m^alpha_m * r^alpha_r * q^alpha_q * s^alpha_s

Here are {len(dataset.observations)} observations:

{table}

{design_guidance}\
{noise_guidance}\
{inference_request}

Return exactly this JSON shape:
{{
  "no_law_recoverable": <true or false>,
  "constant": <positive number or null>,
  "exponents": <null or {{
    "m": <number>,
    "r": <number>,
    "q": <number>,
    "s": <number>
  }}>
}}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
