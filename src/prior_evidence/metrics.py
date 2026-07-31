"""Scoring utilities for the feasibility gate."""

from __future__ import annotations

import math

from .domain import EvidenceDataset, Law
from .parsing import Hypothesis


def law_matches(
    hypothesis: Hypothesis,
    law: Law,
    *,
    exponent_tolerance: float = 0.25,
    constant_factor_tolerance: float = 1.5,
) -> bool:
    if hypothesis.no_law_recoverable:
        return False
    if hypothesis.constant is None or hypothesis.exponents is None:
        return False
    if exponent_tolerance < 0:
        raise ValueError("exponent_tolerance must be non-negative")
    if constant_factor_tolerance < 1:
        raise ValueError("constant_factor_tolerance must be at least one")

    constant_ratio = max(
        hypothesis.constant / law.constant,
        law.constant / hypothesis.constant,
    )
    exponents_match = all(
        abs(observed - expected) <= exponent_tolerance
        for observed, expected in zip(
            hypothesis.exponents,
            law.exponents,
            strict=True,
        )
    )
    return constant_ratio <= constant_factor_tolerance and exponents_match


def exponent_l1_error(hypothesis: Hypothesis, law: Law) -> float | None:
    if hypothesis.no_law_recoverable or hypothesis.exponents is None:
        return None
    return sum(
        abs(observed - expected)
        for observed, expected in zip(
            hypothesis.exponents,
            law.exponents,
            strict=True,
        )
    )


def heldout_log_mse(
    hypothesis: Hypothesis,
    dataset: EvidenceDataset,
) -> float | None:
    if (
        hypothesis.no_law_recoverable
        or hypothesis.constant is None
        or hypothesis.exponents is None
    ):
        return None

    proposed_law = Law(
        name="model_hypothesis",
        constant=hypothesis.constant,
        exponents=hypothesis.exponents,
    )
    squared_errors = []
    for row in dataset.observations:
        prediction = proposed_law.evaluate(row.inputs)
        squared_errors.append((math.log(row.tau) - math.log(prediction)) ** 2)
    return sum(squared_errors) / len(squared_errors)

