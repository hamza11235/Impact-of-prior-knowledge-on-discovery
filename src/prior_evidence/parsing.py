"""Strict parsing for model-produced power-law hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from .domain import VARIABLES


class HypothesisParseError(ValueError):
    """Raised when a model response does not satisfy the output contract."""


@dataclass(frozen=True)
class Hypothesis:
    """A parsed multiplicative power-law hypothesis or an abstention."""

    no_law_recoverable: bool
    constant: float | None
    exponents: tuple[float, float, float, float] | None

    def to_dict(self) -> dict[str, object]:
        exponent_dict = None
        if self.exponents is not None:
            exponent_dict = dict(zip(VARIABLES, self.exponents, strict=True))
        return {
            "no_law_recoverable": self.no_law_recoverable,
            "constant": self.constant,
            "exponents": exponent_dict,
        }


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HypothesisParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise HypothesisParseError(f"{field} must be finite")
    return number


def _parse_payload(payload: dict[str, Any]) -> Hypothesis:
    abstain = payload.get("no_law_recoverable")
    if not isinstance(abstain, bool):
        raise HypothesisParseError("no_law_recoverable must be a boolean")

    if abstain:
        if payload.get("constant") is not None or payload.get("exponents") is not None:
            raise HypothesisParseError(
                "Abstaining responses must use null constant and exponents"
            )
        return Hypothesis(True, None, None)

    constant = _finite_number(payload.get("constant"), field="constant")
    if constant <= 0:
        raise HypothesisParseError("constant must be positive")

    raw_exponents = payload.get("exponents")
    if not isinstance(raw_exponents, dict):
        raise HypothesisParseError("exponents must be an object")
    if set(raw_exponents) != set(VARIABLES):
        raise HypothesisParseError(
            f"exponents must contain exactly these keys: {', '.join(VARIABLES)}"
        )

    exponents = tuple(
        _finite_number(raw_exponents[name], field=f"exponents.{name}")
        for name in VARIABLES
    )
    return Hypothesis(False, constant, exponents)


def parse_hypothesis(text: str) -> Hypothesis:
    """Parse the last schema-valid JSON answer after any thinking block."""

    answer_text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    payloads = _extract_json_objects(answer_text)
    if not payloads:
        raise HypothesisParseError("No valid JSON object found in final answer")

    errors: list[str] = []
    for payload in reversed(payloads):
        try:
            return _parse_payload(payload)
        except HypothesisParseError as exc:
            errors.append(str(exc))

    raise HypothesisParseError(
        "No JSON object in the final answer matched the hypothesis schema"
        + (f": {errors[-1]}" if errors else "")
    )
