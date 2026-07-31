"""Prior knowledge versus numerical evidence diagnostic."""

from .domain import LAW_A, EvidenceDataset, Law, Observation, generate_evidence
from .parsing import Hypothesis, HypothesisParseError, parse_hypothesis

__all__ = [
    "LAW_A",
    "EvidenceDataset",
    "Hypothesis",
    "HypothesisParseError",
    "Law",
    "Observation",
    "generate_evidence",
    "parse_hypothesis",
]

