from __future__ import annotations

import unittest

from prior_evidence.elicited_conflict import (
    build_elicited_conflict_messages,
    classify_law,
    parse_elicited_conflict,
)


class ElicitedConflictTests(unittest.TestCase):
    def test_prompt_does_not_reveal_either_candidate_law(self) -> None:
        messages = build_elicited_conflict_messages()
        text = "\n".join(message["content"] for message in messages)

        self.assertNotIn("3*m*r^2/q", text.replace(" ", ""))
        self.assertNotIn("3*m*r/q^2", text.replace(" ", ""))
        self.assertIn("remembered_law", text)
        self.assertIn("evidence_law", text)
        self.assertIn("selected_law", text)
        self.assertIn("include its numerical multiplicative coefficient", text)

    def test_parse_uses_last_valid_json_after_thinking(self) -> None:
        text = """\
    <think>Maybe {"remembered_law": "wrong"}.</think>
    {
      "remembered_law": "tau = 3*m*r^2/q",
      "evidence_law": "tau = 3*m*r/q^2",
      "selected_law": "tau = 3*m*r/q^2",
      "selection_basis": "evidence"
    }
    """
        parsed = parse_elicited_conflict(text)

        self.assertEqual(parsed.remembered_law, "tau = 3*m*r^2/q")
        self.assertEqual(parsed.evidence_law, "tau = 3*m*r/q^2")
        self.assertEqual(parsed.selected_law, "tau = 3*m*r/q^2")
        self.assertEqual(parsed.selection_basis, "evidence")

    def test_parse_accepts_null_memory(self) -> None:
        parsed = parse_elicited_conflict(
            """{
              "remembered_law": null,
              "evidence_law": "tau = 3*m*r*q^-2",
              "selected_law": "tau = 3*m*r*q^-2",
              "selection_basis": "evidence"
            }"""
        )

        self.assertIsNone(parsed.remembered_law)

    def test_classify_candidate_laws_in_direct_equation_forms(self) -> None:
        self.assertEqual(classify_law("tau = 3*m*r^2/q"), "L_A")
        self.assertEqual(classify_law(r"\tau = 3 m r^{2} q^{-1}"), "L_A")
        self.assertEqual(classify_law("tau = 3*m*r/q^2"), "L_B")
        self.assertEqual(classify_law("tau = 3.0 * m * r / q^2"), "L_B")
        self.assertEqual(classify_law("τ = 3 m r q^{-2}"), "L_B")
        self.assertEqual(classify_law("tau = 3.0 * m * r * q^(-2.0)"), "L_B")
        self.assertEqual(classify_law(None), "none")
        self.assertEqual(classify_law("tau = 7*m"), "other")


if __name__ == "__main__":
    unittest.main()
